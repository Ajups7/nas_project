"""
Tactical tier (0-2 h): hourly features and labels as dense (airport, hour)
arrays on a shared UTC axis, so index t is the same moment at every
airport - which the graph model needs to test simultaneous propagation.

Row (airport, t) = what is known at the end of UTC hour t. Label: a GDP hour
(common/disruption.py: >= 50% of the hour's arrivals with an EDCT, >= 2
held) occurs at t+1 or t+2. Only hours with scheduled arrivals in t+1..t+2
are scored (no arrivals, no GDP possible). The onset subset - hours not
already in a GDP (gdp_hour(t) == 0) - is the operationally important case
(the 90-minute CDM window) and is reported separately.

Rewritten 2026-10 from Person A's per-airport, per-month pandas version
(local-hour sort keys, load_edct()/load_airport_analysis() per month).
Changes: shared GDP-hour definition instead of "any EDCT"; Parquet caches;
UTC axis; Person A's "lightweight rotation signal" (mean arrival delay of
flights landing each hour, from 28 GB of BTS) is replaced by the same
quantity from ASPM Airport Analysis (avg_gate_arr_delay /
avg_airborne_delay per airport-hour); Person A's 3-hour rolling mean/delta
design is kept.

Features (FEATURE_NAMES):
- calendar: local hour-of-day, day-of-week, day-of-year (cyclical)
- schedule: scheduled arrivals/departures at t, t+1, t+2 (published ahead)
- operations at t: arrivals, on-time arrivals, gate/airborne arrival
  delay, taxi-out delay, gate/airport departure delay; 3-h mean and delta
  of arrival, airborne and taxi-out delay
- EDCT state at t: arrival and departure EDCT share (departure EDCTs = GDPs
  at destinations), mean EDCT delay, GDP-hour flag, GDP hours in the last
  3 h and 24 h, 3-h delta of arrival EDCT share
- weather (METAR, UTC hour): visibility, wind, gust, ceiling, thunderstorm,
  temperature-dewpoint spread, altimeter; 3-h deltas of visibility,
  ceiling, altimeter
- climatology: train-period GDP-hour rate for the airport, month and local hour

Build (after both ASPM caches exist):
    python -m features.tactical
"""

import time

import numpy as np
import pandas as pd

from common.aspm_cache import load_airport_analysis_cache, load_edct_cache
from common.data_loader import DATA_DIR, load_weather
from common.disruption import GDP_HOUR_SHARE_THRESHOLD, gdp_hours
from common.temporal_split import TRAIN_END
from features.planning import eligible_airports

TENSOR_PATH = DATA_DIR / "cache" / "tactical_tensor.npz"
GRID = pd.date_range("2016-01-01 00:00", "2026-02-01 12:00", freq="h")
MISSING_EDCT_MONTHS = ("2016-07", "2025-09")

FEATURE_NAMES = [
    "hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "sched_arr_t", "sched_dep_t", "sched_arr_t1", "sched_arr_t2", "sched_dep_t1", "sched_dep_t2",
    "arrivals_t", "pct_ontime_arr", "gate_arr_delay", "airborne_delay", "taxi_out_delay",
    "gate_dep_delay", "airport_dep_delay",
    "gate_arr_delay_m3", "gate_arr_delay_d3", "airborne_delay_m3", "airborne_delay_d3",
    "taxi_out_delay_m3", "taxi_out_delay_d3",
    "arr_edct_share", "dep_edct_share", "edct_delay_gt0", "gdp_hour_t", "gdp_hours_3h", "gdp_hours_24h",
    "arr_edct_share_d3",
    "wx_vsby", "wx_sknt", "wx_gust", "wx_ceiling", "wx_ts", "wx_spread", "wx_alti",
    "wx_vsby_d3", "wx_ceiling_d3", "wx_alti_d3",
    "climo_gdp_hour_rate",
]


def _grid_index(utc: pd.Series) -> np.ndarray:
    return GRID.get_indexer(pd.DatetimeIndex(utc))


def _utc(df: pd.DataFrame) -> pd.Series:
    offset = (df["gmt_hour"].astype(int) - df["local_hour"].astype(int)) % 24
    return df["date"] + pd.to_timedelta(df["local_hour"].astype(int) + offset, unit="h")


def _to_array(df: pd.DataFrame, col: str, ap_index: dict, fill=np.nan) -> np.ndarray:
    out = np.full((len(ap_index), len(GRID)), fill, dtype=np.float32)
    a = df["airport"].map(ap_index).to_numpy()
    t = df["_t"].to_numpy()
    ok = (t >= 0) & ~np.isnan(a.astype(float))
    out[a[ok].astype(int), t[ok]] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)[ok]
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """y[:, t] = x[:, t + k] (k > 0: future, k < 0: past), NaN off the ends."""
    y = np.full_like(x, np.nan)
    if k > 0:
        y[:, :-k] = x[:, k:]
    else:
        y[:, -k:] = x[:, :k]
    return y


def _trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """mean of x[:, t-w+1 .. t], ignoring NaN. Cumulative sums in float64:
    in float32 the rounding drift over ~88k hours is visible by 2019."""
    v = np.nan_to_num(x).astype(np.float64)
    n = (~np.isnan(x)).astype(np.float64)
    cv = np.cumsum(v, axis=1)
    cn = np.cumsum(n, axis=1)
    sv = cv - np.concatenate([np.zeros((x.shape[0], w)), cv[:, :-w]], axis=1)
    sn = cn - np.concatenate([np.zeros((x.shape[0], w)), cn[:, :-w]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sn > 0, sv / sn, np.nan).astype(np.float32)


def _delta3(x: np.ndarray) -> np.ndarray:
    m = _trailing_mean(x, 3)
    return (m - _shift(m, -3)).astype(np.float32)


def _weather(airports: list[str]) -> dict[str, np.ndarray]:
    names = ["vsby", "sknt", "gust", "ceiling", "ts", "spread", "alti"]
    out = {n: np.full((len(airports), len(GRID)), np.nan, np.float32) for n in names}
    for i, airport in enumerate(airports):
        wx = load_weather(airport)
        valid = pd.to_datetime(wx["date"]) + pd.to_timedelta(wx["gmt_hour"], unit="h")
        num = wx[["tmpf", "dwpf", "sknt", "gust", "vsby", "alti"]].apply(pd.to_numeric, errors="coerce")
        ceil = np.full(len(wx), np.nan)
        for k in (3, 2, 1):  # lowest broken/overcast/obscured layer wins
            cover = wx[f"skyc{k}"].astype(str).str.strip()
            height = pd.to_numeric(wx[f"skyl{k}"], errors="coerce").to_numpy()
            layer = cover.isin(["BKN", "OVC", "VV"]).to_numpy()
            ceil = np.where(layer & ~np.isnan(height), height, ceil)
        df = pd.DataFrame({
            "t": GRID.get_indexer(pd.DatetimeIndex(valid)),
            "vsby": num["vsby"], "sknt": num["sknt"], "gust": num["gust"].fillna(num["sknt"]),
            "ceiling": np.where(np.isnan(ceil), 12000.0, np.minimum(ceil, 12000.0)),
            "ts": wx["wxcodes"].astype(str).str.contains("TS", na=False).astype(float),
            "spread": num["tmpf"] - num["dwpf"], "alti": num["alti"],
        })
        df = df[df["t"] >= 0]
        g = df.groupby("t").agg(vsby=("vsby", "min"), sknt=("sknt", "max"), gust=("gust", "max"),
                                ceiling=("ceiling", "min"), ts=("ts", "max"), spread=("spread", "mean"),
                                alti=("alti", "mean"))
        for n in names:
            out[n][i, g.index.to_numpy()] = g[n].to_numpy(dtype=np.float32)
    return out


def build_tensor(share_threshold: float = GDP_HOUR_SHARE_THRESHOLD) -> dict:
    t0 = time.time()
    airports = eligible_airports()
    ap = {a: i for i, a in enumerate(airports)}
    A, T = len(airports), len(GRID)

    aa = load_airport_analysis_cache(["airport", "date", "local_hour", "gmt_hour", "scheduled_arrivals",
                                      "scheduled_departures", "arrivals_for_metric", "pct_ontime_gate_arr",
                                      "avg_gate_arr_delay", "avg_airborne_delay", "avg_taxi_out_delay",
                                      "avg_gate_dep_delay", "avg_airport_dep_delay"])
    aa = aa[aa["airport"].isin(ap)]
    aa["_t"] = _grid_index(_utc(aa))
    local_hour = _to_array(aa, "local_hour", ap)
    offset = np.full((A, T), np.nan, np.float32)
    off = (aa["gmt_hour"].astype(int) - aa["local_hour"].astype(int)) % 24
    offset[aa["airport"].map(ap).to_numpy(), aa["_t"].to_numpy()] = off.to_numpy(dtype=np.float32)
    offset = pd.DataFrame(offset.T).ffill().bfill().to_numpy().T  # hours absent from ASPM
    local_ts = GRID.to_numpy()[None, :] - (offset.astype("int64") * 3600 * 10**9).astype("timedelta64[ns]")
    local_hour = pd.DatetimeIndex(local_ts.ravel()).hour.to_numpy().reshape(A, T)
    local_dow = pd.DatetimeIndex(local_ts.ravel()).dayofweek.to_numpy().reshape(A, T)
    local_doy = pd.DatetimeIndex(local_ts.ravel()).dayofyear.to_numpy().reshape(A, T)
    local_month = pd.DatetimeIndex(local_ts.ravel()).month.to_numpy().reshape(A, T)
    print(f"  ASPM loaded and aligned ({time.time() - t0:.0f}s)", flush=True)

    sched_arr = np.nan_to_num(_to_array(aa, "scheduled_arrivals", ap))
    sched_dep = np.nan_to_num(_to_array(aa, "scheduled_departures", ap))
    f = {}
    f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * local_hour / 24), np.cos(2 * np.pi * local_hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * local_dow / 7), np.cos(2 * np.pi * local_dow / 7)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * local_doy / 365.25), np.cos(2 * np.pi * local_doy / 365.25)
    f["sched_arr_t"], f["sched_dep_t"] = sched_arr, sched_dep
    f["sched_arr_t1"], f["sched_arr_t2"] = _shift(sched_arr, 1), _shift(sched_arr, 2)
    f["sched_dep_t1"], f["sched_dep_t2"] = _shift(sched_dep, 1), _shift(sched_dep, 2)
    f["arrivals_t"] = np.nan_to_num(_to_array(aa, "arrivals_for_metric", ap))
    f["pct_ontime_arr"] = _to_array(aa, "pct_ontime_gate_arr", ap)
    for src, name in (("avg_gate_arr_delay", "gate_arr_delay"), ("avg_airborne_delay", "airborne_delay"),
                      ("avg_taxi_out_delay", "taxi_out_delay"), ("avg_gate_dep_delay", "gate_dep_delay"),
                      ("avg_airport_dep_delay", "airport_dep_delay")):
        f[name] = _to_array(aa, src, ap)
    for name in ("gate_arr_delay", "airborne_delay", "taxi_out_delay"):
        f[f"{name}_m3"], f[f"{name}_d3"] = _trailing_mean(f[name], 3), _delta3(f[name])
    del aa

    e = load_edct_cache(["airport", "date", "local_hour", "gmt_hour", "arrivals_for_metric", "arrivals_with_edct",
                         "departures_for_metric", "departures_with_edct", "avg_edct_arrivals_gt0"])
    e = e[e["airport"].isin(ap)]
    e["_t"] = _grid_index(_utc(e))
    e["arr_share"] = e["arrivals_with_edct"] / e["arrivals_for_metric"].replace(0, np.nan)
    e["dep_share"] = e["departures_with_edct"] / e["departures_for_metric"].replace(0, np.nan)
    known = np.ones((A, T), bool)
    for m in MISSING_EDCT_MONTHS:
        known[:, (GRID.strftime("%Y-%m") == m)] = False
    f["arr_edct_share"] = np.where(known, np.nan_to_num(_to_array(e, "arr_share", ap)), np.nan)
    f["dep_edct_share"] = np.where(known, np.nan_to_num(_to_array(e, "dep_share", ap)), np.nan)
    f["edct_delay_gt0"] = np.where(known, np.nan_to_num(_to_array(e, "avg_edct_arrivals_gt0", ap)), np.nan)
    gh = gdp_hours(share_threshold)
    gh = gh[gh["airport"].isin(ap)].merge(e[["airport", "date", "local_hour", "_t"]],
                                         on=["airport", "date", "local_hour"])
    gdp = np.where(known, np.nan_to_num(_to_array(gh, "gdp_hour", ap)), np.nan).astype(np.float32)
    del e, gh
    f["gdp_hour_t"] = gdp
    f["gdp_hours_3h"] = _trailing_mean(gdp, 3) * 3
    f["gdp_hours_24h"] = _trailing_mean(gdp, 24) * 24
    f["arr_edct_share_d3"] = _delta3(f["arr_edct_share"])
    print(f"  EDCT done ({time.time() - t0:.0f}s)", flush=True)

    wx = _weather(airports)
    for n in ("vsby", "sknt", "gust", "ceiling", "ts", "spread", "alti"):
        f[f"wx_{n}"] = wx[n]
    for n in ("vsby", "ceiling", "alti"):
        f[f"wx_{n}_d3"] = _delta3(wx[n])
    print(f"  weather done ({time.time() - t0:.0f}s)", flush=True)

    train = (GRID <= pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=23))[None, :] & ~np.isnan(gdp)
    climo = np.full((A, T), np.nan, np.float32)
    for i in range(A):
        key = local_month[i] * 24 + local_hour[i]
        s = pd.Series(gdp[i][train[i]]).groupby(key[train[i]]).mean()
        climo[i] = s.reindex(key).to_numpy(dtype=np.float32)
    f["climo_gdp_hour_rate"] = climo

    nxt1, nxt2 = _shift(gdp, 1), _shift(gdp, 2)
    any_gdp = (nxt1 == 1) | (nxt2 == 1)
    all_known = ~np.isnan(nxt1) & ~np.isnan(nxt2)
    label = np.where(any_gdp, 1.0, np.where(all_known, 0.0, np.nan)).astype(np.float32)
    open_ = (f["sched_arr_t1"] + f["sched_arr_t2"]) > 0
    label = np.where(open_, label, np.nan).astype(np.float32)

    X = np.stack([f[n].astype(np.float32) for n in FEATURE_NAMES], axis=2)
    print(f"  tensor {X.shape} built in {time.time() - t0:.0f}s", flush=True)
    return {"X": X, "label": label, "gdp_hour": gdp, "airports": np.array(airports), "grid": GRID.to_numpy(),
            "feature_names": np.array(FEATURE_NAMES)}


if __name__ == "__main__":
    t = build_tensor()
    np.savez(TENSOR_PATH, **t)
    lab = t["label"]
    print(f"saved {TENSOR_PATH}: scored airport-hours {int((~np.isnan(lab)).sum()):,}, "
          f"positive {np.nanmean(lab):.4f}, onset rows (not in GDP now) "
          f"{int(((~np.isnan(lab)) & (t['gdp_hour'] == 0)).sum()):,}")
