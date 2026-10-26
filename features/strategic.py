"""
Strategic tier (2-24h): day-ahead features. One row per (airport, D): what
is known at the END of day D, used to predict whether D+1 is a GDP day
(features/strategic_label.py).

Person B's original design (Step 2) used day D-1's actuals in row D while
the label is D+1 - a one-day gap that skipped day D's actuals, which are
complete by the end of D. Revised 2026-10 so row D carries day D. Also
moved to the shared GDP-day definition (common/disruption.py) and the
Parquet caches (common/aspm_cache.py).

Feature groups:
1. Target-day calendar: cyclical day-of-year and day-of-week of D+1.
2. Day-D operations (ASPM Airport Analysis, airport-hour -> daily mean):
   on-time rates, delays, taxi times.
3. Schedule: scheduled arrivals/departures for D and D+1. Schedules are
   published in advance, so D+1's are known at prediction time; demand
   against capacity is what triggers a GDP.
4. Day-D disruption: share of arrivals with an EDCT, GDP-day flag, and the
   GDP-day rate over the past 7 days (through D).
5. Day-D weather (METAR, observed): daily means plus the extremes that drive
   GDPs - min visibility, max wind, max gust, share of reports with
   thunderstorms. METAR dates are UTC; ASPM dates are local - a known
   misalignment of a few hours.
6. Climatology: train-period GDP-day rate for D+1's calendar month.

Build the cached dataset (after `python -m common.aspm_cache airport_analysis`):
    python -m features.strategic
"""

import numpy as np
import pandas as pd

from common.aspm_cache import load_airport_analysis_cache
from common.data_loader import DATA_DIR, load_weather
from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD, daily_arrival_edct_share, gdp_days
from common.temporal_split import TRAIN_END, VALIDATION_END, assign_temporal_split
from features.planning import eligible_airports
from features.strategic_label import build_strategic_labels

OPERATIONAL_NUMERIC_COLS = [
    "pct_ontime_gate_dep", "pct_ontime_airport_dep", "pct_ontime_gate_arr",
    "avg_gate_dep_delay", "avg_taxi_out_time", "avg_taxi_out_delay",
    "avg_airport_dep_delay", "avg_airborne_delay", "avg_taxi_in_delay",
    "avg_block_delay", "avg_gate_arr_delay",
]
WEATHER_NUMERIC_COLS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "alti"]

CALENDAR_COLUMNS = ["target_doy_sin", "target_doy_cos", "target_dow_sin", "target_dow_cos"]
OPS_COLUMNS = [f"ops_{c}" for c in OPERATIONAL_NUMERIC_COLS]
SCHEDULE_COLUMNS = ["sched_arr_today", "sched_dep_today", "sched_arr_tomorrow", "sched_dep_tomorrow"]
DISRUPTION_COLUMNS = ["arr_edct_share_today", "gdp_day_today", "gdp_rate_7d"]
WEATHER_COLUMNS = ([f"wx_{c}" for c in WEATHER_NUMERIC_COLS]
                   + ["wx_min_vsby", "wx_max_sknt", "wx_max_gust", "wx_ts_share"])
CLIMO_COLUMNS = ["climo_gdp_day_rate_target"]
FEATURE_COLUMNS = (CALENDAR_COLUMNS + OPS_COLUMNS + SCHEDULE_COLUMNS + DISRUPTION_COLUMNS
                   + WEATHER_COLUMNS + CLIMO_COLUMNS)

DAILY_PATH = DATA_DIR / "cache" / "strategic_daily_features.parquet"
LABELS_PATH = DATA_DIR / "cache" / "strategic_labels.parquet"


def _daily_operations() -> pd.DataFrame:
    aa = load_airport_analysis_cache(["airport", "date", "scheduled_arrivals", "scheduled_departures"]
                                     + OPERATIONAL_NUMERIC_COLS)
    for c in OPERATIONAL_NUMERIC_COLS + ["scheduled_arrivals", "scheduled_departures"]:
        aa[c] = pd.to_numeric(aa[c], errors="coerce")
    g = aa.groupby(["airport", "date"])
    out = g[OPERATIONAL_NUMERIC_COLS].mean()
    out.columns = OPS_COLUMNS
    out["sched_arr"] = g["scheduled_arrivals"].sum()
    out["sched_dep"] = g["scheduled_departures"].sum()
    return out.reset_index()


def _daily_weather(airport: str) -> pd.DataFrame:
    wx = load_weather(airport)
    num = wx[WEATHER_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    num["date"] = wx["date"].to_numpy()
    num["ts"] = wx["wxcodes"].astype(str).str.contains("TS", na=False).astype(float)
    g = num.groupby("date")
    out = g[WEATHER_NUMERIC_COLS].mean()
    out.columns = [f"wx_{c}" for c in WEATHER_NUMERIC_COLS]
    out["wx_min_vsby"] = g["vsby"].min()
    out["wx_max_sknt"] = g["sknt"].max()
    out["wx_max_gust"] = g["gust"].max()
    out["wx_ts_share"] = g["ts"].mean()
    return out


def build_strategic_features(airport: str, calendar: pd.DatetimeIndex, ops: pd.DataFrame,
                             threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
                             train_end: str = TRAIN_END) -> pd.DataFrame:
    """Feature rows for every date in `calendar` (row D = known at end of D)."""
    target = calendar + pd.Timedelta(days=1)
    out = pd.DataFrame({"airport": airport, "date": calendar})
    doy = target.dayofyear.to_numpy()
    days_in_year = np.where(target.is_leap_year, 366, 365)
    out["target_doy_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    out["target_doy_cos"] = np.cos(2 * np.pi * doy / days_in_year)
    out["target_dow_sin"] = np.sin(2 * np.pi * target.dayofweek.to_numpy() / 7)
    out["target_dow_cos"] = np.cos(2 * np.pi * target.dayofweek.to_numpy() / 7)

    o = ops[ops["airport"] == airport].set_index("date").reindex(calendar)
    for c in OPS_COLUMNS:
        out[c] = o[c].to_numpy()
    out["sched_arr_today"] = o["sched_arr"].to_numpy()
    out["sched_dep_today"] = o["sched_dep"].to_numpy()
    out["sched_arr_tomorrow"] = o["sched_arr"].shift(-1).to_numpy()
    out["sched_dep_tomorrow"] = o["sched_dep"].shift(-1).to_numpy()

    share = daily_arrival_edct_share()[airport].reindex(calendar)
    gdp = gdp_days(threshold)[airport].reindex(calendar)
    out["arr_edct_share_today"] = share.to_numpy()
    out["gdp_day_today"] = gdp.to_numpy()
    out["gdp_rate_7d"] = gdp.rolling(7, min_periods=1).mean().to_numpy()

    wx = _daily_weather(airport).reindex(calendar)
    for c in WEATHER_COLUMNS:
        out[c] = wx[c].to_numpy()

    train = gdp.loc[:train_end]
    month_rate = train.groupby(train.index.month).mean()
    out["climo_gdp_day_rate_target"] = target.month.map(month_rate).to_numpy()
    return out


def build_strategic_dataset(airports: list[str], threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD):
    """Returns (daily, labels): features for every airport x calendar day
    (sequence/graph models need complete history windows), and the
    labelable rows (airport, date, label, split)."""
    calendar = gdp_days(threshold).index
    ops = _daily_operations()
    daily, labels = [], []
    for airport in airports:
        daily.append(build_strategic_features(airport, calendar, ops, threshold))
        labels.append(build_strategic_labels(airport, calendar.to_series().reset_index(drop=True), threshold))
    labels = pd.concat(labels, ignore_index=True)
    labels["split"] = assign_temporal_split(labels["date"], TRAIN_END, VALIDATION_END).values
    return pd.concat(daily, ignore_index=True), labels


if __name__ == "__main__":
    import time

    t = time.time()
    airports = eligible_airports()
    daily, labels = build_strategic_dataset(airports)
    daily.to_parquet(DAILY_PATH, index=False)
    labels.to_parquet(LABELS_PATH, index=False)
    print(f"{len(airports)} airports: {len(daily):,} daily feature rows, {len(labels):,} labeled rows "
          f"in {time.time() - t:.0f}s")
