"""
Validates features/tactical.py's hourly tensor against the raw sources:
UTC alignment, GDP-hour flags, the 2-hour label, weather timing, and that
no feature looks into the future (scheduled counts excepted - schedules
are published in advance).

Run from the repo root (after `python -m features.tactical`):
    python -m features.validate_tactical_features
"""

import numpy as np
import pandas as pd

from common.aspm_cache import load_airport_analysis_cache
from common.data_loader import load_weather
from common.disruption import gdp_hours
from common.temporal_split import TRAIN_END, VALIDATION_END, assign_temporal_split
from features.tactical import TENSOR_PATH


def ok(flag) -> str:
    return "OK" if bool(flag) else "MISMATCH"


def main():
    z = np.load(TENSOR_PATH, allow_pickle=False)
    X, label, gdp = z["X"], z["label"], z["gdp_hour"]
    airports, grid, names = list(z["airports"]), pd.DatetimeIndex(z["grid"]), list(z["feature_names"])
    col = {n: i for i, n in enumerate(names)}
    a = airports.index("EWR")

    print("=== 1. UTC alignment: EWR 2019-07-15 17:00 local (EDT, UTC-4) is grid 21:00 UTC ===")
    t = grid.get_loc(pd.Timestamp("2019-07-15 21:00"))
    aa = load_airport_analysis_cache(["airport", "date", "local_hour", "scheduled_arrivals"])
    raw = aa[(aa.airport == "EWR") & (aa.date == "2019-07-15") & (aa.local_hour == 17)]["scheduled_arrivals"]
    print(f"  sched_arr_t={X[a, t, col['sched_arr_t']]:.0f}  raw local 17:00={float(raw.iloc[0]):.0f}  "
          f"[{ok(X[a, t, col['sched_arr_t']] == float(raw.iloc[0]))}]")
    hod = np.arctan2(X[a, t, col['hod_sin']], X[a, t, col['hod_cos']]) * 24 / (2 * np.pi) % 24
    print(f"  local hour from hod features={hod:.1f}  [{ok(round(hod) == 17)}]")

    print("\n=== 2. GDP-hour flags match common.disruption.gdp_hours() ===")
    gh = gdp_hours()
    day = gh[(gh.airport == "EWR") & (gh.date == "2019-07-15")].sort_values("local_hour")
    t0 = grid.get_loc(pd.Timestamp("2019-07-15 04:00"))  # local midnight
    ours = gdp[a, t0:t0 + 24][day["local_hour"].to_numpy()]
    print(f"  EWR 2019-07-15 GDP hours (local): {day.loc[day.gdp_hour == 1, 'local_hour'].tolist()}")
    print(f"  tensor agrees on all {len(day)} hours: [{ok(np.array_equal(ours, day['gdp_hour'].to_numpy()))}]")

    print("\n=== 3. Label = GDP hour at t+1 or t+2 ===")
    m = ~np.isnan(label[:, :-2])
    expect = np.fmax(gdp[:, 1:-1], gdp[:, 2:])[m]  # a known GDP hour decides it even if the other is unknown
    print(f"  {int(m.sum()):,} scored hours checked: [{ok(np.array_equal(label[:, :-2][m], expect))}]")

    print("\n=== 4. Weather is the same UTC hour's METAR ===")
    wx = load_weather("EWR")
    valid = pd.to_datetime(wx["date"]) + pd.to_timedelta(wx["gmt_hour"], unit="h")
    v = pd.to_numeric(wx.loc[valid == pd.Timestamp("2019-07-15 21:00"), "vsby"], errors="coerce").min()
    print(f"  wx_vsby={X[a, t, col['wx_vsby']]:.2f}  METAR min at 21Z={v:.2f}  [{ok(np.isclose(X[a, t, col['wx_vsby']], v))}]")

    print("\n=== 5. No look-ahead: perturbing hour t+1 must not change features at t (schedules excepted) ===")
    gdp_now = X[a, t, col["gdp_hour_t"]]
    print(f"  gdp_hour_t at t equals gdp[t] (not t+1): [{ok(gdp_now == gdp[a, t])}]")
    m3 = X[a, t, col["gate_arr_delay_m3"]]
    raw3 = np.nanmean(X[a, t - 2:t + 1, col["gate_arr_delay"]])
    print(f"  3-h mean uses t-2..t: {m3:.3f} vs {raw3:.3f}  [{ok(np.isclose(m3, raw3))}]")

    print("\n=== 6. Dataset ===")
    split = assign_temporal_split(pd.Series(grid.normalize()), TRAIN_END, VALIDATION_END).to_numpy()
    scored = ~np.isnan(label)
    onset = scored & (gdp == 0)
    for s in ("train", "validation", "test"):
        cols = split == s
        sc, on = scored[:, cols], onset[:, cols]
        print(f"  {s:10s} scored {int(sc.sum()):>9,}  prevalence {np.nanmean(label[:, cols]):.4f}   "
              f"onset rows {int(on.sum()):>9,}  onset prevalence {label[:, cols][on].mean():.4f}")
    pos = scored & (label == 1)
    print(f"  share of positives that are onsets (not in GDP now): {(pos & (gdp == 0)).sum() / pos.sum():.3f}")
    tr = scored & (split == "train")[None, :]
    flat = X[tr]
    y = label[tr]
    miss = np.isnan(flat).mean(axis=0)
    print("  missing (train):", {n: round(float(x), 3) for n, x in zip(names, miss) if x > 0})
    cor = [pd.Series(flat[:, i]).corr(pd.Series(y)) for i in range(len(names))]
    print("  top correlations with label (train):",
          {names[i]: round(cor[i], 3) for i in np.argsort(np.nan_to_num(np.abs(cor)))[::-1][:10]})


if __name__ == "__main__":
    main()
