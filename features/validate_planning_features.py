"""
Validates planning.py against REAL data, recomputing each feature group by
hand and checking the leakage guarantee directly.

Run from the repo root (after `python -m features.planning` has built the cache):
    python -m features.validate_planning_features
"""

import numpy as np
import pandas as pd

from common.data_loader import load_weather
from common.disruption import gdp_days
from common.temporal_split import TRAIN_END
from features.planning import (
    ANOMALY_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    build_planning_features,
)

AIRPORT = "SFO"


def main():
    gdp = gdp_days()[AIRPORT]
    train_gdp = gdp.loc[:TRAIN_END]

    print("=== 1. Forward-window climatology: 2019-06-25 covers Jun 26-30 + Jul 1-25 ===")
    month_rate = train_gdp.groupby(train_gdp.index.month).mean()
    expected = (5 * month_rate[6] + 25 * month_rate[7]) / 30
    f = build_planning_features(AIRPORT, pd.to_datetime(["2019-06-25"]))
    print(f"  climo_gdp_day_rate={f['climo_gdp_day_rate'].iloc[0]:.5f}  by hand={expected:.5f}  "
          f"[{'OK' if np.isclose(f['climo_gdp_day_rate'].iloc[0], expected) else 'MISMATCH'}]")

    wx = load_weather(AIRPORT)
    wx = wx[wx["date"] <= TRAIN_END]
    tmpf = pd.to_numeric(wx["tmpf"], errors="coerce").groupby(wx["date"].dt.month).mean()
    expected_t = (5 * tmpf[6] + 25 * tmpf[7]) / 30
    print(f"  climo_tmpf={f['climo_tmpf'].iloc[0]:.3f}  by hand={expected_t:.3f}  "
          f"[{'OK' if np.isclose(f['climo_tmpf'].iloc[0], expected_t) else 'MISMATCH'}]")

    print("\n=== 2. Leakage: climatology must not change if post-train GDPs change ===")
    d = pd.to_datetime(["2024-07-01"])
    before = build_planning_features(AIRPORT, d)["climo_gdp_day_rate"].iloc[0]
    days = gdp_days()
    saved = days.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):, AIRPORT].copy()
    days.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):, AIRPORT] = 1.0
    after = build_planning_features(AIRPORT, d)["climo_gdp_day_rate"].iloc[0]
    days.loc[saved.index, AIRPORT] = saved
    print(f"  2024-07-01 climo before={before:.5f} after setting every post-train day to GDP={after:.5f}  "
          f"[{'OK' if before == after else 'LEAK'}]")

    print("\n=== 3. Recent history: 2019-07-15, past 7 and 30 days through the date itself ===")
    f = build_planning_features(AIRPORT, pd.to_datetime(["2019-07-15"]))
    exp7 = gdp.loc["2019-07-09":"2019-07-15"].mean()
    exp30 = gdp.loc["2019-06-16":"2019-07-15"].mean()
    print(f"  7d={f['recent_gdp_rate_7d'].iloc[0]:.4f} by hand={exp7:.4f}   "
          f"30d={f['recent_gdp_rate_30d'].iloc[0]:.4f} by hand={exp30:.4f}  "
          f"[{'OK' if np.isclose(f['recent_gdp_rate_7d'].iloc[0], exp7) and np.isclose(f['recent_gdp_rate_30d'].iloc[0], exp30) else 'MISMATCH'}]")
    print("  (window ends at the as-of date; the label window starts the day after - no overlap)")
    exp_anom = exp30 - (15 * month_rate[6] + 15 * month_rate[7]) / 30  # Jun 16-30 + Jul 1-15
    got_anom = f["recent_gdp_anom_30d"].iloc[0]
    print(f"  anom30={got_anom:.4f} by hand={exp_anom:.4f}  "
          f"[{'OK' if np.isclose(got_anom, exp_anom) else 'MISMATCH'}]")

    print("\n=== 4. Cached all-airport dataset ===")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  rows {len(df):,}, airports {df['airport'].nunique()}")
    print("  rows / label prevalence by split:")
    print(df.groupby("split")["label"].agg(["size", "mean"]).round(3).to_string())
    print("  missing values per feature (train rows):")
    na = df.loc[df["split"] == "train", FEATURE_COLUMNS].isna().mean()
    print(na[na > 0].round(4).to_string() if (na > 0).any() else "  none")
    tr = df[df["split"] == "train"]
    prev = tr.groupby("airport")["label"].mean()
    print(f"  per-airport train prevalence (median is unstable - bimodal): "
          f"quartiles {prev.quantile([.25, .5, .75]).round(3).tolist()}, "
          f">=0.95: {(prev >= 0.95).sum()}, <=0.05: {(prev <= 0.05).sum()}, "
          f"between: {((prev > 0.05) & (prev < 0.95)).sum()}")
    print("  feature correlation with label (train):")
    print(tr[FEATURE_COLUMNS + ANOMALY_COLUMNS].corrwith(tr["label"]).round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
