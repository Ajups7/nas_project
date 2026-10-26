"""
Validates features/strategic.py against REAL data, with emphasis on timing
(the original version had a one-day gap between features and label).

Run from the repo root (after `python -m features.strategic`):
    python -m features.validate_strategic_features
"""

import numpy as np
import pandas as pd

from common.aspm_cache import load_airport_analysis_cache
from common.disruption import gdp_days
from common.temporal_split import TRAIN_END
from features.strategic import DAILY_PATH, FEATURE_COLUMNS, LABELS_PATH
from models.strategic.data import load_strategic_splits

AIRPORT, DAY = "EWR", pd.Timestamp("2019-07-15")


def ok(flag: bool) -> str:
    return "OK" if flag else "MISMATCH"


def main():
    daily = pd.read_parquet(DAILY_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    row = daily[(daily["airport"] == AIRPORT) & (daily["date"] == DAY)].iloc[0]
    nxt = daily[(daily["airport"] == AIRPORT) & (daily["date"] == DAY + pd.Timedelta(days=1))].iloc[0]

    print(f"=== 1. Row {DAY.date()} carries day {DAY.date()}'s operations (not the day before) ===")
    aa = load_airport_analysis_cache(["airport", "date", "avg_gate_arr_delay", "scheduled_arrivals"])
    aa = aa[aa["airport"] == AIRPORT]
    same_day = pd.to_numeric(aa.loc[aa["date"] == DAY, "avg_gate_arr_delay"]).mean()
    prev_day = pd.to_numeric(aa.loc[aa["date"] == DAY - pd.Timedelta(days=1), "avg_gate_arr_delay"]).mean()
    print(f"  ops_avg_gate_arr_delay={row['ops_avg_gate_arr_delay']:.3f}  same day={same_day:.3f}  "
          f"previous day={prev_day:.3f}  [{ok(np.isclose(row['ops_avg_gate_arr_delay'], same_day))}]")

    print("\n=== 2. Tomorrow's schedule = the next row's own schedule ===")
    print(f"  sched_arr_tomorrow={row['sched_arr_tomorrow']:.0f}  next row sched_arr_today="
          f"{nxt['sched_arr_today']:.0f}  [{ok(row['sched_arr_tomorrow'] == nxt['sched_arr_today'])}]")

    print("\n=== 3. Label is D+1's GDP flag; features hold only D and earlier ===")
    merged = labels.merge(daily[["airport", "date", "gdp_day_today"]], on=["airport", "date"])
    g = gdp_days()
    nextday = np.array([g.at[d + pd.Timedelta(days=1), a] for a, d in
                        zip(merged["airport"].iloc[:20000], merged["date"].iloc[:20000])])
    print(f"  label == gdp_days(D+1) on first 20,000 rows: "
          f"[{ok(np.array_equal(merged['label'].iloc[:20000].to_numpy(), nextday))}]")
    same = (merged["label"] == merged["gdp_day_today"]).mean()
    print(f"  label equals today's GDP flag on {same:.1%} of rows (persistence, not identity)")

    print("\n=== 4. Climatology uses train dates only ===")
    train = g[AIRPORT].loc[:TRAIN_END]
    expect = train[train.index.month == 7].mean()
    print(f"  climo for {AIRPORT}, target in July={row['climo_gdp_day_rate_target']:.4f}  "
          f"by hand={expect:.4f}  [{ok(np.isclose(row['climo_gdp_day_rate_target'], expect))}]")

    print("\n=== 5. Dataset ===")
    print(f"  daily rows {len(daily):,}, labeled rows {len(labels):,}, airports {daily['airport'].nunique()}")
    print(labels.groupby("split")["label"].agg(["size", "mean"]).round(4).to_string())
    miss = daily.loc[daily["date"] <= TRAIN_END, FEATURE_COLUMNS].isna().mean()
    print("  missing before fill (train-period days):")
    print(miss[miss > 0].round(4).to_string())
    tr = load_strategic_splits()["train"]
    print("  correlation with label (train, after fill):")
    print(tr[FEATURE_COLUMNS].corrwith(tr["label"]).round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
