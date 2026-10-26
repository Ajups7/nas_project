"""
Validates planning_label.py against REAL EDCT data.

The independent check recomputes each label from the raw .xls files via
load_edct() with a plain per-day loop - a different code path from both
the Parquet cache and the rolling-window implementation, so it checks both.

Run from the repo root:
    python -m features.validate_planning_label
"""

import pandas as pd

from common.data_loader import load_edct
from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD
from features.planning_label import build_planning_labels
from features.strategic_label import build_strategic_labels

DATES = pd.to_datetime(["2019-06-20", "2019-06-25", "2019-06-30", "2019-07-05"])
CHECK_AIRPORTS = ["EWR", "ATL", "ABQ"]  # ABQ: zero GDP days in 2019, exercises label=0


def independent_label(raw: pd.DataFrame, airport: str, date: pd.Timestamp, horizon_days: int = 30) -> int:
    a = raw[raw["airport"] == airport]
    for k in range(1, horizon_days + 1):
        day = a[a["date"] == date + pd.Timedelta(days=k)]
        arrivals = day["arrivals_for_metric"].sum()
        if arrivals > 0 and day["arrivals_with_edct"].sum() / arrivals >= GDP_ARRIVAL_SHARE_THRESHOLD:
            return 1
    return 0


def main():
    print("=== 1. Cross-check against raw .xls (2019-06 .. 2019-08) ===")
    raw = pd.concat([load_edct(2019, m) for m in (6, 7, 8)], ignore_index=True)
    mismatches = 0
    for airport in CHECK_AIRPORTS:
        labels = build_planning_labels(airport, DATES).set_index("date")["label"]
        for date in DATES:
            expected = independent_label(raw, airport, date)
            actual = int(labels[date])
            mismatches += expected != actual
            print(f"  {airport} {date.date()}: label={actual} independent={expected} "
                  f"[{'OK' if expected == actual else 'MISMATCH'}]")
    print(f"mismatches: {mismatches}")

    print("\n=== 2. Label takes both values (all 2019 dates) ===")
    year = pd.date_range("2019-01-01", "2019-12-31")
    for airport in CHECK_AIRPORTS:
        lab = build_planning_labels(airport, year)["label"]
        print(f"  {airport}: prevalence {lab.mean():.3f} over {len(lab)} dates")

    print("\n=== 3. Missing month 2016-07: a 0 can't be confirmed, a 1 still can ===")
    near_gap = pd.to_datetime(["2016-06-10", "2016-06-20"])
    for airport in ["EWR", "ABQ"]:  # EWR: 1 survives the gap; ABQ: no GDP seen, so dropped
        out = build_planning_labels(airport, near_gap)
        print(f"  {airport}: kept {len(out)}/{len(near_gap)} -> {out[['date', 'label']].to_dict('records')}")

    print("\n=== 4. Window past end of data (2026-01-31) is dropped ===")
    end = pd.to_datetime(["2025-12-31", "2026-01-15"])
    print(f"  EWR horizon 30: kept {len(build_planning_labels('EWR', end))}/2 (expect 1)")

    print("\n=== 5. Strategic wrapper (horizon 1) still works ===")
    s = build_strategic_labels("EWR", year)["label"]
    print(f"  EWR 2019 next-day GDP prevalence: {s.mean():.3f} over {len(s)} dates")

    print("\n=== 6. ASE (in ASPM only from mid-2023): 2019 dates are all unknown, so dropped ===")
    print(f"  ASE 2019 rows kept: {len(build_planning_labels('ASE', year))} (expect 0)")
    print(f"  ASE 2024 rows kept: {len(build_planning_labels('ASE', pd.date_range('2024-03-01', '2024-03-31')))} (expect >0)")

    print("\n=== 7. Airport absent from EDCT entirely raises ===")
    try:
        build_planning_labels("ZZZ", DATES)
        print("  no error (unexpected)")
    except ValueError as e:
        print(f"  ValueError: {e}")


if __name__ == "__main__":
    main()
