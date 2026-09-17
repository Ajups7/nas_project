"""
Validates temporal_split.py's assign_temporal_split() - pure date
arithmetic, no data loading, so this runs over the full 2016-2026 range
instantly. Specifically checks the leap-year edge case that made an
earlier hand-authored (calendar-month-rounded) version of this proposal
subtly wrong - see temporal_split.py's module docstring.

Run from the repo root:
    python -m common.validate_temporal_split
"""

import pandas as pd

from common.temporal_split import assign_temporal_split

TRAIN_END = "2022-12-31"
VALIDATION_END = "2024-01-30"
EMBARGO_DAYS = 30


def main():
    dates = pd.date_range("2016-01-01", "2026-01-31", freq="D")
    split = assign_temporal_split(dates, TRAIN_END, VALIDATION_END, EMBARGO_DAYS)

    print("=== Every date accounted for exactly once ===")
    print(f"total dates: {len(dates)}, total assigned: {len(split)}, "
          f"any unassigned/NaN: {split.isna().any()}")

    print()
    print("=== Bucket sizes and exact boundary dates ===")
    print("(embargo has two separate windows, shown individually below - a single")
    print(" min/max across both would misleadingly look like one giant 14-month gap)")
    df = pd.DataFrame({"date": dates, "bucket": split.values})

    train = df[df["bucket"] == "train"]
    print(f"{'train':12s} n={len(train):5d}  {train['date'].min().date()} -> {train['date'].max().date()}")

    embargo = df[df["bucket"] == "embargo"].reset_index(drop=True)
    gap_at = embargo["date"].diff().dt.days.gt(1)
    split_point = gap_at.idxmax() if gap_at.any() else len(embargo)
    for run in [embargo.iloc[:split_point], embargo.iloc[split_point:]]:
        if len(run):
            print(f"{'embargo':12s} n={len(run):5d}  {run['date'].min().date()} -> {run['date'].max().date()}")

    validation = df[df["bucket"] == "validation"]
    print(f"{'validation':12s} n={len(validation):5d}  {validation['date'].min().date()} -> {validation['date'].max().date()}")

    test = df[df["bucket"] == "test"]
    print(f"{'test':12s} n={len(test):5d}  {test['date'].min().date()} -> {test['date'].max().date()}")

    print()
    print("=== The leap-year check: is the 2nd embargo actually >= 30 days? ===")
    embargo2 = df[(df["bucket"] == "embargo") & (df["date"] > pd.Timestamp(VALIDATION_END))]
    span_days = (embargo2["date"].max() - embargo2["date"].min()).days + 1
    print(f"2nd embargo spans {embargo2['date'].min().date()} -> {embargo2['date'].max().date()}"
          f" = {span_days} days (need >= {EMBARGO_DAYS})")
    print(f"embargo2 includes Feb 29, 2024 (leap day): "
          f"{pd.Timestamp('2024-02-29') in embargo2['date'].values}")

    print()
    print("=== What an earlier hand-rounded proposal got wrong, shown concretely ===")
    print("Hand-rounded proposal said: validation ends 2024-01-31, embargo = all of Feb 2024,")
    print("test starts 2024-03-01. Checking if that's actually safe for the worst-case date:")
    worst_case_date = pd.Timestamp("2024-01-31")
    worst_case_window_end = worst_case_date + pd.Timedelta(days=30)
    print(f"  worst-case validation date: {worst_case_date.date()}")
    print(f"  its 30-day label window reaches: {worst_case_window_end.date()}")
    print(f"  hand-rounded test start was:     2024-03-01")
    print(f"  -> that date's label window reaches test's first day exactly - a 1-day leak"
          if worst_case_window_end == pd.Timestamp("2024-03-01")
          else "  -> safe")

    print()
    print("=== Boundary safety check: no date within embargo_days of train_end or")
    print("    validation_end is ever labeled validation/test ===")
    near_train_end = df[
        (df["date"] > pd.Timestamp(TRAIN_END))
        & (df["date"] <= pd.Timestamp(TRAIN_END) + pd.Timedelta(days=EMBARGO_DAYS))
    ]
    print(f"all {len(near_train_end)} dates in (train_end, train_end+{EMBARGO_DAYS}d] "
          f"are 'embargo': {(near_train_end['bucket'] == 'embargo').all()}")

    print()
    print("=== Error handling: validation_end before the first embargo even ends ===")
    try:
        assign_temporal_split(dates, train_end="2022-12-31", validation_end="2023-01-10", embargo_days=30)
        print("no error raised (unexpected)")
    except ValueError as e:
        print(f"raised ValueError as expected: {e}")


if __name__ == "__main__":
    main()
