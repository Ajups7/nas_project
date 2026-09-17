"""
Validates planning_label.py against REAL downloaded EDCT data, same
philosophy as validate_planning_features.py - no synthetic inputs, because
real data exists to check this against.

Dates chosen straddle a month boundary (late June into early July 2019) on
purpose: each date's 30-day forward window needs EDCT from 2+ calendar
months, so this also exercises the multi-month loading path, not just the
single-month case.

Run from the repo root:
    python -m features.validate_planning_label
"""

import time

import pandas as pd

from common.data_loader import load_edct
from features.planning_label import build_planning_labels

AIRPORT = "ATL"
DATES = pd.to_datetime(["2019-06-20", "2019-06-25", "2019-06-30", "2019-07-05"])


def independent_check(airport: str, date: pd.Timestamp, cache: dict, horizon_days: int = 30) -> int:
    """Recomputes one date's label a completely different way than
    planning_label.py does - loading months directly and filtering with a
    single boolean mask, rather than the groupby/reindex path the real
    function uses - to catch a bug that might be shared between "the
    function" and "the same logic written slightly differently again."

    `cache` is a plain dict shared across calls, keyed by (year, month):
    each EDCT month costs ~13s to load, and the 4 validation dates here
    overlap on 2019-07, so without caching this would reload it twice.
    """
    window_start = date + pd.Timedelta(days=1)
    window_end = date + pd.Timedelta(days=horizon_days)
    months = sorted({
        (window_start.year, window_start.month),
        (window_end.year, window_end.month),
    })
    frames = []
    for year, month in months:
        if (year, month) not in cache:
            edct = load_edct(year, month)
            cache[(year, month)] = edct[edct["airport"] == airport]
        frames.append(cache[(year, month)])
    combined = pd.concat(frames, ignore_index=True)
    in_window = (combined["date"] >= window_start) & (combined["date"] <= window_end)
    had_edct = (combined["arrivals_with_edct"] > 0) | (combined["departures_with_edct"] > 0)
    return int((in_window & had_edct).any())


def main():
    print(f"=== build_planning_labels for {AIRPORT}, {len(DATES)} dates spanning a month boundary ===")
    t0 = time.time()
    labels = build_planning_labels(AIRPORT, DATES)
    elapsed = time.time() - t0
    print(f"(computed in {elapsed:.1f}s)")
    print(labels.to_string(index=False))

    print()
    print("=== Independent cross-check (different code path, same 4 dates) ===")
    cache: dict = {}
    for date in DATES:
        expected = independent_check(AIRPORT, date, cache)
        actual = labels.loc[labels["date"] == date, "label"].iloc[0]
        match = "OK" if expected == actual else "MISMATCH"
        print(f"  {date.date()}: build_planning_labels={actual}  independent_check={expected}  [{match}]")

    print()
    print("=== Edge case: horizon_days=1 (shortest real window) at ATL ===")
    single_day = pd.to_datetime(["2019-06-20"])
    edge_labels = build_planning_labels(AIRPORT, single_day, horizon_days=1)
    print(edge_labels.to_string(index=False))
    print("(all-1s so far - ATL is a busy hub, so this alone doesn't prove the")
    print(" label can ever come back 0. Checking a quiet airport next.)")

    print()
    print("=== Does the label ever actually return 0? Same dates, quiet airport (ADK) ===")
    quiet_labels = build_planning_labels("ADK", DATES, horizon_days=1)
    print(quiet_labels.to_string(index=False))


if __name__ == "__main__":
    main()
