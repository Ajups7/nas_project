"""
Validates strategic_label.py against REAL downloaded EDCT data (same
approach as validate_planning_features.py and validate_strategic_features.py).

Runs on airport ATL, June 28 - July 3 2023, deliberately spanning a month
boundary (June 30's label looks at July 1st's EDCT activity, a different
monthly file) to check the underlying build_planning_labels handles a
cross-month window correctly when called with horizon_days=1, and
cross-checks one date's label directly against raw EDCT data rather than
just trusting the wrapper.

Run from the repo root:
    python -m features.validate_strategic_label
"""

import pandas as pd

from common.data_loader import load_edct
from features.strategic_label import build_strategic_labels

AIRPORT = "ATL"
DATES = pd.date_range("2023-06-28", "2023-07-03")


def main():
    print(f"=== Strategic labels (1-day horizon), {AIRPORT}, {DATES.min().date()} - {DATES.max().date()} ===")
    labels = build_strategic_labels(AIRPORT, DATES)
    print(labels.to_string(index=False))

    print()
    print("=== Direct cross-check against raw EDCT data (not trusting the wrapper) ===")
    check_date = pd.Timestamp("2023-06-30")
    next_day = check_date + pd.Timedelta(days=1)  # 2023-07-01
    print(f"Checking label for {check_date.date()} - should reflect EDCT activity on {next_day.date()}")

    july_edct = load_edct(2023, 7)
    july1 = july_edct[(july_edct["airport"] == AIRPORT) & (july_edct["date"] == next_day)]
    had_edct_manually = bool(
        (july1["arrivals_with_edct"] > 0).any() or (july1["departures_with_edct"] > 0).any()
    )
    label_from_wrapper = int(labels[labels["date"] == check_date]["label"].iloc[0])
    print(f"  manual check on raw July EDCT file: had_edct = {had_edct_manually}")
    print(f"  build_strategic_labels' label for {check_date.date()}: {label_from_wrapper}")
    print(f"  match: {had_edct_manually == bool(label_from_wrapper)}")

    print()
    print("=== Joins cleanly with build_strategic_features (Step 2) on (airport, date) ===")
    from features.strategic import build_strategic_features
    features = build_strategic_features(AIRPORT, DATES, [(2023, 6), (2023, 7)])
    joined = features.merge(labels, on=["airport", "date"], how="inner")
    print(f"features rows: {len(features)}, labels rows: {len(labels)}, joined rows: {len(joined)}")
    print(joined[["airport", "date", "recent_disruption_rate", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
