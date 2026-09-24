"""
Validates strategic.py against REAL downloaded data (same approach as
validate_planning_features.py - the data already exists locally).

Runs on airport ATL, June 1-10 2023, deliberately spanning a month
boundary (June 1's "yesterday" is May 31) to exercise the cross-month
shift in build_strategic_features, and including one EDCT-known-gap-
adjacent case is not needed here since 2023 isn't one of the 2 gap months
(2016-07, 2025-09) - both required (year, month) pairs are real data.

Run from the repo root:
    python -m features.validate_strategic_features
"""

import pandas as pd

from features.strategic import (
    build_recent_disruption_features,
    build_recent_operational_features,
    build_recent_weather_features,
    build_strategic_features,
)

AIRPORT = "ATL"
DATES = pd.date_range("2023-06-01", "2023-06-10")
YEAR_MONTHS = [(2023, 5), (2023, 6)]  # May needed for June 1's "yesterday"


def main():
    print(f"=== Recent operational features, {AIRPORT}, {YEAR_MONTHS} ===")
    operational = build_recent_operational_features(AIRPORT, YEAR_MONTHS)
    print(operational[operational["date"].dt.month == 6].head(3).to_string(index=False))

    print()
    print(f"=== Recent disruption features, {AIRPORT}, {YEAR_MONTHS} ===")
    disruption = build_recent_disruption_features(AIRPORT, YEAR_MONTHS)
    print(disruption[disruption["date"].dt.month == 6].head(3).to_string(index=False))

    print()
    print(f"=== Recent weather features, {AIRPORT}, {DATES.min()} - {DATES.max()} ===")
    weather = build_recent_weather_features(AIRPORT, DATES - pd.Timedelta(days=1))
    print(weather.to_string(index=False))

    print()
    print(f"=== Full assembly: build_strategic_features, {AIRPORT}, {DATES.min().date()} - {DATES.max().date()} ===")
    full = build_strategic_features(AIRPORT, DATES, YEAR_MONTHS)
    print(full.to_string(index=False))
    print()
    print("Missing values per column (June 1 depends on May 31 - checking the cross-month shift actually pulled real data, not NaN):")
    print(full.isna().sum().to_string())

    print()
    print("Manual cross-check: June 2's recent_disruption_rate should equal June 1's raw disruption_rate")
    june1_raw = disruption[disruption["date"] == pd.Timestamp("2023-06-01")]
    print(f"  raw June 1 recent_disruption_rate row: {june1_raw['recent_disruption_rate'].values}")
    june2_feature = full[full["date"] == pd.Timestamp("2023-06-02")]["recent_disruption_rate"].values
    print(f"  build_strategic_features June 2 row's recent_disruption_rate: {june2_feature}")


if __name__ == "__main__":
    main()
