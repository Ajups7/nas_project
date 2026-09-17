"""
Validates planning.py against REAL downloaded data (unlike the losses/ and
models/planning/ validations, which used synthetic inputs - there's real
BTS/ASPM/EDCT/METAR data to check this against, so this uses it).

Runs on airport ATL, full 2016-2026 METAR history for weather climatology
(fast - one file per airport), but only a 6-month SAMPLE for disruption
climatology (3 Januaries + 3 Julys across different years) rather than the
full 119-month history - see planning_features_validation.md for why:
load_edct() takes ~15s/month, so the full history takes ~30 minutes and
should be run once as a batch/cache job, not inside a quick validation run.

Run from the repo root:
    python -m features.validate_planning_features
"""

import time

import pandas as pd

from features.planning import (
    build_calendar_features,
    build_disruption_climatology,
    build_planning_features,
    build_weather_climatology,
)

AIRPORT = "ATL"
# Deliberately includes (2016, 7), one of the 2 documented gaps in the EDCT
# download (README.md, "119/121 months") - this doubles as a test that
# build_disruption_climatology skips missing months gracefully.
SAMPLE_EDCT_MONTHS = [(2016, 1), (2019, 1), (2022, 1), (2016, 7), (2019, 7), (2022, 7)]


def main():
    print("=== Calendar features (pure date arithmetic, no data loading) ===")
    dates = pd.date_range("2024-01-01", "2024-01-07")
    calendar = build_calendar_features(dates)
    print(calendar.to_string(index=False))
    print()
    print("Sanity check - day_of_year_sin/cos should each stay in [-1, 1]:")
    print(f"  sin range: [{calendar['day_of_year_sin'].min():.3f}, {calendar['day_of_year_sin'].max():.3f}]")
    print(f"  cos range: [{calendar['day_of_year_cos'].min():.3f}, {calendar['day_of_year_cos'].max():.3f}]")

    print()
    print(f"=== Weather climatology for {AIRPORT}, full 2016-2026 history ===")
    t0 = time.time()
    weather_clim = build_weather_climatology(AIRPORT)
    elapsed = time.time() - t0
    print(f"(computed in {elapsed:.1f}s, from the full downloaded METAR history)")
    print(weather_clim[["airport", "month", "climo_tmpf", "climo_sknt", "climo_vsby"]].to_string(index=False))

    print()
    print(f"=== Disruption climatology for {AIRPORT}, SAMPLE of 6/119 months ===")
    print(f"months used: {SAMPLE_EDCT_MONTHS}")
    t0 = time.time()
    disruption_clim = build_disruption_climatology(AIRPORT, SAMPLE_EDCT_MONTHS)
    elapsed = time.time() - t0
    print(f"(computed in {elapsed:.1f}s for 6 months -> "
          f"projects to roughly {elapsed / 6 * 119 / 60:.0f} minutes for the full 119-month history)")
    print(disruption_clim.to_string(index=False))

    print()
    print("=== Full assembly: build_planning_features (same sample months) ===")
    dates = pd.date_range("2024-01-01", "2024-01-05")
    full = build_planning_features(AIRPORT, dates, SAMPLE_EDCT_MONTHS)
    print(full.to_string(index=False))
    print()
    print(f"No missing values after merge: {not full.isna().any().any()}")


if __name__ == "__main__":
    main()
