"""
Person A (Tactical tier) - Step 4 of the build sequence: apply the shared
temporal split (see TEAM_PLAN.md and common/temporal_split.py) to tactical
data.

Uses TEAM_PLAN.md's agreed boundaries directly: train_end="2022-12-31",
validation_end="2024-01-30", embargo_days=30 (the shared
DEFAULT_EMBARGO_DAYS, sized for Planning's 30-day horizon per TEAM_PLAN.md
- a real but small cost for Tactical: ~30 embargoed days out of ~2557
training days, and hourly granularity means each day still carries 24
rows, so the lost N is negligible for a 0-2h model).

These same three values are independently duplicated as local constants in
common/validate_temporal_split.py - but that's a standalone diagnostic
script (guarded by if __name__ == "__main__", run via
`python -m common.validate_temporal_split`), not an importable config
module, so they are NOT imported from there. There is currently no single
shared config module for these three values - each tier holds its own
copy per TEAM_PLAN.md. Worth raising with Person B: independent copies of
the same three numbers across tiers is a drift risk if TEAM_PLAN.md's
numbers ever change.

Operates purely on the `date` column - correct for hourly rows too, since
every row sharing a calendar date gets the same bucket (you don't want
some hours of a day in train and others in test).
"""

import pandas as pd

from common.temporal_split import DEFAULT_EMBARGO_DAYS, assign_temporal_split

# Mirrors TEAM_PLAN.md's agreed split - see module docstring.
TRAIN_END = "2022-12-31"
VALIDATION_END = "2024-01-30"
EMBARGO_DAYS = DEFAULT_EMBARGO_DAYS  # 30, per TEAM_PLAN.md


def assign_tactical_split(dates: pd.Series) -> pd.Series:
    """Buckets each date into "train"/"embargo"/"validation"/"embargo"/
    "test" using TEAM_PLAN.md's agreed boundaries. One bucket label per
    row - safe to call directly on an hourly table's `date` column."""
    return assign_temporal_split(dates, TRAIN_END, VALIDATION_END, EMBARGO_DAYS)
