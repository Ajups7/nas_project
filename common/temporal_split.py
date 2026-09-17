"""
PROPOSED Phase 0 shared utility - NOT yet agreed on with the team.

Phase 0 (see TEAM_PLAN.md, "Before splitting off") calls for a temporal
split - fixed train/validation/test date boundaries, used identically by
all three tiers - decided together, in one sitting. This file is a
concrete proposal for that split, written as a reusable function so the
embargo logic can be checked mechanically rather than eyeballed off a
calendar. Nothing here is final until Person A and Person B have signed
off - see TEAM_PLAN.md's temporal split section for the human-readable
version of this same proposal.

Why an embargo, not just three adjacent date ranges: the Planning tier's
label (features/planning_label.py) looks forward up to 30 days from any
given date. Without a gap, a training example near the train/validation
boundary would have its label computed from data that's actually inside
the validation period - a real leak, not a hypothetical one. The embargo
is a dead zone between splits, sized to at least the longest horizon
across all three tiers (30 days, Planning's own) - no date inside it is
ever used by train, validation, or test.

assign_temporal_split() takes exact dates and an exact embargo_days,
not calendar months - see below for why that matters: an earlier draft of
this proposal used "the rest of the calendar month" as the embargo and
that was WRONG for any embargo landing on February in a leap year (29
days - less than the required 30), which is exactly the kind of mistake
this function exists to make impossible.
"""

import pandas as pd

DEFAULT_EMBARGO_DAYS = 30  # matches the Planning tier's own 1-30 day horizon,
                            # the longest of the three tiers and therefore the
                            # binding constraint for everyone's embargo width


def assign_temporal_split(
    dates: pd.Series,
    train_end: str,
    validation_end: str,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> pd.Series:
    """
    Assigns each date in `dates` to one of four buckets: "train",
    "embargo", "validation", "test". Two embargo windows are inserted
    automatically, immediately after train_end and after validation_end,
    each embargo_days wide - the caller never has to compute embargo
    boundaries by hand.

    train:      date <= train_end
    embargo:    train_end < date <= train_end + embargo_days
    validation: train_end + embargo_days < date <= validation_end
    embargo:    validation_end < date <= validation_end + embargo_days
    test:       date > validation_end + embargo_days

    train_end, validation_end: 'YYYY-MM-DD' strings (or anything
    pd.Timestamp can parse). embargo_days must be >= the longest forecast
    horizon of any tier using this split (30, for Planning) or the
    leakage this function exists to prevent isn't actually prevented.
    """
    dates = pd.to_datetime(pd.Series(dates))
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    embargo1_end = train_end + pd.Timedelta(days=embargo_days)
    embargo2_end = validation_end + pd.Timedelta(days=embargo_days)

    if validation_end <= embargo1_end:
        raise ValueError(
            f"validation_end ({validation_end.date()}) must be after "
            f"train_end + embargo_days ({embargo1_end.date()}) - there's no "
            f"room left for a validation set with this embargo width"
        )

    conditions = [
        dates <= train_end,
        dates <= embargo1_end,
        dates <= validation_end,
        dates <= embargo2_end,
    ]
    choices = ["train", "embargo", "validation", "embargo"]

    result = pd.Series("test", index=dates.index)
    for condition, choice in zip(reversed(conditions), reversed(choices)):
        result = result.where(~condition, choice)
    return result
