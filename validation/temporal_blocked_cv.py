"""
Person C (Planning tier) - Step 10 of the build sequence: temporal blocked
CV. "Regime-stratified folds, blocked in time so no disruption event leaks
across a split." Shared utility (TEAM_PLAN.md) - builds directly on Step
5's embargo logic (common/temporal_split.py) rather than reimplementing it,
generalized from one train/validation/test split into multiple rolling
folds.

Walk-forward (expanding window) design: fold 0 trains on
[data_start, initial_train_end], with an embargo gap (same mechanism and
same DEFAULT_EMBARGO_DAYS as Step 5 - the Planning tier's 30-day horizon
is still the binding constraint), then validates on a fixed-length window
right after. Each later fold expands the training set to include the
previous fold's validation window, then slides forward - standard
walk-forward CV, just with an embargo inserted at every fold boundary, not
only the one boundary Step 5 cared about.

"Regime-stratified" - honestly scoped: real GDP/Ground-Stop/MIT regime
labels are still blocked on ASPM Advisories access (README.md). Until then,
stratification here checks the binary EDCT-based label from
features/planning_label.py: does each fold's validation window have a
reasonable MIX of disruption/non-disruption days, not almost entirely one
or the other? A degenerate fold (e.g. all-positive or all-negative) makes
metrics like regime_transition_f1 or cost_weighted_recall
(validation/kpi_suite.py) undefined or meaningless - this check exists to
catch that before anyone trusts a fold's results.
"""

import pandas as pd

from common.temporal_split import DEFAULT_EMBARGO_DAYS, assign_temporal_split


def generate_temporal_folds(
    data_start: str,
    data_end: str,
    n_folds: int,
    initial_train_end: str,
    validation_days: int,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> list[dict]:
    """Returns up to n_folds fold definitions (fewer if the data range runs
    out first), each a dict with train_start/train_end/validation_start/
    validation_end. Consecutive folds' train sets expand to absorb the
    prior fold's validation window; embargo_days separates every
    train/validation boundary, same as Step 5."""
    folds = []
    train_end = pd.Timestamp(initial_train_end)
    data_start_ts = pd.Timestamp(data_start)
    data_end_ts = pd.Timestamp(data_end)

    for i in range(n_folds):
        validation_start = train_end + pd.Timedelta(days=embargo_days + 1)
        validation_end = validation_start + pd.Timedelta(days=validation_days - 1)
        if validation_end > data_end_ts:
            break
        folds.append({
            "fold": i,
            "train_start": data_start_ts,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
        })
        train_end = validation_end

    return folds


def assign_fold_buckets(dates: pd.Series, fold: dict, embargo_days: int = DEFAULT_EMBARGO_DAYS) -> pd.Series:
    """Reuses Step 5's assign_temporal_split() directly for one fold,
    rather than re-deriving train/embargo/validation membership with
    separate logic - the same embargo-safety guarantee applies per fold,
    not just to the one split Step 5 defined."""
    return assign_temporal_split(
        dates,
        train_end=str(fold["train_end"].date()),
        validation_end=str(fold["validation_end"].date()),
        embargo_days=embargo_days,
    )


def check_fold_regime_balance(
    dates: pd.Series,
    labels: pd.Series,
    fold: dict,
    min_positive_rate: float = 0.05,
) -> dict:
    """Checks whether `fold`'s validation window has a reasonable mix of
    positive/negative labels - see module docstring for why this is a
    binary stand-in for real regime stratification, not the real thing.
    Raises if the fold's validation window contains no dates at all
    (a configuration error, not a data finding)."""
    dates = pd.to_datetime(pd.Series(dates))
    mask = (dates >= fold["validation_start"]) & (dates <= fold["validation_end"])
    fold_labels = pd.Series(labels)[mask.values]
    if len(fold_labels) == 0:
        raise ValueError(f"fold {fold['fold']}: no dates found in its validation window")

    positive_rate = float(fold_labels.mean())
    return {
        "fold": fold["fold"],
        "n_dates": len(fold_labels),
        "positive_rate": positive_rate,
        "well_stratified": min_positive_rate <= positive_rate <= (1 - min_positive_rate),
    }
