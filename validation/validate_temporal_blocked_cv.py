"""
Validates temporal_blocked_cv.py two ways: pure date-arithmetic checks
over the full fold-generation mechanics (fast, no data loading), and a
REAL-DATA check of check_fold_regime_balance() using features/
planning_label.py's build_planning_labels() at ATL and ADK - reusing the
exact same busy-vs-quiet-airport contrast Step 4 already established
(features/planning_label_validation.md), to see if the stratification
check correctly flags what we already know from Step 4: ATL is
near-always-positive, ADK is near-always-negative - both degenerate for
cross-validation purposes, for opposite reasons.

Run from the repo root:
    python -m validation.validate_temporal_blocked_cv
"""

import pandas as pd

from features.planning_label import build_planning_labels
from validation.temporal_blocked_cv import (
    assign_fold_buckets,
    check_fold_regime_balance,
    generate_temporal_folds,
)

DATA_START = "2016-01-01"
DATA_END = "2026-01-31"


def validate_fold_generation():
    print("=== generate_temporal_folds: mechanics ===")
    folds = generate_temporal_folds(
        DATA_START, DATA_END, n_folds=3, initial_train_end="2018-12-31", validation_days=180
    )
    print(f"generated {len(folds)} folds (requested 3)")
    for f in folds:
        print(f"  fold {f['fold']}: train [{f['train_start'].date()} -> {f['train_end'].date()}]  "
              f"validation [{f['validation_start'].date()} -> {f['validation_end'].date()}]")

    print()
    print("=== Embargo gap check: exactly 30 days between each fold's train_end and validation_start ===")
    for f in folds:
        gap_days = (f["validation_start"] - f["train_end"]).days - 1
        print(f"  fold {f['fold']}: gap = {gap_days} days (expect 30)")

    print()
    print("=== Walk-forward property: next fold's train_end == this fold's validation_end ===")
    for i in range(len(folds) - 1):
        matches = folds[i]["validation_end"] == folds[i + 1]["train_end"]
        print(f"  fold {i} validation_end ({folds[i]['validation_end'].date()}) == "
              f"fold {i + 1} train_end ({folds[i + 1]['train_end'].date()}): {matches}")

    print()
    print("=== Integration check: assign_fold_buckets reuses Step 5's assign_temporal_split ===")
    daily_dates = pd.date_range(DATA_START, DATA_END, freq="D")
    buckets = assign_fold_buckets(daily_dates, folds[0])
    print(f"fold 0 bucket counts:\n{buckets.value_counts().to_string()}")

    print()
    print("=== Boundary case: requesting more folds than the data can support ===")
    too_many = generate_temporal_folds(
        DATA_START, DATA_END, n_folds=20, initial_train_end="2018-12-31", validation_days=180
    )
    print(f"requested 20 folds, data only supports {len(too_many)} before running out "
          f"(last fold's validation_end: {too_many[-1]['validation_end'].date()}, "
          f"data_end: {DATA_END})")

    return folds


def validate_regime_balance(folds):
    print()
    print("=== check_fold_regime_balance on REAL data: does it catch what Step 4 already found? ===")
    print("Recall from features/planning_label_validation.md: ATL is a busy hub (near-constant")
    print("EDCT activity), ADK is near-empty. Both should be flagged as poorly stratified,")
    print("for opposite reasons - this reuses fold 0's validation window only, to limit how")
    print("many EDCT months (~13s/month to load) this validation run needs.")

    fold = folds[0]
    dates = pd.date_range(fold["validation_start"], fold["validation_end"], freq="D")

    for airport in ["ATL", "ADK"]:
        print(f"\n--- {airport}, fold 0 validation window "
              f"({fold['validation_start'].date()} -> {fold['validation_end'].date()}) ---")
        label_df = build_planning_labels(airport, dates, horizon_days=30)
        result = check_fold_regime_balance(dates, label_df["label"], fold)
        print(f"  n_dates={result['n_dates']}  positive_rate={result['positive_rate']:.3f}  "
              f"well_stratified={result['well_stratified']}")


if __name__ == "__main__":
    generated_folds = validate_fold_generation()
    validate_regime_balance(generated_folds)
