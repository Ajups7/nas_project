"""
Reference baselines for the Planning tier - the floor every real model must
beat. Scored on VALIDATION only; test is untouched until the final comparison.

1. Climatology lookup: P(label) = the train-period label rate for this
   airport and calendar month of the as-of date. No model at all.
2. Logistic regression on standardized features, with and without the
   recent-history columns.

Each is fit twice: on all train rows, and with the COVID period removed
(models/planning/data.py), to measure whether excluding it helps.

Run from the repo root:
    python -m models.planning.train_reference
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.planning import FEATURE_COLUMNS, RECENT_COLUMNS
from models.planning.data import Standardizer, load_planning_splits
from models.planning.metrics import evaluate, results_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NO_RECENT = [c for c in FEATURE_COLUMNS if c not in RECENT_COLUMNS]


def climatology_lookup(train: pd.DataFrame, target: pd.DataFrame):
    month = train["date"].dt.month
    by_airport_month = train.groupby([train["airport"], month])["label"].mean()
    by_airport = train.groupby("airport")["label"].mean()
    keys = pd.MultiIndex.from_arrays([target["airport"], target["date"].dt.month])
    p = by_airport_month.reindex(keys).to_numpy()
    fallback = target["airport"].map(by_airport).fillna(train["label"].mean()).to_numpy()
    return pd.Series(p).fillna(pd.Series(fallback)).to_numpy()


def logistic(train: pd.DataFrame, target: pd.DataFrame, columns: list[str]):
    scaler = Standardizer(columns).fit(train)
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.transform(train), train["label"])
    return model.predict_proba(scaler.transform(target))[:, 1], model, columns


def main():
    rows = {}
    coefs = None
    for exclude_covid in (False, True):
        tag = "excl. COVID" if exclude_covid else "all train"
        splits = load_planning_splits(exclude_covid=exclude_covid)
        train, val = splits["train"], splits["validation"]
        print(f"[{tag}] train rows {len(train):,}  prevalence {train['label'].mean():.3f}")

        rows[f"climatology lookup | {tag}"] = evaluate(
            val["label"], climatology_lookup(train, val), val["airport"])

        p, _, _ = logistic(train, val, NO_RECENT)
        rows[f"logistic, no recent | {tag}"] = evaluate(val["label"], p, val["airport"])

        p, model, cols = logistic(train, val, FEATURE_COLUMNS)
        rows[f"logistic, all features | {tag}"] = evaluate(val["label"], p, val["airport"])
        if exclude_covid:
            coefs = pd.Series(model.coef_[0], index=cols).sort_values()

    table = results_table(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "reference_baselines_validation.csv")
    pd.set_option("display.width", 200)
    print("\nVALIDATION (2023-01-31 .. 2024-01-30)")
    print(table.drop(columns=["prevalence"]).round(3).to_string())
    print(f"validation prevalence: {table['prevalence'].iloc[0]:.3f}")
    print("\nlogistic coefficients (all features, excl. COVID; standardized units):")
    print(coefs.round(3).to_string())


if __name__ == "__main__":
    main()
