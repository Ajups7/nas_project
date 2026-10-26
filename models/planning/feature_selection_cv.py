"""
Chooses the Planning feature set with walk-forward blocked CV inside the
train period (validation/temporal_blocked_cv.py), instead of trusting one
validation year: recent-history features helped within airports in the
train years but reversed in 2023.

Per fold:
- climatology (and the anomaly features built on it) is recomputed from the
  fold's own train dates - the cached dataset's climatology covers all of
  2016-2022, which would leak each fold's validation year into its features;
- COVID rows (models/planning/data.py) are removed from train and
  validation; folds that are mostly COVID are skipped.

Each feature set is scored by logistic regression (deterministic) and
PlanningDisruptNet (2 seeds, early stopping on the fold's validation window -
slightly optimistic, equally so for every feature set).

Run from the repo root:
    python -m models.planning.feature_selection_cv
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.planning import (
    ANOMALY_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    RECENT_COLUMNS,
    build_planning_features,
    eligible_airports,
)
from models.planning.data import Standardizer, fill_missing, in_covid
from models.planning.metrics import evaluate
from models.planning.train_disruptnet import RESULTS_DIR, Config, train_one
from validation.temporal_blocked_cv import assign_fold_buckets, generate_temporal_folds

NO_RECENT = [c for c in FEATURE_COLUMNS if c not in RECENT_COLUMNS]
FEATURE_SETS = {
    "no_recent": NO_RECENT,
    "raw_recent": FEATURE_COLUMNS,
    "anomaly": NO_RECENT + ANOMALY_COLUMNS,
}
MAX_COVID_SHARE = 0.5


def fold_dataset(labels: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.Series(sorted(labels["date"].unique()))
    feats = pd.concat(
        [build_planning_features(a, dates, train_end=str(train_end.date())) for a in eligible_airports()],
        ignore_index=True,
    )
    return fill_missing(feats.merge(labels, on=["airport", "date"], how="inner"))


def main():
    labels = pd.read_parquet(DATASET_PATH, columns=["airport", "date", "label", "split"])
    labels = labels[labels["split"] == "train"].drop(columns="split")
    # Two years minimum before the first fold: 2016 alone has no July
    # (missing EDCT month), so a 2016-only climatology would be incomplete.
    folds = generate_temporal_folds("2016-01-01", "2022-12-31", 8, "2017-12-31", 365)

    rows = []
    for fold in folds:
        val_dates = pd.Series(pd.date_range(fold["validation_start"], fold["validation_end"]))
        covid_share = in_covid(val_dates).mean()
        tag = f"fold {fold['fold']} (val {fold['validation_start'].date()} -> {fold['validation_end'].date()})"
        if covid_share > MAX_COVID_SHARE:
            print(f"{tag}: skipped, {covid_share:.0%} COVID")
            continue

        t = time.time()
        df = fold_dataset(labels, fold["train_end"])
        bucket = assign_fold_buckets(df["date"], fold).values
        keep = ~in_covid(df["date"]).values
        splits = {
            "train": df[(bucket == "train") & keep].reset_index(drop=True),
            "validation": df[(bucket == "validation") & keep].reset_index(drop=True),
        }
        tr, va = splits["train"], splits["validation"]
        print(f"{tag}: train {len(tr):,} rows, val {len(va):,} rows, "
              f"val prevalence {va['label'].mean():.3f}  (features built in {time.time() - t:.0f}s)", flush=True)

        for set_name, cols in FEATURE_SETS.items():
            scaler = Standardizer(cols).fit(tr)
            lr = LogisticRegression(max_iter=2000).fit(scaler.transform(tr), tr["label"])
            m = evaluate(va["label"], lr.predict_proba(scaler.transform(va))[:, 1], va["airport"])
            rows.append({"fold": fold["fold"], "features": set_name, "model": "logistic", **m})

            for seed in (0, 1):
                _, _, m = train_one(Config(f"cv_{set_name}", feature_columns=list(cols), seed=seed), splits)
                rows.append({"fold": fold["fold"], "features": set_name, "model": f"disruptnet_s{seed}", **m})
            print(f"   {set_name:11s} within-AUC  logistic {rows[-3]['within_airport_auc']:.3f}  "
                  f"disruptnet {np.mean([rows[-2]['within_airport_auc'], rows[-1]['within_airport_auc']]):.3f}   "
                  f"log loss  logistic {rows[-3]['log_loss']:.3f}  "
                  f"disruptnet {np.mean([rows[-2]['log_loss'], rows[-1]['log_loss']]):.3f}", flush=True)

    res = pd.DataFrame(rows)
    res["model_family"] = np.where(res["model"] == "logistic", "logistic", "disruptnet")
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "feature_selection_cv.csv", index=False)

    pd.set_option("display.width", 200)
    for metric in ("within_airport_auc", "roc_auc", "log_loss", "brier"):
        piv = res.groupby(["model_family", "features", "fold"])[metric].mean().unstack("fold")
        piv["mean"] = piv.mean(axis=1)
        print(f"\n{metric} by fold:")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
