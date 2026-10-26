"""
PlanningDisruptNet never predicts below ~5% at quiet airports (CVG: 0
positives in train, mean prediction 8.4%), which sits right at the ~7%
cost-ratio threshold and costs ~$61/airport-day of false alarms vs TFT
(investigate_tft_cost.py). Working theory: the ~22 quiet airports look
alike in the features (GDP climatology ~0, recent rate 0) but differ in
label rate (0-8%), so the model predicts their pooled rate.

Variants, each with 3 seeds, scored on 2023 validation and the three
walk-forward CV years (feature_selection_cv.py protocol):
- base          current config
- longer        lr 3e-4, patience 20
- airport       + one-hot airport identity
- log_rates     + log(rate + 1e-3) of GDP climatology and recent rates
- isotonic      base, then isotonic regression fitted on train predictions

Run from the repo root:
    python -m models.planning.fix_probability_floor
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from common.temporal_split import TRAIN_END
from features.planning import DATASET_PATH, FEATURE_COLUMNS, eligible_airports
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.data import fill_missing, in_covid, load_planning_splits
from models.planning.feature_selection_cv import fold_dataset
from models.planning.metrics import decision_costs, evaluate
from models.planning.train_disruptnet import RESULTS_DIR, Config, train_one
from validation.temporal_blocked_cv import assign_fold_buckets, generate_temporal_folds

SEEDS = (0, 1, 2)
AIRPORTS = eligible_airports()
AIRPORT_COLUMNS = [f"ap_{a}" for a in AIRPORTS]
LOG_COLUMNS = ["log_climo_gdp_day_rate", "log_recent_gdp_rate_7d", "log_recent_gdp_rate_30d"]
VARIANTS = {
    "base": dict(columns=FEATURE_COLUMNS),
    "longer": dict(columns=FEATURE_COLUMNS, lr=3e-4, patience=20, max_epochs=200),
    "airport": dict(columns=FEATURE_COLUMNS + AIRPORT_COLUMNS),
    "log_rates": dict(columns=FEATURE_COLUMNS + LOG_COLUMNS),
    "isotonic": dict(columns=FEATURE_COLUMNS, isotonic=True),
}


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, c in zip(AIRPORTS, AIRPORT_COLUMNS):
        df[c] = (df["airport"] == a).astype(np.float32)
    for src, c in zip(["climo_gdp_day_rate", "recent_gdp_rate_7d", "recent_gdp_rate_30d"], LOG_COLUMNS):
        df[c] = np.log(df[src] + 1e-3)
    return df


def settings() -> dict:
    out = {}
    s = load_planning_splits(exclude_covid=True)
    out["val2023"] = {"train": s["train"], "validation": s["validation"]}
    labels = pd.read_parquet(DATASET_PATH, columns=["airport", "date", "label", "split"])
    labels = labels[labels["split"] == "train"].drop(columns="split")
    folds = {f["fold"]: f for f in generate_temporal_folds("2016-01-01", TRAIN_END, 8, "2017-12-31", 365)}
    for k in (0, 1, 3):
        df = fill_missing(fold_dataset(labels, folds[k]["train_end"]))
        bucket = assign_fold_buckets(df["date"], folds[k]).values
        keep = ~in_covid(df["date"]).values
        out[f"cv_fold{k}"] = {
            "train": df[(bucket == "train") & keep].reset_index(drop=True),
            "validation": df[(bucket == "validation") & keep].reset_index(drop=True),
        }
    return {k: {s: add_columns(v) for s, v in frames.items()} for k, frames in out.items()}


def predict(model, scaler, df) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(scaler.transform(df))).numpy()


def main():
    t = time.time()
    all_frames = settings()
    print(f"settings built in {time.time() - t:.0f}s", flush=True)
    faoc = FAOCLoss()

    rows = []
    for setting, frames in all_frames.items():
        tr, va = frames["train"], frames["validation"]
        y, a = va["label"].to_numpy(), va["airport"].to_numpy()
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
        bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
        quiet = tr.groupby("airport")["label"].sum()
        quiet = set(quiet[quiet == 0].index)
        for variant, spec in VARIANTS.items():
            for seed in SEEDS:
                ts = time.time()
                cfg = Config(f"floor_{variant}", feature_columns=list(spec["columns"]), seed=seed,
                             lr=spec.get("lr", 1e-3), patience=spec.get("patience", 6),
                             max_epochs=spec.get("max_epochs", 60))
                model, scaler, _ = train_one(cfg, frames)
                p = predict(model, scaler, va)
                if spec.get("isotonic"):
                    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
                    iso.fit(predict(model, scaler, tr), tr["label"])
                    p = iso.predict(p)
                m = evaluate(y, p, a)
                qmask = np.isin(a, list(quiet))
                rows.append({
                    "setting": setting, "variant": variant, "seed": seed, **m,
                    "cost": decision_costs(y, p >= bayes, a, faoc)["cost_per_airport_day"],
                    "quiet_airports": len(quiet),
                    "quiet_mean_p": float(p[qmask].mean()) if qmask.any() else np.nan,
                    "quiet_actual": float(y[qmask].mean()) if qmask.any() else np.nan,
                })
                r = rows[-1]
                print(f"{setting:9s} {variant:10s} s{seed}: cost ${r['cost']:.0f}  within {r['within_airport_auc']:.3f}  "
                      f"logloss {r['log_loss']:.3f}  ece {r['ece']:.3f}  quiet p {r['quiet_mean_p']:.3f} "
                      f"(actual {r['quiet_actual']:.3f}, n={len(quiet)})  ({time.time() - ts:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "probability_floor_fixes.csv", index=False)
    pd.set_option("display.width", 220)
    for metric in ("cost", "log_loss", "within_airport_auc", "ece", "quiet_mean_p"):
        piv = res.groupby(["variant", "setting"], sort=False)[metric].mean().unstack("setting")
        piv["mean"] = piv.mean(axis=1)
        print(f"\n{metric} (mean of 3 seeds):")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
