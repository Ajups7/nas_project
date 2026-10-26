"""
Sensitivity checks promised in features/planning_label_validation.md and
models/planning/training_results.md, on the VALIDATION window only:

- thr05 / thr20     GDP-day threshold 5% / 20% instead of 10% (label and
                    the GDP-derived features both change)
- covid_included    10% threshold, COVID period kept in train

Models (all with airport identity, the frozen configuration): logistic
regression, PlanningDisruptNet standard + FAOC, TFT, Regime Transformer
(graph passed on the command line). 3 seeds. Seq2Seq and SARIMA are left
out to keep the run to ~1.5 h; neither is a candidate main model.

Every finished run is appended to the results CSV; a restart skips what's
already there.

Run from the repo root:
    python -m models.planning.sensitivity <regime-transformer-graph>
"""

import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import models.planning.train_regime_transformer as rt
import models.planning.train_sequence_baselines as tsb
from common.data_loader import DATA_DIR
from features.planning import DATASET_PATH, FEATURE_COLUMNS, build_planning_dataset, eligible_airports
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import TFTBaseline
from models.planning.data import AIRPORT_COLUMNS, Standardizer, load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.sequences import build_sequence_splits, standardize
from models.planning.train_disruptnet import RESULTS_DIR, Config, train_one

SEEDS = (0, 1, 2)
RESULTS_PATH = RESULTS_DIR / "sensitivity_validation.csv"
COLUMNS = FEATURE_COLUMNS + AIRPORT_COLUMNS
VARIANTS = {
    "thr05": dict(threshold=0.05, exclude_covid=True),
    "thr20": dict(threshold=0.20, exclude_covid=True),
    "covid_included": dict(threshold=0.10, exclude_covid=False),
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def dataset_path(threshold: float):
    if threshold == 0.10:
        return DATASET_PATH
    path = DATA_DIR / "cache" / f"planning_dataset_thr{int(threshold * 100):02d}.parquet"
    if not path.exists():
        t = time.time()
        build_planning_dataset(eligible_airports(threshold), threshold=threshold).to_parquet(path, index=False)
        log(f"built {path.name} in {time.time() - t:.0f}s")
    return path


def main(graph: str):
    faoc = FAOCLoss()
    done = set()
    if RESULTS_PATH.exists():
        r = pd.read_csv(RESULTS_PATH)
        done = set(zip(r["variant"], r["model"], r["seed"]))

    def record(variant, model, seed, y, p, a, rule, seconds):
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
        bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
        m = evaluate(y, p, a)
        cost = decision_costs(y, p >= (0.5 if rule == "0.5" else bayes), a, faoc)["cost_per_airport_day"]
        row = {"variant": variant, "model": model, "seed": seed, "seconds": round(seconds, 1),
               **m, "cost_rule": rule, "cost": cost}
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)
        log(f"{variant:15s} {model:22s} seed {seed}: within {m['within_airport_auc']:.3f}  AUC {m['roc_auc']:.3f}  "
            f"log loss {m['log_loss']:.3f}  ece {m['ece']:.3f}  mean p {m['mean_pred']:.3f} "
            f"(actual {m['prevalence']:.3f})  cost ${cost:.0f} ({rule})  ({seconds:.0f}s)")

    for variant, spec in VARIANTS.items():
        path = dataset_path(spec["threshold"])
        frames = load_planning_splits(exclude_covid=spec["exclude_covid"], dataset_path=path)
        frames = {"train": frames["train"], "validation": frames["validation"]}
        tr, va = frames["train"], frames["validation"]
        y, a = va["label"].to_numpy(), va["airport"].to_numpy()
        log(f"{variant}: train {len(tr):,} rows (prevalence {tr['label'].mean():.3f}), "
            f"validation {len(va):,} (prevalence {y.mean():.3f})")

        if (variant, "logistic", 0) not in done:
            t = time.time()
            sc = Standardizer(COLUMNS).fit(tr)
            p = LogisticRegression(max_iter=2000).fit(sc.transform(tr), tr["label"]).predict_proba(sc.transform(va))[:, 1]
            record(variant, "logistic", 0, y, p, a, "bayes", time.time() - t)

        for loss_type in ("standard", "faoc"):
            for seed in SEEDS:
                name = f"disruptnet_{loss_type}"
                if (variant, name, seed) in done:
                    continue
                t = time.time()
                net, sc, _ = train_one(Config(f"sens_{name}", feature_columns=list(COLUMNS),
                                              loss_type=loss_type, seed=seed), frames)
                net.eval()
                with torch.no_grad():
                    p = net(torch.from_numpy(sc.transform(va))).numpy()
                record(variant, name, seed, y, p, a, "0.5" if loss_type == "faoc" else "bayes", time.time() - t)

        if any((variant, "tft", s) not in done for s in SEEDS):
            seq = standardize(build_sequence_splits(exclude_covid=spec["exclude_covid"], airport_onehot=True,
                                                    threshold=spec["threshold"], dataset_path=path))
            assert np.array_equal(seq["validation"][1], y)
            for seed in SEEDS:
                if (variant, "tft", seed) in done:
                    continue
                t = time.time()
                _, p, _, _ = tsb.train_one(TFTBaseline, seed, seq)
                record(variant, "tft", seed, y, p, a, "bayes", time.time() - t)
            del seq

        name = f"regime_transformer_{graph}"
        if any((variant, name, s) not in done for s in SEEDS):
            data = rt.DateGraphData(airport_onehot=True, threshold=spec["threshold"], dataset_path=path,
                                    exclude_covid=spec["exclude_covid"])
            for seed in SEEDS:
                if (variant, name, seed) in done:
                    continue
                t = time.time()
                model, adj, _, _ = rt.train_one(data, graph, seed)
                val = rt.scored_rows(data, "validation", rt.predict(model, data, "validation", adj))
                assert np.array_equal(val["label"].to_numpy(), y)
                record(variant, name, seed, y, val["p"].to_numpy(), a, "bayes", time.time() - t)
            del data

    r = pd.read_csv(RESULTS_PATH)
    cols = ["within_airport_auc", "roc_auc", "log_loss", "ece", "mean_pred", "prevalence", "cost"]
    pd.set_option("display.width", 220)
    print(r.groupby(["variant", "model"], sort=False)[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
