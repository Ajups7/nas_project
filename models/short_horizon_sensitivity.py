"""
Sensitivity checks for the Strategic and Tactical tiers, mirroring the
Planning tier's (models/planning/sensitivity.py). VALIDATION ONLY: no test
prediction is ever computed, and nothing is written to the tiers' main
results or sealed prediction folders.

Strategic: GDP-day threshold 5% / 20% (main 10%); COVID kept in train.
Tactical:  GDP-hour threshold 30% / 70% (main 50%); COVID kept in train.
Models (3 seeds; logistic 1): logistic regression, DisruptNet standard and
FAOC, STGNN with the top-10 rotation graph and with no graph.

Questions: does FAOC still match the standard loss, does the graph effect
(none at Strategic; large at Tactical, from neighbors' departure EDCTs)
still hold, and does the model ranking change?

Each finished run is appended to results/short_horizon_sensitivity.csv; a
restart skips what's there. Run one (tier, variant) per process to bound
memory:
    python -m models.short_horizon_sensitivity strategic thr05
    python -m models.short_horizon_sensitivity tactical covid_included
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import models.planning.train_regime_transformer as rt
from common.data_loader import DATA_DIR
from common.disruption import gdp_days
from features.planning import eligible_airports
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.data import Standardizer
from models.planning.metrics import decision_costs, evaluate
from models.planning.train_disruptnet import Config, train_one
from models.strategic.rotation_gnn import RotationChainSpatioTemporalGNN

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "short_horizon_sensitivity.csv"
SEEDS = (0, 1, 2)
GRAPHS = ("top10", "none")
VARIANTS = {
    "strategic": {"thr05": (0.05, True), "thr20": (0.20, True), "covid_included": (0.10, False)},
    "tactical": {"thr30": (0.30, True), "thr70": (0.70, True), "covid_included": (0.50, False)},
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Scorer:
    def __init__(self, tier, variant, y, airports, onset=None, clean=None):
        self.tier, self.variant, self.y, self.a = tier, variant, y, airports
        self.onset, self.clean = onset, clean
        self.faoc = FAOCLoss()
        dm = np.array([self.faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in airports])
        self.bayes = self.faoc.fp_weight / (self.faoc.fp_weight + self.faoc.fn_weight * dm)
        self.done = set()
        if RESULTS_PATH.exists():
            r = pd.read_csv(RESULTS_PATH)
            self.done = set(zip(r["tier"], r["variant"], r["model"], r["seed"]))

    def has(self, model, seed):
        return (self.tier, self.variant, model, seed) in self.done

    def record(self, model, seed, p, rule="rule", seconds=0.0):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
        thr = 0.5 if rule == "0.5" else self.bayes
        m = evaluate(self.y, p, self.a)
        row = {"tier": self.tier, "variant": self.variant, "model": model, "seed": seed, "rule": rule, **m,
               "cost": decision_costs(self.y, p >= thr, self.a, self.faoc)["cost_per_airport_day"]}
        for tag, mask in (("onset", self.onset), ("clean", self.clean)):
            # Same columns for every tier, so Strategic and Tactical rows share one CSV.
            mm = evaluate(self.y[mask], p[mask], self.a[mask]) if mask is not None else {}
            row.update({f"{tag}_{k}": mm.get(k, np.nan) for k in ("roc_auc", "pr_auc", "prevalence")})
        row["seconds"] = round(seconds, 1)
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)
        extra = f"  clean pr-auc {row['clean_pr_auc']:.3f}" if self.clean is not None else ""
        log(f"{self.tier} {self.variant:14s} {model:20s} s{seed}: cost {row['cost']:.1f}  "
            f"within {m['within_airport_auc']:.3f}  log loss {m['log_loss']:.4f}  "
            f"mean p {m['mean_pred']:.4f} (actual {m['prevalence']:.4f}){extra}  ({seconds:.0f}s)")


def _tabular(sc: Scorer, frames: dict, cols: list[str], dn_kwargs: dict):
    tr, va = frames["train"], frames["validation"]
    if not sc.has("logistic", 0):
        t = time.time()
        s = Standardizer(cols).fit(tr)
        lr = LogisticRegression(max_iter=1000).fit(s.transform(tr), tr["label"])
        sc.record("logistic", 0, lr.predict_proba(s.transform(va))[:, 1], seconds=time.time() - t)
    for loss_type in ("standard", "faoc"):
        for seed in SEEDS:
            name = f"disruptnet {loss_type}"
            if sc.has(name, seed):
                continue
            t = time.time()
            net, s, _ = train_one(Config(f"sens_{loss_type}", feature_columns=list(cols), loss_type=loss_type,
                                         seed=seed, **dn_kwargs), frames)
            net.eval()
            with torch.no_grad():
                p = net(torch.from_numpy(s.transform(va))).numpy()
            sc.record(name, seed, p, "0.5" if loss_type == "faoc" else "rule", time.time() - t)


def strategic(variant: str):
    from features.strategic import FEATURE_COLUMNS, build_strategic_dataset
    from models.planning.data import AIRPORT_COLUMNS
    from models.strategic.data import load_daily, load_strategic_splits
    from models.strategic.train import SEQ_LEN, daily_matrix

    threshold, exclude_covid = VARIANTS["strategic"][variant]
    tag = f"thr{int(round(threshold * 100)):02d}"
    daily_path = DATA_DIR / "cache" / f"strategic_daily_features_{tag}.parquet"
    labels_path = DATA_DIR / "cache" / f"strategic_labels_{tag}.parquet"
    if threshold == 0.10:
        from features.strategic import DAILY_PATH, LABELS_PATH
        daily_path, labels_path = DAILY_PATH, LABELS_PATH
    elif not labels_path.exists():
        t = time.time()
        daily, labels = build_strategic_dataset(eligible_airports(threshold), threshold)
        daily.to_parquet(daily_path, index=False)
        labels.to_parquet(labels_path, index=False)
        log(f"built strategic dataset at {threshold:.0%} in {time.time() - t:.0f}s")

    daily = load_daily(daily_path)
    s = load_strategic_splits(exclude_covid=exclude_covid, daily=daily, labels_path=labels_path)
    frames = {"train": s["train"], "validation": s["validation"]}
    va = frames["validation"]
    sc = Scorer("strategic", variant, va["label"].to_numpy(), va["airport"].to_numpy())
    log(f"strategic {variant}: train {len(frames['train']):,} (prev {frames['train']['label'].mean():.4f}), "
        f"validation {len(va):,} (prev {va['label'].mean():.4f})")
    _tabular(sc, frames, FEATURE_COLUMNS + AIRPORT_COLUMNS, {})

    airports = eligible_airports(threshold)
    data = rt.DateGraphData.from_arrays(daily_matrix(daily, airports), frames, gdp_days(threshold).index,
                                        airports, SEQ_LEN, n_identity=len(airports))
    for graph in GRAPHS:
        for seed in SEEDS:
            name = f"stgnn {graph}"
            if sc.has(name, seed):
                continue
            t = time.time()
            model, adj, _, _ = rt.train_one(data, graph, seed, model_cls=RotationChainSpatioTemporalGNN)
            rows = rt.scored_rows(data, "validation", rt.predict(model, data, "validation", adj))
            assert np.array_equal(rows["label"].to_numpy(), sc.y)
            sc.record(name, seed, rows["p"].to_numpy(), seconds=time.time() - t)


def tactical(variant: str):
    import models.tactical.train as tt
    from features.tactical import TENSOR_PATH, build_tensor

    threshold, exclude_covid = VARIANTS["tactical"][variant]
    path = TENSOR_PATH
    if threshold != 0.50:
        path = DATA_DIR / "cache" / f"tactical_tensor_thr{int(round(threshold * 100)):02d}.npz"
        if not path.exists():
            t = time.time()
            np.savez(path, **build_tensor(threshold))
            log(f"built tactical tensor at {threshold:.0%} in {time.time() - t:.0f}s")

    data = tt.TacticalData(path, exclude_covid=exclude_covid)
    a, t = data.rows["validation"]
    y = data.label[a, t]
    onset = data.gdp[a, t] == 0
    sc = Scorer("tactical", variant, y, np.array(data.airports)[a], onset, onset & ~data.edct_now[a, t])
    log(f"tactical {variant}: train {len(data.rows['train'][0]):,} rows, validation {len(a):,} "
        f"(prev {y.mean():.4f}, clean-onset prev {y[sc.clean].mean():.4f})")

    cols = data.names + [f"ap_{x}" for x in data.airports]
    if not all(sc.has(m, s) for m in ("disruptnet standard", "disruptnet faoc") for s in SEEDS):
        frames = {"train": data.tabular("train"), "validation": data.tabular("validation")}
        _tabular(sc, frames, cols, {"batch_size": 4096, "max_epochs": 30, "patience": 4})
        del frames

    g = tt._HourGraph(data)
    full = dict(g.days)
    for graph in GRAPHS:
        for seed in SEEDS:
            name = f"stgnn {graph}"
            if sc.has(name, seed):
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            g.days = {"train": rng.choice(full["train"], 2000, replace=False),
                      "validation": np.sort(rng.choice(full["validation"], 2000, replace=False))}
            model, adj, _, _ = rt.train_one(g, graph, seed, model_cls=RotationChainSpatioTemporalGNN,
                                            resample_train=lambda: rng.choice(full["train"], 2000, replace=False))
            g.days["validation"] = full["validation"]
            P = rt.predict(model, g, "validation", adj)
            sc.record(name, seed, P[t, a], seconds=time.time() - t0)


if __name__ == "__main__":
    tier, variant = sys.argv[1], sys.argv[2]
    {"strategic": strategic, "tactical": tactical}[tier](variant)
