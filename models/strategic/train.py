"""
Strategic tier (2-24h): train and score every model on VALIDATION. Label:
D+1 is a GDP day; features: what is known at the end of D
(features/strategic.py). Same protocol as the Planning tier: COVID excluded
from train, airport identity in every learned model, 3 seeds, early
stopping on validation, cost-ratio decision threshold (FAOC: 0.5).

Each run also writes its TEST predictions to results/predictions/ - sealed:
nothing reads them until models/strategic/evaluate_test.py runs once, after
the choices are frozen.

Models:
  reference   climatology lookup (airport x target month), persistence
              (P(label | airport, GDP today)), logistic regression
  disruptnet  the MoE PlanningDisruptNet architecture, standard + FAOC loss
  sequence    Seq2Seq, TFT on 14-day windows
  graph       rotation-chain spatio-temporal GNN (Person B's
              models/strategic/rotation_gnn.py) with top-10 / full / no
              graph - the key test of the rotation-propagation claim
  sarima      the Planning tier's per-airport SARIMA, 1-day-ahead forecast

Run from the repo root (after `python -m features.strategic`):
    python -m models.strategic.train reference disruptnet sequence graph sarima
"""

import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import models.planning.train_regime_transformer as rt
import models.planning.train_sequence_baselines as tsb
from common.disruption import gdp_days
from features.planning import eligible_airports
from features.strategic import FEATURE_COLUMNS
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import Seq2SeqBaseline, TFTBaseline
from models.planning.data import AIRPORT_COLUMNS, Standardizer
from models.planning.metrics import decision_costs, evaluate
from models.planning.sequences import standardize, windows_for_frames
from models.planning.train_disruptnet import Config, train_one
from models.planning.train_sarima import _forecast_all_origins, fit_airport
from models.strategic.data import load_daily, load_strategic_splits
from models.strategic.rotation_gnn import RotationChainSpatioTemporalGNN

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PRED_DIR = RESULTS_DIR / "predictions"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULTS_PATH = RESULTS_DIR / "strategic_validation.csv"
SEEDS = (0, 1, 2)
SEQ_LEN = 14
GRAPHS = ("top10", "full", "none")
COLS = FEATURE_COLUMNS + AIRPORT_COLUMNS


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Recorder:
    def __init__(self, splits: dict):
        self.va, self.te = splits["validation"], splits["test"]
        faoc = FAOCLoss()
        self.faoc = faoc
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in self.va["airport"]])
        self.bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
        self.done = set()
        if RESULTS_PATH.exists():
            r = pd.read_csv(RESULTS_PATH)
            self.done = set(zip(r["model"], r["seed"]))

    def record(self, model: str, seed: int, p_val, p_test, rule: str = "rule", seconds: float = 0.0,
               best_epoch: int | None = None):
        y, a = self.va["label"].to_numpy(), self.va["airport"].to_numpy()
        m = evaluate(y, p_val, a)
        cost = decision_costs(y, p_val >= (0.5 if rule == "0.5" else self.bayes), a, self.faoc)["cost_per_airport_day"]
        row = {"model": model, "seed": seed, "rule": rule, **m, "cost": cost, "seconds": round(seconds, 1),
               "best_epoch": best_epoch}
        RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"p": np.asarray(p_test, dtype=np.float64)}).to_parquet(
            PRED_DIR / f"test_{model}_seed{seed}.parquet", index=False)
        pd.DataFrame({"p": np.asarray(p_val, dtype=np.float64)}).to_parquet(
            PRED_DIR / f"val_{model}_seed{seed}.parquet", index=False)
        log(f"{model:22s} seed {seed}: within {m['within_airport_auc']:.3f}  AUC {m['roc_auc']:.3f}  "
            f"log loss {m['log_loss']:.3f}  ece {m['ece']:.3f}  mean p {m['mean_pred']:.3f} "
            f"(actual {m['prevalence']:.3f})  cost ${cost:.0f} ({rule})  ({seconds:.0f}s)")


def run_reference(splits, rec: Recorder):
    tr, va, te = splits["train"], splits["validation"], splits["test"]

    def lookup(keys_tr, keys_x):
        rate = tr.groupby(keys_tr)["label"].mean()
        by_ap = tr.groupby("airport")["label"].mean()
        p = rate.reindex(pd.MultiIndex.from_arrays(keys_x)).to_numpy()
        return pd.Series(p).fillna(pd.Series(keys_x[0]).map(by_ap)).to_numpy()

    tmonth = lambda df: (df["date"] + pd.Timedelta(days=1)).dt.month
    if ("climatology lookup", 0) not in rec.done:
        rec.record("climatology lookup", 0,
                   lookup([tr["airport"], tmonth(tr)], [va["airport"], tmonth(va)]),
                   lookup([tr["airport"], tmonth(tr)], [te["airport"], tmonth(te)]))
    if ("persistence", 0) not in rec.done:
        rec.record("persistence", 0,
                   lookup([tr["airport"], tr["gdp_day_today"]], [va["airport"], va["gdp_day_today"]]),
                   lookup([tr["airport"], tr["gdp_day_today"]], [te["airport"], te["gdp_day_today"]]))
    if ("logistic", 0) not in rec.done:
        t = time.time()
        sc = Standardizer(COLS).fit(tr)
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(tr), tr["label"])
        rec.record("logistic", 0, lr.predict_proba(sc.transform(va))[:, 1],
                   lr.predict_proba(sc.transform(te))[:, 1], seconds=time.time() - t)


def run_disruptnet(splits, rec: Recorder):
    frames = {"train": splits["train"], "validation": splits["validation"]}
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for loss_type in ("standard", "faoc"):
        name = f"disruptnet {loss_type}"
        for seed in SEEDS:
            if (name, seed) in rec.done:
                continue
            t = time.time()
            net, sc, m = train_one(Config(f"strategic_{loss_type}", feature_columns=list(COLS),
                                          loss_type=loss_type, seed=seed), frames)
            torch.save({"state_dict": net.state_dict(), "columns": COLS, "mean": sc.scaler.mean_,
                        "scale": sc.scaler.scale_}, CHECKPOINT_DIR / f"disruptnet_{loss_type}_seed{seed}.pt")
            net.eval()
            with torch.no_grad():
                pv = net(torch.from_numpy(sc.transform(splits["validation"]))).numpy()
                pt = net(torch.from_numpy(sc.transform(splits["test"]))).numpy()
            rec.record(name, seed, pv, pt, "0.5" if loss_type == "faoc" else "rule", time.time() - t,
                       best_epoch=m["best_epoch"])


def daily_matrix(daily: pd.DataFrame, airports: list[str]) -> np.ndarray:
    """(airports, calendar days, features + one-hot airport)."""
    calendar = gdp_days().index
    M = np.stack([daily[daily["airport"] == a].set_index("date").reindex(calendar)[FEATURE_COLUMNS]
                  .to_numpy(dtype=np.float32) for a in airports])
    return rt.with_airport_onehot(M)


def run_sequence(splits, rec: Recorder, M: np.ndarray, airports: list[str]):
    if all((n, s) in rec.done for n in ("seq2seq", "tft") for s in SEEDS):
        return
    mats = {a: M[i] for i, a in enumerate(airports)}
    data = standardize(windows_for_frames(splits, mats, SEQ_LEN))
    for split in ("validation", "test"):
        assert np.array_equal(data[split][1], splits[split]["label"].to_numpy())
    for name, cls in (("seq2seq", Seq2SeqBaseline), ("tft", TFTBaseline)):
        for seed in SEEDS:
            if (name, seed) in rec.done:
                continue
            t = time.time()
            model, pv, best_epoch, _ = tsb.train_one(cls, seed, data)
            model.eval()
            X = torch.from_numpy(data["test"][0])
            with torch.no_grad():
                pt = torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)]).numpy()
            rec.record(name, seed, pv, pt, seconds=time.time() - t, best_epoch=best_epoch)


def run_graph(splits, rec: Recorder, M: np.ndarray, airports: list[str]):
    if all((f"stgnn {g}", s) in rec.done for g in GRAPHS for s in SEEDS):
        return
    data = rt.DateGraphData.from_arrays(M, splits, gdp_days().index, airports, SEQ_LEN, n_identity=len(airports))
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for graph in GRAPHS:
        for seed in SEEDS:
            name = f"stgnn {graph}"
            if (name, seed) in rec.done:
                continue
            t = time.time()
            model, adj, best_epoch, _ = rt.train_one(data, graph, seed, model_cls=RotationChainSpatioTemporalGNN)
            torch.save({"graph": graph, "state_dict": model.state_dict()},
                       CHECKPOINT_DIR / f"stgnn_{graph}_seed{seed}.pt")
            pv = rt.scored_rows(data, "validation", rt.predict(model, data, "validation", adj))
            pt = rt.scored_rows(data, "test", rt.predict(model, data, "test", adj))
            assert np.array_equal(pv["label"].to_numpy(), splits["validation"]["label"].to_numpy())
            assert np.array_equal(pt["label"].to_numpy(), splits["test"]["label"].to_numpy())
            rec.record(name, seed, pv["p"].to_numpy(), pt["p"].to_numpy(), seconds=time.time() - t,
                       best_epoch=best_epoch)


def run_sarima(splits, rec: Recorder, airports: list[str]):
    if ("sarima", 0) in rec.done:
        return
    t = time.time()
    with Pool(3) as pool:
        fitted = {a: (full, index) for a, _, full, index in pool.map(fit_airport, airports)}

    def predict(frame):
        p = np.empty(len(frame))
        for airport, idx in frame.groupby("airport").indices.items():
            full, index = fitted[airport]
            origins = pd.Series(np.arange(len(index)), index=index).reindex(frame["date"].iloc[idx]).to_numpy()
            p[idx] = np.clip(_forecast_all_origins(full, origins, horizon=1)[:, 0], 0, 1)
        return p

    rec.record("sarima", 0, np.clip(predict(splits["validation"]), 1e-6, 1 - 1e-6),
               np.clip(predict(splits["test"]), 1e-6, 1 - 1e-6), seconds=time.time() - t)


def main(stages: list[str]):
    t = time.time()
    daily = load_daily()
    splits = load_strategic_splits(exclude_covid=True, daily=daily)
    airports = eligible_airports()
    log(f"data in {time.time() - t:.0f}s: train {len(splits['train']):,} (prevalence "
        f"{splits['train']['label'].mean():.3f}), validation {len(splits['validation']):,} "
        f"({splits['validation']['label'].mean():.3f}), test {len(splits['test']):,}")
    rec = Recorder(splits)
    M = daily_matrix(daily, airports) if {"sequence", "graph"} & set(stages) else None
    for stage in stages:
        try:
            {"reference": lambda: run_reference(splits, rec),
             "disruptnet": lambda: run_disruptnet(splits, rec),
             "sequence": lambda: run_sequence(splits, rec, M, airports),
             "graph": lambda: run_graph(splits, rec, M, airports),
             "sarima": lambda: run_sarima(splits, rec, airports)}[stage]()
        except Exception as e:  # one failed stage shouldn't lose the rest of an unattended run
            log(f"{stage}: FAILED - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    r = pd.read_csv(RESULTS_PATH)
    cols = ["roc_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred", "cost"]
    pd.set_option("display.width", 220)
    print("\nVALIDATION, mean (std) over seeds")
    print(r.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["reference", "disruptnet", "sequence", "graph", "sarima"])
