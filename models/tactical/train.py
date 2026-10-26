"""
Tactical tier (0-2 h): train and score every model on VALIDATION, from the
hourly UTC tensor (features/tactical.py). Label: a GDP hour at t+1 or t+2.
Every metric is reported twice: on all scored hours, and on the ONSET
subset (hours not already in a GDP) - the case the 90-minute CDM window is
about, where persistence can't help.

Same protocol as the other tiers: COVID excluded from train, airport
identity in every learned model, 3 seeds, early stopping on validation,
cost-ratio threshold (FAOC: 0.5), test predictions written sealed to
results/predictions/ and scored only by models/tactical/evaluate_test.py.

At ~5M airport-hours, the sequence and graph models train on a random
sample of rows / hours each epoch and early-stop on a fixed validation
sample; final predictions cover every validation and test row.

Models:
  reference   persistence P(label | airport, GDP hour now), climatology
              (airport x month x local hour), logistic regression
  disruptnet  MoE DisruptNet on the hourly rows, standard + FAOC
  sequence    Seq2Seq, TFT (= Person A's TacticalDisruptNet with plain BCE)
              on 12-hour windows
  graph       rotation-chain STGNN (top-10 / full / no graph) on 12-hour
              windows of all airports at once
SARIMA is not run here: an hourly seasonal model per airport (~60k points,
period 24) is costly, and it was the weakest learned baseline at both
other tiers.

Run from the repo root (after `python -m features.tactical`), one stage
per process to bound memory:
    python -m models.tactical.train reference
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

import models.planning.train_regime_transformer as rt
from common.temporal_split import TRAIN_END, VALIDATION_END, assign_temporal_split
from features.tactical import TENSOR_PATH
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import Seq2SeqBaseline, TFTBaseline
from models.planning.data import in_covid
from models.planning.metrics import decision_costs, evaluate
from models.planning.train_disruptnet import Config, train_one
from models.strategic.rotation_gnn import RotationChainSpatioTemporalGNN

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PRED_DIR = RESULTS_DIR / "predictions"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULTS_PATH = RESULTS_DIR / "tactical_validation.csv"
SEEDS = (0, 1, 2)
SEQ_LEN = 12
GRAPHS = ("top10", "full", "none")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class TacticalData:
    """Filled, standardized tensor + scored (airport, hour) rows per split."""

    def __init__(self, path=TENSOR_PATH, exclude_covid: bool = True):
        z = np.load(path, allow_pickle=False)
        X, self.label, self.gdp = z["X"], z["label"], z["gdp_hour"]
        self.airports, self.grid = list(z["airports"]), pd.DatetimeIndex(z["grid"])
        self.names = list(z["feature_names"])
        A, T, _ = X.shape
        hour_split = assign_temporal_split(pd.Series(self.grid.normalize()), TRAIN_END, VALIDATION_END).to_numpy()
        covid = in_covid(pd.Series(self.grid.normalize())).to_numpy()
        scored = ~np.isnan(self.label)
        self.rows = {}
        for s in ("train", "validation", "test"):
            cols = (hour_split == s) & (~covid if (s == "train" and exclude_covid) else True)
            a, t = np.nonzero(scored & cols[None, :])
            self.rows[s] = (a, t)
        # Any arrival held with an EDCT this hour: the program is already
        # issued. "Clean onset" rows exclude these (see Recorder).
        self.edct_now = np.nan_to_num(X[:, :, self.names.index("arr_edct_share")]) > 0
        a, t = self.rows["train"]
        flat = X[a, t]
        self.median = np.nanmedian(flat, axis=0)
        X = np.where(np.isnan(X), self.median[None, None, :], X)
        flat = X[a, t]
        self.mean, self.std = flat.mean(0), flat.std(0) + 1e-6
        self.Xs = ((X - self.mean) / self.std).astype(np.float32)  # (A, T, F), standardized
        self.n_features = self.Xs.shape[2] + A
        self.hour_split = hour_split

    def tabular(self, split: str) -> pd.DataFrame:
        a, t = self.rows[split]
        df = pd.DataFrame(self.Xs[a, t], columns=self.names)
        onehot = np.zeros((len(a), len(self.airports)), np.float32)
        onehot[np.arange(len(a)), a] = 1
        df[[f"ap_{x}" for x in self.airports]] = onehot
        df["airport"] = np.array(self.airports)[a]
        df["label"] = self.label[a, t]
        df["onset"] = self.gdp[a, t] == 0
        return df

    def windows(self, a: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """(n, SEQ_LEN, features + one-hot airport) for rows (a, t)."""
        idx = t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        idx = np.clip(idx, 0, self.Xs.shape[1] - 1)
        x = self.Xs[a[:, None], idx]
        onehot = np.zeros((len(a), SEQ_LEN, len(self.airports)), np.float32)
        onehot[np.arange(len(a)), :, a] = 1
        return torch.from_numpy(np.concatenate([x, onehot], axis=2))


class Recorder:
    def __init__(self, data: TacticalData):
        a, t = data.rows["validation"]
        self.y, self.a = data.label[a, t], np.array(data.airports)[a]
        self.onset = data.gdp[a, t] == 0
        self.clean = self.onset & ~data.edct_now[a, t]
        self.faoc = FAOCLoss()
        dm = np.array([self.faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in self.a])
        self.bayes = self.faoc.fp_weight / (self.faoc.fp_weight + self.faoc.fn_weight * dm)
        self.done = set()
        if RESULTS_PATH.exists():
            r = pd.read_csv(RESULTS_PATH)
            self.done = set(zip(r["model"], r["seed"]))

    def record(self, model: str, seed: int, p_val, p_test, rule: str = "rule", seconds: float = 0.0,
               best_epoch: int | None = None):
        p_val = np.clip(np.asarray(p_val, dtype=np.float64), 1e-7, 1 - 1e-7)
        thr = 0.5 if rule == "0.5" else self.bayes
        m = evaluate(self.y, p_val, self.a)
        row = {"model": model, "seed": seed, "rule": rule, **m,
               "cost": decision_costs(self.y, p_val >= thr, self.a, self.faoc)["cost_per_airport_day"]}
        for tag, mask in (("onset", self.onset), ("clean", self.clean)):
            mm = evaluate(self.y[mask], p_val[mask], self.a[mask])
            tm = thr if np.isscalar(thr) else thr[mask]
            row.update({f"{tag}_{k}": v for k, v in mm.items()})
            row[f"{tag}_cost"] = decision_costs(self.y[mask], p_val[mask] >= tm, self.a[mask],
                                                self.faoc)["cost_per_airport_day"]
        row.update(seconds=round(seconds, 1), best_epoch=best_epoch)
        mo = {"roc_auc": row["onset_roc_auc"], "within_airport_auc": row["onset_within_airport_auc"],
              "pr_auc": row["onset_pr_auc"]}
        RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"p": np.asarray(p_test, dtype=np.float64)}).to_parquet(
            PRED_DIR / f"test_{model}_seed{seed}.parquet", index=False)
        pd.DataFrame({"p": p_val}).to_parquet(PRED_DIR / f"val_{model}_seed{seed}.parquet", index=False)
        log(f"{model:20s} s{seed}: ALL within {m['within_airport_auc']:.3f} AUC {m['roc_auc']:.3f} "
            f"ll {m['log_loss']:.4f} cost ${row['cost']:.0f} | ONSET AUC {mo['roc_auc']:.3f} "
            f"within {mo['within_airport_auc']:.3f} pr-auc {mo['pr_auc']:.3f} cost ${row['onset_cost']:.0f} | "
            f"CLEAN AUC {row['clean_roc_auc']:.3f} pr-auc {row['clean_pr_auc']:.3f} "
            f"(prev {row['clean_prevalence']:.4f}) ({seconds:.0f}s)")


def run_reference(data: TacticalData, rec: Recorder):
    tr, va, te = data.tabular("train"), data.tabular("validation"), data.tabular("test")
    month = lambda s: data.grid[data.rows[s][1]].month.to_numpy()
    hod = lambda df: np.round(np.arctan2(df["hod_sin"] * data.std[0] + data.mean[0],
                                         df["hod_cos"] * data.std[1] + data.mean[1]) * 24 / (2 * np.pi)) % 24

    def lookup(keys_tr, keys_x):
        rate = tr["label"].groupby(keys_tr).mean()
        p = rate.reindex(pd.MultiIndex.from_arrays(keys_x)).to_numpy()
        return np.where(np.isnan(p), tr["label"].mean(), p)

    if ("persistence", 0) not in rec.done:
        gt = lambda df: (df["onset"] == False).astype(int)  # noqa: E712 - 1 = in GDP now
        rec.record("persistence", 0, lookup([tr["airport"], gt(tr)], [va["airport"], gt(va)]),
                   lookup([tr["airport"], gt(tr)], [te["airport"], gt(te)]))
    if ("climatology lookup", 0) not in rec.done:
        k = lambda df, s: [df["airport"], month(s), hod(df)]
        rec.record("climatology lookup", 0, lookup(k(tr, "train"), k(va, "validation")),
                   lookup(k(tr, "train"), k(te, "test")))
    if ("logistic", 0) not in rec.done:
        t = time.time()
        cols = data.names + [f"ap_{x}" for x in data.airports]
        lr = LogisticRegression(max_iter=500, tol=1e-4).fit(tr[cols].to_numpy(), tr["label"].to_numpy())
        rec.record("logistic", 0, lr.predict_proba(va[cols].to_numpy())[:, 1],
                   lr.predict_proba(te[cols].to_numpy())[:, 1], seconds=time.time() - t)


def run_disruptnet(data: TacticalData, rec: Recorder):
    cols = data.names + [f"ap_{x}" for x in data.airports]
    frames = {"train": data.tabular("train"), "validation": data.tabular("validation")}
    te = data.tabular("test")
    for loss_type in ("standard", "faoc"):
        name = f"disruptnet {loss_type}"
        for seed in SEEDS:
            if (name, seed) in rec.done:
                continue
            t = time.time()
            net, sc, m = train_one(Config(f"tactical_{loss_type}", feature_columns=cols, loss_type=loss_type,
                                          seed=seed, batch_size=4096, max_epochs=30, patience=4), frames)
            net.eval()
            with torch.no_grad():
                pv = net(torch.from_numpy(sc.transform(frames["validation"]))).numpy()
                pt = net(torch.from_numpy(sc.transform(te))).numpy()
            rec.record(name, seed, pv, pt, "0.5" if loss_type == "faoc" else "rule", time.time() - t,
                       m["best_epoch"])


def _predict_rows(model, data, split, batch=8192):
    a, t = data.rows[split]
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(a), batch):
            out.append(model(data.windows(a[i:i + batch], t[i:i + batch])).numpy())
    return np.concatenate(out)


def run_sequence(data: TacticalData, rec: Recorder, rows_per_epoch=400_000, val_sample=100_000,
                 max_epochs=20, patience=4):
    a_tr, t_tr = data.rows["train"]
    a_va, t_va = data.rows["validation"]
    for name, cls in (("seq2seq", Seq2SeqBaseline), ("tft", TFTBaseline)):
        for seed in SEEDS:
            if (name, seed) in rec.done:
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            vs = rng.choice(len(a_va), val_sample, replace=False)
            xv, yv = data.windows(a_va[vs], t_va[vs]), torch.from_numpy(data.label[a_va[vs], t_va[vs]])
            model = cls(data.n_features)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            best, state, best_epoch, bad = np.inf, None, 0, 0
            for epoch in range(max_epochs):
                model.train()
                idx = rng.choice(len(a_tr), rows_per_epoch, replace=False)
                for i in range(0, len(idx), 1024):
                    j = idx[i:i + 1024]
                    p = model(data.windows(a_tr[j], t_tr[j])).clamp(1e-7, 1 - 1e-7)
                    loss = F.binary_cross_entropy(p, torch.from_numpy(data.label[a_tr[j], t_tr[j]]))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    pv = torch.cat([model(xv[i:i + 8192]) for i in range(0, len(xv), 8192)]).clamp(1e-7, 1 - 1e-7)
                vl = float(F.binary_cross_entropy(pv, yv))
                if vl < best * (1 - 1e-4):
                    best, best_epoch, bad = vl, epoch, 0
                    state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            model.load_state_dict(state)
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save({"state_dict": state}, CHECKPOINT_DIR / f"{name}_seed{seed}.pt")
            rec.record(name, seed, _predict_rows(model, data, "validation"), _predict_rows(model, data, "test"),
                       seconds=time.time() - t0, best_epoch=best_epoch)


class _HourGraph:
    """Adapter so models.planning.train_regime_transformer's date-batched
    graph training runs on hours: 'days' are hour indices. drop: feature
    names neutralized (set to their train mean) at every airport."""

    def __init__(self, data: TacticalData, drop: tuple = ()):
        A, T = len(data.airports), data.Xs.shape[1]
        Xs = data.Xs
        if drop:
            Xs = Xs.copy()
            for n in drop:
                Xs[:, :, data.names.index(n)] = 0.0
        onehot = np.broadcast_to(np.eye(A, dtype=np.float32)[:, None, :], (A, T, A))
        self.M = torch.from_numpy(np.concatenate([Xs, onehot], axis=2))
        self.airports, self.calendar, self.seq_len = data.airports, data.grid, SEQ_LEN
        self.labels, self.days = {}, {}
        for s in ("train", "validation", "test"):
            a, t = data.rows[s]
            L = np.full((T, A), np.nan, np.float32)
            L[t, a] = data.label[a, t]
            self.labels[s], self.days[s] = L, np.unique(t[t >= SEQ_LEN - 1])

    def batch(self, split, hours):
        window = hours[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        return self.M[:, window].permute(1, 0, 2, 3), torch.from_numpy(self.labels[split][hours])


def run_graph(data: TacticalData, rec: Recorder, hours_per_epoch=2000, val_hours=2000,
              graphs=GRAPHS, drop: tuple = (), suffix: str = ""):
    g = _HourGraph(data, drop)
    full_days = dict(g.days)
    for graph in graphs:
        for seed in SEEDS:
            name = f"stgnn {graph}{suffix}"
            if (name, seed) in rec.done:
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            g.days = {"train": rng.choice(full_days["train"], hours_per_epoch, replace=False),
                      "validation": np.sort(rng.choice(full_days["validation"], val_hours, replace=False))}
            resample = lambda: rng.choice(full_days["train"], hours_per_epoch, replace=False)
            model, adj, best_epoch, _ = rt.train_one(g, graph, seed, model_cls=RotationChainSpatioTemporalGNN,
                                                     resample_train=resample)
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save({"graph": graph, "drop": drop, "state_dict": model.state_dict()},
                       CHECKPOINT_DIR / f"stgnn_{graph}{suffix.replace(' ', '_')}_seed{seed}.pt")
            preds = {}
            for split in ("validation", "test"):
                g.days[split] = full_days[split]
                P = rt.predict(model, g, split, adj)  # (T, A)
                a, t = data.rows[split]
                preds[split] = P[t, a]
            rec.record(name, seed, preds["validation"], preds["test"], seconds=time.time() - t0, best_epoch=best_epoch)


def main(stages: list[str]):
    t = time.time()
    data = TacticalData()
    log(f"data in {time.time() - t:.0f}s: " + ", ".join(
        f"{s} {len(data.rows[s][0]):,} rows (prev {np.nanmean(data.label[data.rows[s]]):.4f})"
        for s in ("train", "validation", "test")))
    rec = Recorder(data)
    # Departure EDCTs at a neighbor mean flights to *this* airport are already
    # being held - an issued program, visible through the graph before any
    # held flight arrives here. graph_nodep tests whether the graph's gain
    # survives without that signal.
    nodep = lambda d, r: run_graph(d, r, graphs=("top10", "none"), drop=("dep_edct_share",), suffix=" no-dep-edct")
    runners = {"reference": run_reference, "disruptnet": run_disruptnet, "sequence": run_sequence,
               "graph": run_graph, "graph_nodep": nodep}
    for stage in stages:
        try:
            runners[stage](data, rec)
        except Exception as e:  # keep an unattended chain going
            log(f"{stage}: FAILED - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:])
