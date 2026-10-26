"""
Trains the sector-graph-attention Regime Transformer
(models/planning/regime_transformer.py) on real data.

One sample = one date: every airport's past-30-day window (the same inputs
as Seq2Seq/TFT, models/planning/sequences.py) -> one probability per
airport. Airports attend to rotation-chain neighbors
(models/planning/rotation_adjacency.py), then to 4 learned regime
embeddings. Airport-dates without a label (dropped by planning_label.py,
or COVID in train) are masked out of the loss and the metrics, so scoring
uses exactly the validation rows every other Planning model is scored on.

Graph ablation: "top10" (strongest rotation partners), "full" (any
flights), "none" (self only). If the rotation graph doesn't beat "none",
propagation over rotations isn't adding anything at this horizon.

Run from the repo root (after `python -m models.planning.rotation_adjacency`):
    python -m models.planning.train_regime_transformer [graph ...] [--airport-onehot]
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD, gdp_days
from features.planning import DATASET_PATH, eligible_airports
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.data import load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.regime_transformer import SectorGraphAttentionRegimeTransformer
from models.planning.rotation_adjacency import adjacency, load_counts
from models.planning.sequences import SEQ_LEN, daily_matrices
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR

SEEDS = (0, 1, 2)
BATCH_DATES = 16
LR = 1e-3
MAX_EPOCHS = 60
PATIENCE = 6


def with_airport_onehot(M: np.ndarray) -> np.ndarray:
    """Appends a constant one-hot airport-identity block to (A, D, F)."""
    a = M.shape[0]
    eye = np.broadcast_to(np.eye(a, dtype=np.float32)[:, None, :], (a, M.shape[1], a))
    return np.concatenate([M, eye], axis=2)


class DateGraphData:
    """Daily feature tensor M (airports, days, features) + label matrix
    (days, airports) per split, NaN where an airport-date isn't scored.
    The constructor builds the Planning tier's data; other tiers use
    from_arrays with their own daily feature matrix."""

    def __init__(self, airport_onehot: bool = False, threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
                 dataset_path=DATASET_PATH, exclude_covid: bool = True):
        airports = eligible_airports(threshold)
        mats = daily_matrices(threshold=threshold)
        M = np.stack([mats[a] for a in airports])
        if airport_onehot:
            M = with_airport_onehot(M)
        frames = load_planning_splits(exclude_covid=exclude_covid, dataset_path=dataset_path)
        self._setup(M, frames, gdp_days(threshold).index, airports, SEQ_LEN,
                    len(airports) if airport_onehot else 0)

    @classmethod
    def from_arrays(cls, M: np.ndarray, frames: dict, calendar: pd.DatetimeIndex, airports: list[str],
                    seq_len: int, n_identity: int = 0) -> "DateGraphData":
        """M: (airports, len(calendar), features), no NaN; the last
        n_identity feature columns (one-hot identity) are left unscaled."""
        obj = cls.__new__(cls)
        obj._setup(M, frames, calendar, airports, seq_len, n_identity)
        return obj

    def _setup(self, M, frames, calendar, airports, seq_len, n_identity):
        _, order = load_counts()
        assert order == airports, "adjacency airport order must match the data's airport order"
        assert not np.isnan(M).any(), "fill missing values before building DateGraphData"
        self.airports, self.calendar, self.seq_len, self.frames = airports, calendar, seq_len, frames
        pos = pd.Series(np.arange(len(calendar)), index=calendar)
        self.labels, self.days = {}, {}
        col = {a: i for i, a in enumerate(airports)}
        for name, df in frames.items():
            L = np.full((len(calendar), len(airports)), np.nan, dtype=np.float32)
            d = pos.reindex(df["date"]).to_numpy()
            L[d, df["airport"].map(col).to_numpy()] = df["label"].to_numpy()
            self.labels[name], self.days[name] = L, np.unique(d[d >= seq_len - 1])

        train_days = self.days["train"]
        window_days = np.unique((train_days[:, None] + np.arange(-seq_len + 1, 1)[None, :]).ravel())
        sample = M[:, window_days].reshape(-1, M.shape[2])
        mean, std = sample.mean(0), sample.std(0) + 1e-6
        if n_identity:
            mean[-n_identity:], std[-n_identity:] = 0.0, 1.0
        self.M = torch.from_numpy(((M - mean) / std).astype(np.float32))

    def batch(self, split: str, days: np.ndarray):
        window = days[:, None] + np.arange(-self.seq_len + 1, 1)[None, :]  # (B, T)
        x = self.M[:, window].permute(1, 0, 2, 3)  # (B, A, T, F)
        y = torch.from_numpy(self.labels[split][days])  # (B, A)
        return x, y


def masked_bce(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = ~torch.isnan(y)
    p = p[m].clamp(1e-7, 1 - 1e-7)
    t = y[m]
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()


def predict(model, data: DateGraphData, split: str, adj: torch.Tensor) -> np.ndarray:
    model.eval()
    days = data.days[split]
    out = np.full((len(data.calendar), len(data.airports)), np.nan, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(days), 64):
            d = days[i:i + 64]
            x, _ = data.batch(split, d)
            out[d] = model(x, adj).numpy()
    return out


def scored_rows(data: DateGraphData, split: str, P: np.ndarray) -> pd.DataFrame:
    df = data.frames[split]
    pos = pd.Series(np.arange(len(data.calendar)), index=data.calendar)
    col = {a: i for i, a in enumerate(data.airports)}
    d = pos.reindex(df["date"]).to_numpy()
    ok = d >= data.seq_len - 1
    df = df[ok].reset_index(drop=True)
    p = P[d[ok], df["airport"].map(col).to_numpy()]
    return df.assign(p=p)


def train_one(data: DateGraphData, graph: str, seed: int, model_cls=SectorGraphAttentionRegimeTransformer,
              resample_train=None):
    """model_cls: any (num_features) -> module with forward(x, adjacency).
    resample_train: optional () -> array of training 'days' drawn fresh each
    epoch (for data too large to sweep fully every epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    adj = torch.from_numpy(adjacency(graph))
    model = model_cls(data.M.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_days = data.days["train"]
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        t = time.time()
        model.train()
        if resample_train is not None:
            train_days = resample_train()
        perm = rng.permutation(train_days)
        for i in range(0, len(perm), BATCH_DATES):
            x, y = data.batch("train", perm[i:i + BATCH_DATES])
            loss = masked_bce(model(x, adj), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        P = predict(model, data, "validation", adj)
        v = data.days["validation"]
        val_loss = float(masked_bce(torch.from_numpy(P[v]), torch.from_numpy(data.labels["validation"][v])))
        if epoch == 0:
            print(f"    epoch time {time.time() - t:.0f}s", flush=True)
        if val_loss < best * (1 - 1e-4):
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v_.clone() for k, v_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, adj, best_epoch, epoch + 1


def main(graphs: list[str], airport_onehot: bool):
    t = time.time()
    data = DateGraphData(airport_onehot=airport_onehot)
    tag = "_onehot" if airport_onehot else ""
    print(f"data built in {time.time() - t:.0f}s: M {tuple(data.M.shape)}, "
          f"train dates {len(data.days['train'])}, validation dates {len(data.days['validation'])}", flush=True)

    faoc = FAOCLoss()
    rows = []
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for graph in graphs:
        for seed in SEEDS:
            ts = time.time()
            model, adj, best_epoch, epochs = train_one(data, graph, seed)
            torch.save({"graph": graph, "airport_onehot": airport_onehot, "state_dict": model.state_dict()},
                       CHECKPOINT_DIR / f"regime_transformer_{graph}{tag}_seed{seed}.pt")
            val = scored_rows(data, "validation", predict(model, data, "validation", adj))
            y, a, p = val["label"].to_numpy(), val["airport"].to_numpy(), val["p"].to_numpy()
            dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
            bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
            m = evaluate(y, p, a)
            cost = decision_costs(y, p >= bayes, a, faoc)["cost_per_airport_day"]
            rows.append({"graph": graph + tag, "seed": seed, "best_epoch": best_epoch, "epochs": epochs,
                         "rows_scored": len(val), **m, "cost": cost})
            print(f"{graph + tag:13s} seed {seed}: within-AUC {m['within_airport_auc']:.3f}  AUC {m['roc_auc']:.3f}  "
                  f"log loss {m['log_loss']:.3f}  ece {m['ece']:.3f}  cost ${cost:.0f}  "
                  f"best epoch {best_epoch}/{epochs}  rows {len(val):,}  ({time.time() - ts:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"regime_transformer_validation{tag}.csv"
    res.to_csv(out, index=False)
    cols = ["roc_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred", "cost"]
    print("\nVALIDATION, mean (std) over 3 seeds")
    print(res.groupby("graph", sort=False)[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    onehot = "--airport-onehot" in args
    graphs = [a for a in args if not a.startswith("--")] or ["top10", "full", "none"]
    main(graphs, onehot)
