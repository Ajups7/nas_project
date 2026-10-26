"""
Trains the paper's neural baselines (models/planning/baselines.py: Seq2Seq,
simplified TFT) on 30-day sequence windows (models/planning/sequences.py).
Same protocol as PlanningDisruptNet's standard run: plain BCE, Adam,
early stopping on validation BCE, 3 seeds, COVID excluded from train.
Scored on validation only.

Run from the repo root:
    python -m models.planning.train_sequence_baselines [--airport-onehot]
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import Seq2SeqBaseline, TFTBaseline
from models.planning.metrics import decision_costs, evaluate
from models.planning.sequences import build_sequence_splits, standardize
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR

MODELS = {"seq2seq": Seq2SeqBaseline, "tft": TFTBaseline}
SEEDS = (0, 1, 2)
BATCH_SIZE = 512
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 5


def _predict(model, X: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()


def train_one(model_cls, seed: int, data: dict):
    torch.manual_seed(seed)
    X_tr, y_tr, _, _ = data["train"]
    X_va, y_va, _, _ = data["validation"]
    X_tr, X_va = torch.from_numpy(X_tr), torch.from_numpy(X_va)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    model = model_cls(X_tr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        t = time.time()
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            p = model(X_tr[idx]).clamp(1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(p, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        p_va = torch.from_numpy(np.clip(_predict(model, X_va), 1e-7, 1 - 1e-7))
        val_loss = float(F.binary_cross_entropy(p_va, y_va_t))
        if epoch == 0:
            print(f"    epoch time {time.time() - t:.0f}s", flush=True)
        if val_loss < best * (1 - 1e-4):
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, _predict(model, X_va), best_epoch, epoch + 1


def main(airport_onehot: bool = False):
    tag = "_onehot" if airport_onehot else ""
    t = time.time()
    data = standardize(build_sequence_splits(airport_onehot=airport_onehot))
    print(f"sequences built in {time.time() - t:.0f}s; train {data['train'][0].shape}", flush=True)
    _, y_va, a_va, _ = data["validation"]

    faoc = FAOCLoss()
    dm = np.array([faoc.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in a_va])
    bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)

    rows = []
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for name, cls in MODELS.items():
        for seed in SEEDS:
            t = time.time()
            model, p, best_epoch, epochs = train_one(cls, seed, data)
            torch.save({"model": name, "state_dict": model.state_dict()},
                       CHECKPOINT_DIR / f"baseline_{name}{tag}_seed{seed}.pt")
            m = evaluate(y_va, p, a_va)
            c = decision_costs(y_va, p >= bayes, a_va, faoc)
            rows.append({"model": name, "seed": seed, "best_epoch": best_epoch, "epochs": epochs,
                         **m, "cost_bayes": c["cost_per_airport_day"]})
            print(f"{name:8s} seed {seed}: within-AUC {m['within_airport_auc']:.3f}  pooled AUC {m['roc_auc']:.3f}  "
                  f"log loss {m['log_loss']:.3f}  brier {m['brier']:.4f}  ece {m['ece']:.3f}  "
                  f"cost ${c['cost_per_airport_day']:.0f}  best epoch {best_epoch}/{epochs}  "
                  f"({time.time() - t:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / f"sequence_baselines_validation{tag}.csv", index=False)
    cols = ["roc_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred", "cost_bayes"]
    print("\nVALIDATION, mean (std) over 3 seeds")
    print(res.groupby("model")[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    import sys

    main(airport_onehot="--airport-onehot" in sys.argv)
