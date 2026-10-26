"""
Trains PlanningDisruptNet (MoE router, models/planning/planning_disruptnet.py)
on the real Planning dataset. COVID period excluded from train by default
(models/planning/data.py). Early stopping on validation log loss (unweighted
BCE, so it's comparable across loss types). Test is never touched here.

Run from the repo root:
    python -m models.planning.train_disruptnet standard   # loss weighting / recent-history sweep
    python -m models.planning.train_disruptnet anomaly    # recent-anomaly features
    python -m models.planning.train_disruptnet faoc       # standard vs FAOC-Loss, same architecture
    python -m models.planning.train_disruptnet final      # frozen config: + airport identity, both losses
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.planning import ANOMALY_COLUMNS, FEATURE_COLUMNS, RECENT_COLUMNS
from models.planning.data import AIRPORT_COLUMNS, Standardizer, load_planning_splits
from models.planning.metrics import evaluate, results_table
from models.planning.planning_disruptnet import PlanningDisruptNet

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = HERE / "checkpoints"
NO_RECENT = [c for c in FEATURE_COLUMNS if c not in RECENT_COLUMNS]
WITH_ANOMALY = NO_RECENT + ANOMALY_COLUMNS


@dataclass
class Config:
    name: str
    feature_columns: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    loss_type: str = "standard"
    pos_weight: float = 1.0
    hidden_dim: int = 32
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 6
    seed: int = 0
    exclude_covid: bool = True


def _predict(model, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).numpy()


def train_one(cfg: Config, splits: dict) -> tuple[PlanningDisruptNet, Standardizer, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train, val = splits["train"], splits["validation"]
    scaler = Standardizer(cfg.feature_columns).fit(train)
    x_tr = torch.from_numpy(scaler.transform(train))
    y_tr = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
    a_tr = train["airport"].to_numpy()
    x_va = torch.from_numpy(scaler.transform(val))
    y_va = val["label"].to_numpy()
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    a_va = list(val["airport"])

    model = PlanningDisruptNet(len(cfg.feature_columns), cfg.hidden_dim, pos_weight=cfg.pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = model.compute_loss(x_tr[idx], y_tr[idx], list(a_tr[idx.numpy()]), cfg.loss_type)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Early stopping on the model's own objective: for loss_type="standard"
        # with pos_weight=1 this is plain BCE; FAOC is judged by FAOC.
        model.eval()
        with torch.no_grad():
            val_loss = float(model.compute_loss(x_va, y_va_t, a_va, cfg.loss_type))
        if val_loss < best_loss * (1 - 1e-4):
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    metrics = evaluate(y_va, _predict(model, x_va), val["airport"])
    metrics.update(best_epoch=best_epoch, epochs_run=epoch + 1)
    return model, scaler, metrics


def save_checkpoint(cfg: Config, model: PlanningDisruptNet, scaler: Standardizer) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{cfg.name}_seed{cfg.seed}.pt"
    torch.save({
        "config": cfg.__dict__,
        "state_dict": model.state_dict(),
        "scaler_mean": scaler.scaler.mean_,
        "scaler_scale": scaler.scaler.scale_,
    }, path)
    return path


def run_sweep(configs: list[Config], seeds=(0, 1, 2), out_name: str = "disruptnet_validation") -> pd.DataFrame:
    splits_cache = {}
    rows = {}
    for cfg in configs:
        if cfg.exclude_covid not in splits_cache:
            splits_cache[cfg.exclude_covid] = load_planning_splits(cfg.exclude_covid)
        for seed in seeds:
            cfg.seed = seed
            t = time.time()
            model, scaler, m = train_one(cfg, splits_cache[cfg.exclude_covid])
            save_checkpoint(cfg, model, scaler)
            rows[(cfg.name, seed)] = m
            print(f"{cfg.name:32s} seed {seed}: within-AUC {m['within_airport_auc']:.3f}  "
                  f"pooled AUC {m['roc_auc']:.3f}  brier {m['brier']:.4f}  ece {m['ece']:.3f}  "
                  f"mean_pred {m['mean_pred']:.3f}  best epoch {m['best_epoch']}/{m['epochs_run']}  "
                  f"({time.time() - t:.0f}s)", flush=True)

    per_seed = results_table(rows)
    per_seed.index = pd.MultiIndex.from_tuples(per_seed.index, names=["config", "seed"])
    RESULTS_DIR.mkdir(exist_ok=True)
    per_seed.to_csv(RESULTS_DIR / f"{out_name}_per_seed.csv")
    cols = ["roc_auc", "pr_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred"]
    summary = per_seed[cols].astype(float).groupby(level="config", sort=False).agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / f"{out_name}_summary.csv")
    return summary


def main(sweep: str):
    if sweep == "standard":
        train = load_planning_splits(exclude_covid=True)["train"]
        prevalence = train["label"].mean()
        pos_ratio = (1 - prevalence) / prevalence
        print(f"train prevalence {prevalence:.3f} -> balancing pos_weight {pos_ratio:.2f}\n")
        configs = [
            Config("standard_pw1_all"),
            Config("standard_pw1_norecent", feature_columns=list(NO_RECENT)),
            Config("standard_pwbal_all", pos_weight=pos_ratio),
        ]
    elif sweep == "final":
        cols = list(FEATURE_COLUMNS + AIRPORT_COLUMNS)
        configs = [
            Config("final_standard", feature_columns=cols),
            Config("final_faoc", feature_columns=cols, loss_type="faoc"),
        ]
    elif sweep == "faoc":
        configs = [
            Config("standard_pw1_all"),
            Config("faoc_all", loss_type="faoc"),
        ]
    elif sweep == "anomaly":
        configs = [
            Config("standard_pw1_anomaly", feature_columns=list(WITH_ANOMALY)),
            Config("standard_pw1_all_plus_anomaly", feature_columns=list(FEATURE_COLUMNS + ANOMALY_COLUMNS)),
        ]
    else:
        raise ValueError(f"unknown sweep {sweep!r}")
    summary = run_sweep(configs, out_name=f"disruptnet_{sweep}_validation")
    pd.set_option("display.width", 220)
    print("\nVALIDATION, mean ± std over 3 seeds")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "standard")
