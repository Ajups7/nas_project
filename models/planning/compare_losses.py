"""
Standard loss vs FAOC-Loss, same architecture and features, on VALIDATION.

FAOC-Loss deliberately inflates probabilities (a miss costs ~8-16x a false
alarm), so log loss / calibration are reported but are not its target. The
comparison that matters is the dollar cost of the decisions each model
leads to, under three decision rules:

- "0.5":     flag if p >= 0.5
- "bayes":   flag if p >= fp / (fp + fn * dm_airport) - the cost-optimal
             rule IF p is calibrated (~6-8% here)
- "tuned":   the single threshold minimizing validation cost for that
             model (each model's best case; optimistic, equally for all)

Run from the repo root, after `python -m models.planning.train_disruptnet faoc`:
    python -m models.planning.compare_losses
"""

import numpy as np
import pandas as pd
import torch

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.data import load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.planning_disruptnet import PlanningDisruptNet
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR

MODELS = ["standard_pw1_all", "faoc_all"]
SEEDS = (0, 1, 2)
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def predict_from_checkpoint(path, df: pd.DataFrame) -> np.ndarray:
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    x = (df[cfg["feature_columns"]].to_numpy() - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    model = PlanningDisruptNet(len(cfg["feature_columns"]), cfg["hidden_dim"], pos_weight=cfg["pos_weight"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def main():
    val = load_planning_splits(exclude_covid=True)["validation"]
    y, airports = val["label"].to_numpy(), val["airport"].to_numpy()
    faoc = FAOCLoss()
    dm = np.array([faoc.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in airports])
    bayes_threshold = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
    missing = sorted(set(airports) - set(faoc.multipliers))
    print(f"FAOC weights: miss ${faoc.fn_weight:,.0f} x multiplier, false alarm ${faoc.fp_weight:,.0f}")
    print(f"cost-ratio thresholds: {bayes_threshold.min():.3f} - {bayes_threshold.max():.3f}")
    print(f"airports using the national-composite multiplier: {missing or 'none'}\n")

    rows = []
    for name, flags in (("always flag", np.ones_like(y)), ("never flag", np.zeros_like(y))):
        rows.append({"model": name, "seed": -1, "rule": "-", "threshold": np.nan,
                     **decision_costs(y, flags, airports, faoc)})

    prob_rows = []
    for name in MODELS:
        for seed in SEEDS:
            p = predict_from_checkpoint(CHECKPOINT_DIR / f"{name}_seed{seed}.pt", val)
            prob_rows.append({"model": name, "seed": seed, **evaluate(y, p, airports)})

            grid_costs = [decision_costs(y, p >= t, airports, faoc)["cost_per_airport_day"]
                          for t in THRESHOLD_GRID]
            tuned = THRESHOLD_GRID[int(np.argmin(grid_costs))]
            for rule, thr in (("0.5", 0.5), ("bayes", bayes_threshold), ("tuned", tuned)):
                rows.append({"model": name, "seed": seed, "rule": rule,
                             "threshold": float(np.mean(thr)),
                             **decision_costs(y, p >= thr, airports, faoc)})

    costs = pd.DataFrame(rows)
    probs = pd.DataFrame(prob_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    costs.to_csv(RESULTS_DIR / "loss_comparison_costs_validation.csv", index=False)
    probs.to_csv(RESULTS_DIR / "loss_comparison_probs_validation.csv", index=False)

    pd.set_option("display.width", 200)
    print("DECISION COST on validation (mean over 3 seeds; $ per airport-day)")
    cols = ["cost_per_airport_day", "flag_rate", "cost_weighted_recall", "miss_cost_share", "threshold"]
    summary = costs.groupby(["model", "rule"], sort=False)[cols].mean()
    summary["cost_std"] = costs.groupby(["model", "rule"], sort=False)["cost_per_airport_day"].std()
    print(summary.round(3).to_string())
    print("\nPROBABILITY QUALITY (mean over 3 seeds)")
    pcols = ["roc_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred"]
    print(probs.groupby("model", sort=False)[pcols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
