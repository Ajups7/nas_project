"""
FINAL test-set evaluation for the Strategic tier. Runs ONCE (marker file).

Reads the sealed test predictions written at training time
(models/strategic/train.py -> results/predictions/test_*.parquet); nothing
was scored on test before this. Decision rules come from train/validation
only: the cost-ratio threshold (FAOC: 0.5), plus a threshold tuned on the
VALIDATION predictions and then applied to test.

Run from the repo root:
    python -m models.strategic.evaluate_test
"""

from datetime import datetime

import numpy as np
import pandas as pd

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.metrics import decision_costs, evaluate
from models.strategic.data import load_strategic_splits
from models.strategic.train import PRED_DIR, RESULTS_DIR, RESULTS_PATH

MARKER = RESULTS_DIR / "TEST_EVALUATED.txt"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def main():
    if MARKER.exists():
        raise SystemExit(f"test set already evaluated ({MARKER.read_text().strip()}); refusing to run again")
    splits = load_strategic_splits(exclude_covid=True)
    va, te = splits["validation"], splits["test"]
    runs = pd.read_csv(RESULTS_PATH)[["model", "seed", "rule"]].drop_duplicates(["model", "seed"])
    faoc = FAOCLoss()

    def bayes(a):
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
        return faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)

    y_va, a_va = va["label"].to_numpy(), va["airport"].to_numpy()
    y_te, a_te = te["label"].to_numpy(), te["airport"].to_numpy()
    thr_te = bayes(a_te)
    year = te["date"].dt.year.to_numpy()
    MARKER.write_text(f"evaluated {datetime.now():%Y-%m-%d %H:%M}\n")

    rows = [{"model": n, "seed": 0, "cost_rule": decision_costs(y_te, f, a_te, faoc)["cost_per_airport_day"]}
            for n, f in (("never flag", np.zeros_like(y_te)), ("always flag", np.ones_like(y_te)))]
    preds = {}
    for model, seed, rule in runs.itertuples(index=False):
        p_va = pd.read_parquet(PRED_DIR / f"val_{model}_seed{seed}.parquet")["p"].to_numpy()
        p_te = pd.read_parquet(PRED_DIR / f"test_{model}_seed{seed}.parquet")["p"].to_numpy()
        assert len(p_va) == len(va) and len(p_te) == len(te)
        tuned = GRID[int(np.argmin([decision_costs(y_va, p_va >= t, a_va, faoc)["cost_per_airport_day"]
                                    for t in GRID]))]
        primary = 0.5 if rule == "0.5" else thr_te
        r = {"model": model, "seed": seed, **evaluate(y_te, p_te, a_te),
             "cost_rule": decision_costs(y_te, p_te >= primary, a_te, faoc)["cost_per_airport_day"],
             "val_tuned_threshold": float(tuned),
             "cost_val_tuned": decision_costs(y_te, p_te >= tuned, a_te, faoc)["cost_per_airport_day"]}
        for yr in (2024, 2025):
            m = year == yr
            pt = primary if np.isscalar(primary) else primary[m]
            r[f"cost_rule_{yr}"] = decision_costs(y_te[m], p_te[m] >= pt, a_te[m], faoc)["cost_per_airport_day"]
            r[f"within_auc_{yr}"] = evaluate(y_te[m], p_te[m], a_te[m])["within_airport_auc"]
            r[f"mean_pred_{yr}"], r[f"prevalence_{yr}"] = float(p_te[m].mean()), float(y_te[m].mean())
        rows.append(r)
        preds.setdefault(model, []).append(p_te)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "test_results_per_seed.csv", index=False)
    out = te[["airport", "date", "label"]].copy()
    for model, ps in preds.items():
        out[model] = np.mean(ps, axis=0)
    out.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)

    cols = ["roc_auc", "within_airport_auc", "log_loss", "ece", "mean_pred", "cost_rule", "cost_val_tuned",
            "cost_rule_2024", "cost_rule_2025", "mean_pred_2025", "prevalence_2025"]
    g = res.groupby("model", sort=False)
    summary = g[cols].mean()
    summary["cost_rule_std"] = g["cost_rule"].std()
    summary.to_csv(RESULTS_DIR / "test_results_summary.csv")
    pd.set_option("display.width", 240)
    print(f"TEST: {len(te):,} airport-days, prevalence {y_te.mean():.4f} "
          f"(2024 {y_te[year == 2024].mean():.4f}, 2025 {y_te[year == 2025].mean():.4f})")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
