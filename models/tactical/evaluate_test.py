"""
FINAL test-set evaluation for the Tactical tier. Runs ONCE (marker file).

Reads the sealed test predictions written at training time
(models/tactical/train.py -> results/predictions/test_*.parquet). Decision
rules from train/validation only: cost-ratio threshold (FAOC: 0.5), plus a
threshold tuned on VALIDATION predictions and applied to test. Every metric
is reported on all scored hours, on onset hours (not in a GDP now), and on
clean-onset hours (no arrival held with an EDCT yet).

Run from the repo root:
    python -m models.tactical.evaluate_test
"""

from datetime import datetime

import numpy as np
import pandas as pd

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.metrics import decision_costs, evaluate
from models.tactical.train import PRED_DIR, RESULTS_DIR, RESULTS_PATH, TacticalData

MARKER = RESULTS_DIR / "TEST_EVALUATED.txt"
GRID = np.round(np.concatenate([np.arange(0.005, 0.05, 0.005), np.arange(0.05, 1.0, 0.01)]), 3)


def main():
    if MARKER.exists():
        raise SystemExit(f"test set already evaluated ({MARKER.read_text().strip()}); refusing to run again")
    data = TacticalData()
    faoc = FAOCLoss()

    def subset(split):
        a, t = data.rows[split]
        y, ap = data.label[a, t], np.array(data.airports)[a]
        onset = data.gdp[a, t] == 0
        clean = onset & ~data.edct_now[a, t]
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in ap])
        return y, ap, onset, clean, faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm), data.grid[t].year

    y_va, a_va, _, _, _, _ = subset("validation")
    y_te, a_te, on_te, cl_te, thr_te, year = subset("test")
    runs = pd.read_csv(RESULTS_PATH)[["model", "seed", "rule"]].drop_duplicates(["model", "seed"])
    MARKER.write_text(f"evaluated {datetime.now():%Y-%m-%d %H:%M}\n")

    rows = [{"model": n, "seed": 0, "cost_rule": decision_costs(y_te, f, a_te, faoc)["cost_per_airport_day"]}
            for n, f in (("never flag", np.zeros_like(y_te)), ("always flag", np.ones_like(y_te)))]
    preds = {}
    for model, seed, rule in runs.itertuples(index=False):
        p_va = pd.read_parquet(PRED_DIR / f"val_{model}_seed{seed}.parquet")["p"].to_numpy()
        p_te = np.clip(pd.read_parquet(PRED_DIR / f"test_{model}_seed{seed}.parquet")["p"].to_numpy(), 1e-7, 1 - 1e-7)
        assert len(p_va) == len(y_va) and len(p_te) == len(y_te)
        tuned = GRID[int(np.argmin([decision_costs(y_va, p_va >= t, a_va, faoc)["cost_per_airport_day"]
                                    for t in GRID]))]
        primary = 0.5 if rule == "0.5" else thr_te
        r = {"model": model, "seed": seed, **evaluate(y_te, p_te, a_te),
             "cost_rule": decision_costs(y_te, p_te >= primary, a_te, faoc)["cost_per_airport_day"],
             "val_tuned_threshold": float(tuned),
             "cost_val_tuned": decision_costs(y_te, p_te >= tuned, a_te, faoc)["cost_per_airport_day"]}
        for tag, m in (("onset", on_te), ("clean", cl_te), ("y2024", year == 2024), ("y2025", year == 2025)):
            ev = evaluate(y_te[m], p_te[m], a_te[m])
            pm = primary if np.isscalar(primary) else primary[m]
            r.update({f"{tag}_roc_auc": ev["roc_auc"], f"{tag}_pr_auc": ev["pr_auc"],
                      f"{tag}_within_auc": ev["within_airport_auc"], f"{tag}_prevalence": ev["prevalence"],
                      f"{tag}_mean_pred": ev["mean_pred"],
                      f"{tag}_cost": decision_costs(y_te[m], p_te[m] >= pm, a_te[m], faoc)["cost_per_airport_day"]})
        rows.append(r)
        preds.setdefault(model, []).append(p_te)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "test_results_per_seed.csv", index=False)
    a, t = data.rows["test"]
    out = pd.DataFrame({"airport": np.array(data.airports)[a], "utc_hour": data.grid[t], "label": y_te,
                        "onset": on_te, "clean_onset": cl_te})
    for model, ps in preds.items():
        out[model] = np.mean(ps, axis=0)
    out.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)

    cols = ["roc_auc", "within_airport_auc", "log_loss", "ece", "mean_pred", "cost_rule", "cost_val_tuned",
            "onset_roc_auc", "onset_pr_auc", "onset_cost", "clean_roc_auc", "clean_pr_auc", "clean_cost",
            "y2025_cost", "y2025_mean_pred", "y2025_prevalence"]
    g = res.groupby("model", sort=False)
    summary = g[cols].mean()
    summary["cost_rule_std"] = g["cost_rule"].std()
    summary.to_csv(RESULTS_DIR / "test_results_summary.csv")
    pd.set_option("display.width", 260)
    print(f"TEST: {len(y_te):,} airport-hours, prevalence {y_te.mean():.4f}; onset rows {int(on_te.sum()):,} "
          f"(prevalence {y_te[on_te].mean():.4f}); clean-onset rows {int(cl_te.sum()):,} "
          f"(prevalence {y_te[cl_te].mean():.4f})")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
