"""
FINAL test-set evaluation for the Planning tier. Runs ONCE: it writes a
marker file and refuses to run again, so the test split can't drift into
model selection.

Nothing is trained or tuned on test. Every model is the frozen checkpoint
trained on train (COVID excluded) with early stopping on validation; every
decision rule comes from train/validation:
- "rule":  cost-ratio threshold (FAOC: 0.5) - fixed in advance
- "tuned": the threshold minimizing VALIDATION cost, then applied to test

Requires: train_disruptnet final, train_sequence_baselines --airport-onehot,
train_regime_transformer top10 full none --airport-onehot,
train_sarima --with-test.

Run from the repo root:
    python -m models.planning.evaluate_test
"""

from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import models.planning.train_regime_transformer as rt
from features.planning import FEATURE_COLUMNS
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import Seq2SeqBaseline, TFTBaseline
from models.planning.compare_losses import predict_from_checkpoint
from models.planning.data import AIRPORT_COLUMNS, Standardizer, load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.sequences import build_sequence_splits, standardize
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR
from models.planning.train_reference import climatology_lookup

MARKER = RESULTS_DIR / "TEST_EVALUATED.txt"
SEEDS = (0, 1, 2)
GRAPHS = ("top10", "full", "none")
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def main():
    if MARKER.exists():
        raise SystemExit(f"test set already evaluated ({MARKER.read_text().strip()}); refusing to run again")

    splits = load_planning_splits(exclude_covid=True)
    tr, va, te = splits["train"], splits["validation"], splits["test"]
    faoc = FAOCLoss()

    def bayes(a):
        dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
        return faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)

    thr_va, thr_te = bayes(va["airport"]), bayes(te["airport"])
    preds = {}  # (model, seed) -> (p_val, p_test, rule)

    preds[("climatology lookup", 0)] = (climatology_lookup(tr, va), climatology_lookup(tr, te), "rule")
    cols = FEATURE_COLUMNS + AIRPORT_COLUMNS
    sc = Standardizer(cols).fit(tr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(tr), tr["label"])
    preds[("logistic", 0)] = (lr.predict_proba(sc.transform(va))[:, 1], lr.predict_proba(sc.transform(te))[:, 1], "rule")

    for name, label, rule in (("final_standard", "disruptnet standard", "rule"),
                              ("final_faoc", "disruptnet faoc", "0.5")):
        for s in SEEDS:
            path = CHECKPOINT_DIR / f"{name}_seed{s}.pt"
            preds[(label, s)] = (predict_from_checkpoint(path, va), predict_from_checkpoint(path, te), rule)

    seq = standardize(build_sequence_splits(airport_onehot=True))
    assert np.array_equal(seq["validation"][1], va["label"].to_numpy())
    assert np.array_equal(seq["test"][1], te["label"].to_numpy())
    for name, cls in (("seq2seq", Seq2SeqBaseline), ("tft", TFTBaseline)):
        for s in SEEDS:
            model = cls(seq["train"][0].shape[-1])
            model.load_state_dict(torch.load(CHECKPOINT_DIR / f"baseline_{name}_onehot_seed{s}.pt")["state_dict"])
            model.eval()
            out = []
            for split in ("validation", "test"):
                X = torch.from_numpy(seq[split][0])
                with torch.no_grad():
                    out.append(torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)]).numpy())
            preds[(name, s)] = (out[0], out[1], "rule")
    del seq

    data = rt.DateGraphData(airport_onehot=True)
    for graph in GRAPHS:
        adj = torch.from_numpy(rt.adjacency(graph))
        for s in SEEDS:
            model = rt.SectorGraphAttentionRegimeTransformer(data.M.shape[2])
            model.load_state_dict(torch.load(CHECKPOINT_DIR / f"regime_transformer_{graph}_onehot_seed{s}.pt")["state_dict"])
            pv = rt.scored_rows(data, "validation", rt.predict(model, data, "validation", adj))
            pt = rt.scored_rows(data, "test", rt.predict(model, data, "test", adj))
            assert np.array_equal(pv["label"].to_numpy(), va["label"].to_numpy())
            assert np.array_equal(pt["label"].to_numpy(), te["label"].to_numpy())
            preds[(f"regime transformer {graph}", s)] = (pv["p"].to_numpy(), pt["p"].to_numpy(), "rule")

    def align(path, frame):
        sar = pd.read_parquet(path)
        m = frame[["airport", "date"]].merge(sar[["airport", "date", "p"]], on=["airport", "date"], how="left")
        assert m["p"].notna().all()
        return m["p"].to_numpy()

    preds[("sarima", 0)] = (align(RESULTS_DIR / "sarima_validation_predictions.parquet", va),
                            align(RESULTS_DIR / "sarima_test_predictions.parquet", te), "rule")

    MARKER.write_text(f"evaluated {datetime.now():%Y-%m-%d %H:%M}\n")

    y_va, a_va = va["label"].to_numpy(), va["airport"].to_numpy()
    y_te, a_te = te["label"].to_numpy(), te["airport"].to_numpy()
    year = te["date"].dt.year.to_numpy()
    rows, row_preds = [], {}
    for name, flags in (("never flag", np.zeros_like(y_te)), ("always flag", np.ones_like(y_te))):
        rows.append({"model": name, "seed": 0, "cost_rule": decision_costs(y_te, flags, a_te, faoc)["cost_per_airport_day"]})
    for (model, seed), (p_va, p_te, rule) in preds.items():
        va_costs = [decision_costs(y_va, p_va >= t, a_va, faoc)["cost_per_airport_day"] for t in GRID]
        tuned = GRID[int(np.argmin(va_costs))]
        primary_te = 0.5 if rule == "0.5" else thr_te
        r = {"model": model, "seed": seed, **evaluate(y_te, p_te, a_te),
             "cost_rule": decision_costs(y_te, p_te >= primary_te, a_te, faoc)["cost_per_airport_day"],
             "val_tuned_threshold": float(tuned),
             "cost_val_tuned": decision_costs(y_te, p_te >= tuned, a_te, faoc)["cost_per_airport_day"]}
        for yr in (2024, 2025):
            m = year == yr
            r[f"within_auc_{yr}"] = evaluate(y_te[m], p_te[m], a_te[m])["within_airport_auc"]
            r[f"mean_pred_{yr}"], r[f"prevalence_{yr}"] = float(p_te[m].mean()), float(y_te[m].mean())
            pt = primary_te if np.isscalar(primary_te) else primary_te[m]
            r[f"cost_rule_{yr}"] = decision_costs(y_te[m], p_te[m] >= pt, a_te[m], faoc)["cost_per_airport_day"]
        rows.append(r)
        row_preds.setdefault(model, []).append(p_te)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "test_results_per_seed.csv", index=False)
    out = te[["airport", "date", "label"]].copy()
    for model, ps in row_preds.items():
        out[model] = np.mean(ps, axis=0)
    out.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)

    cols = ["roc_auc", "within_airport_auc", "log_loss", "brier", "ece", "mean_pred", "cost_rule", "cost_val_tuned",
            "within_auc_2024", "within_auc_2025", "cost_rule_2024", "cost_rule_2025", "mean_pred_2025", "prevalence_2025"]
    summary = res.groupby("model", sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / "test_results_summary.csv")
    pd.set_option("display.width", 260)
    print(f"TEST: {len(te):,} airport-days, prevalence {y_te.mean():.3f} "
          f"(2024 {y_te[year == 2024].mean():.3f}, 2025 {y_te[year == 2025].mean():.3f})")
    print(res.groupby("model", sort=False)[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
