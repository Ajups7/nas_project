"""
Objective 4 for the Planning tier, on the FROZEN models (no retraining, no
tuning): the three KPIs not yet computed on trained models, adversarial
stress tests, and a shadow-mode replay. Uses the test window, which is
legitimate here because nothing below feeds back into any model choice.

KPIs (validation/kpi_suite.py), per model, at its fixed decision rule:
- lead-time credit: for each caught GDP window, days from the as-of date to
  the first GDP day in (date+1, date+30], / 30. Mostly a property of when
  GDPs fall in the window, so compared across models, not read absolutely.
- regime-transition F1: per airport, F1 on day-to-day changes of the
  label vs. of the decision; mean over airports with any transition.
- forecast stability: the kpi_suite metric is defined for repeated
  forecasts of ONE target date; the Planning label is a rolling window, so
  the closest analogue is used - per airport, mean |p(t) - p(t-1)| across
  consecutive as-of dates - plus the day-to-day decision flip rate.
  (False-alarm cost, cost-weighted recall and calibration are in
  results/test_results_*.csv.)

Stress tests on the main model (final_standard, 3 seeds):
- probe_model_robustness: extreme magnitudes, zeros, single example
- NaN propagation
- FGSM on the 12 real features only (airport identity masked), epsilon in
  standard-deviation units, measuring probability change and decision flips
- domain scenarios: recent-history feed outage (rates replaced by
  climatology, as imputation does), a persistent-GDP spike (recent rates = 1)

Shadow mode: chronological replay of the test window with the main model
(seed 0), checking the replayed log reproduces the direct evaluation.

Run from the repo root (after evaluate_test):
    python -m models.planning.operational_validation
"""

import numpy as np
import pandas as pd
import torch

from common.disruption import gdp_days
from features.planning import FEATURE_COLUMNS
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.compare_losses import predict_from_checkpoint
from models.planning.data import AIRPORT_COLUMNS, load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.planning_disruptnet import PlanningDisruptNet
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR
from validation.adversarial_stress_tests import fgsm_perturb, probe_model_robustness, test_nan_propagates_visibly
from validation.kpi_suite import lead_time_credit, regime_transition_f1
from validation.shadow_mode_replay import run_shadow_mode_replay

SEEDS = (0, 1, 2)
HORIZON = 30
EPSILONS = (0.05, 0.1, 0.25, 0.5)


def first_gdp_lead(test: pd.DataFrame) -> np.ndarray:
    """Days from each as-of date to the first GDP day in its window (NaN if none)."""
    g = gdp_days()
    out = np.full(len(test), np.nan)
    for airport, idx in test.groupby("airport").indices.items():
        s = g[airport].fillna(0).to_numpy()
        nxt = np.full(len(s) + 1, np.inf)
        for i in range(len(s) - 1, -1, -1):
            nxt[i] = i if s[i] == 1 else nxt[i + 1]
        pos = g.index.get_indexer(test["date"].iloc[idx])
        lead = nxt[pos + 1] - pos
        out[idx] = np.where(lead <= HORIZON, lead, np.nan)
    return out


def kpis(test: pd.DataFrame, p: np.ndarray, flags: np.ndarray, lead: np.ndarray) -> dict:
    y = test["label"].to_numpy()
    caught = (flags == 1) & (y == 1) & ~np.isnan(lead)
    f1s, stab, flips = [], [], []
    dates = test["date"].to_numpy()
    for _, idx in test.groupby("airport").indices.items():
        order = idx[np.argsort(dates[idx], kind="stable")]
        ya, fa, pa = y[order], flags[order], p[order]
        try:
            f1s.append(regime_transition_f1(torch.tensor(ya), torch.tensor(fa)))
        except ValueError:
            pass
        if len(pa) > 1:
            stab.append(float(np.abs(np.diff(pa)).mean()))
            flips.append(float((np.diff(fa) != 0).mean()))
    return {
        "lead_time_credit": lead_time_credit(torch.tensor(lead[caught])),
        "mean_lead_days_caught": float(lead[caught].mean()),
        "regime_transition_f1": float(np.mean(f1s)),
        "airports_with_transitions": len(f1s),
        "forecast_stability_mean_abs_dp": float(np.mean(stab)),
        "decision_flip_rate": float(np.mean(flips)),
    }


def load_net(seed: int):
    ckpt = torch.load(CHECKPOINT_DIR / f"final_standard_seed{seed}.pt", weights_only=False)
    cfg = ckpt["config"]
    net = PlanningDisruptNet(len(cfg["feature_columns"]), cfg["hidden_dim"], pos_weight=cfg["pos_weight"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, cfg["feature_columns"], ckpt["scaler_mean"], ckpt["scaler_scale"]


def main():
    test = load_planning_splits(exclude_covid=True)["test"]
    preds = pd.read_parquet(RESULTS_DIR / "test_predictions.parquet")
    assert (preds[["airport", "date"]].to_numpy() == test[["airport", "date"]].to_numpy()).all()
    faoc = FAOCLoss()
    a = test["airport"].to_numpy()
    dm = np.array([faoc.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
    bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
    lead = first_gdp_lead(test)

    print("=== 1. Remaining KPIs on test (seed-averaged predictions, fixed decision rule) ===")
    rows = []
    for model in [c for c in preds.columns if c not in ("airport", "date", "label")]:
        p = preds[model].to_numpy()
        flags = (p >= (0.5 if model == "disruptnet faoc" else bayes)).astype(int)
        rows.append({"model": model, **kpis(test, p, flags, lead),
                     "cost": decision_costs(test["label"], flags, a, faoc)["cost_per_airport_day"]})
    for name, flags in (("always flag", np.ones(len(test), int)),):
        rows.append({"model": name, **kpis(test, np.ones(len(test)), flags, lead),
                     "cost": decision_costs(test["label"], flags, a, faoc)["cost_per_airport_day"]})
    k = pd.DataFrame(rows).set_index("model")
    k.to_csv(RESULTS_DIR / "test_operational_kpis.csv")
    pd.set_option("display.width", 220)
    print(k.round(4).to_string())

    print("\n=== 2. Stress tests on the main model (final_standard) ===")
    x_all = None
    stress = []
    for seed in SEEDS:
        net, cols, mean, scale = load_net(seed)
        x_all = torch.tensor((test[cols].to_numpy() - mean) / scale, dtype=torch.float32)
        base = net(x_all).detach().numpy()
        r = probe_model_robustness("final_standard", lambda x: net(x), x_all[:256])
        nan = test_nan_propagates_visibly(net, x_all[:8])
        mask = torch.tensor([1.0 if c in FEATURE_COLUMNS else 0.0 for c in cols])
        y = torch.tensor(test["label"].to_numpy(), dtype=torch.float32)
        row = {"seed": seed,
               "all_scenarios_finite_and_valid": all(v.get("finite") and v.get("valid_range") for v in r.values()),
               "nan_visible": nan["output_contains_nan"], "nan_isolated": nan["nan_isolated_to_affected_example"]}
        for eps in EPSILONS:
            f = fgsm_perturb(net, x_all, y, list(a), eps, loss_type="standard", feature_mask=mask)
            before, after = np.array(f["pred_before"]), np.array(f["pred_after"])
            row[f"fgsm{eps}_mean_dp"] = float(np.abs(after - before).mean())
            row[f"fgsm{eps}_flip_rate"] = float(((before >= bayes) != (after >= bayes)).mean())
            row[f"fgsm{eps}_cost"] = decision_costs(test["label"], after >= bayes, a, faoc)["cost_per_airport_day"]
        for scen, fn in (
            ("feed_outage", lambda df: df.assign(recent_gdp_rate_7d=df["climo_gdp_day_rate"],
                                                 recent_gdp_rate_30d=df["climo_gdp_day_rate"])),
            ("persistent_gdp_spike", lambda df: df.assign(recent_gdp_rate_7d=1.0, recent_gdp_rate_30d=1.0)),
        ):
            xs = torch.tensor((fn(test)[cols].to_numpy() - mean) / scale, dtype=torch.float32)
            with torch.no_grad():
                ps = net(xs).numpy()
            row[f"{scen}_mean_dp"] = float((ps - base).mean())
            row[f"{scen}_flip_rate"] = float(((base >= bayes) != (ps >= bayes)).mean())
            row[f"{scen}_cost"] = decision_costs(test["label"], ps >= bayes, a, faoc)["cost_per_airport_day"]
        row["clean_cost"] = decision_costs(test["label"], base >= bayes, a, faoc)["cost_per_airport_day"]
        stress.append(row)
    s = pd.DataFrame(stress).set_index("seed")
    s.to_csv(RESULTS_DIR / "test_stress_tests.csv")
    print(s.T.round(4).to_string())

    print("\n=== 3. Shadow-mode replay of the test window (main model, seed 0) ===")
    path = CHECKPOINT_DIR / "final_standard_seed0.pt"
    log, pending = run_shadow_mode_replay(test, lambda rows: predict_from_checkpoint(path, rows))
    pairs = log.resolved_pairs()
    direct = predict_from_checkpoint(path, test)
    replay_p = np.array([p for p, _ in pairs])
    replay_y = np.array([o for _, o in pairs]).astype(int)
    order = test.sort_values(["date"], kind="stable").index.to_numpy()
    print(f"days replayed: {len(pending)}; predictions: {len(pairs):,}; "
          f"unresolved after clock runs out: {log.unresolved_count()}")
    print(f"live backlog (predictions awaiting their 30-day outcome): median {int(np.median(pending)):,}, "
          f"max {max(pending):,}")
    same = np.allclose(np.sort(replay_p), np.sort(direct), atol=1e-6) and replay_y.sum() == test["label"].sum()
    m = evaluate(replay_y, replay_p, [p.airport for p in log._predictions])
    print(f"replay reproduces direct evaluation: {same};  replay log loss {m['log_loss']:.4f}, "
          f"within-airport AUC {m['within_airport_auc']:.4f}")
    pd.Series({"days": len(pending), "predictions": len(pairs), "unresolved_end": log.unresolved_count(),
               "backlog_median": float(np.median(pending)), "backlog_max": max(pending),
               "reproduces_direct": same, "log_loss": m["log_loss"]}).to_csv(RESULTS_DIR / "test_shadow_mode.csv")


if __name__ == "__main__":
    main()
