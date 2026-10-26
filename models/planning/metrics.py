"""
One scoring function for every Planning model, so results are comparable.

Pooled metrics can be inflated by airport identity alone (EWR is almost
always 1, ABQ almost always 0), so within-airport AUC is reported too: the
mean ROC-AUC computed separately per airport, over airports whose labels
take both values in the scored split. It measures whether the model can
tell an airport's bad periods from its good ones.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER
from validation.kpi_suite import calibration_error


def evaluate(labels, probs, airports) -> dict:
    y = np.asarray(labels).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)
    a = np.asarray(airports)

    per_airport = []
    for airport in np.unique(a):
        m = a == airport
        if 0 < y[m].sum() < m.sum():
            per_airport.append(roc_auc_score(y[m], p[m]))

    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "within_airport_auc": float(np.mean(per_airport)),
        "airports_scored": len(per_airport),
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "ece": calibration_error(p, y.astype(float)),
        "mean_pred": float(p.mean()),
        "prevalence": float(y.mean()),
    }


def decision_costs(labels, flags, airports, faoc_loss) -> dict:
    """Dollar cost of yes/no decisions under FAOC-Loss's own cost model:
    a miss costs fn_weight x the airport's delay propagation multiplier, a
    false alarm costs fp_weight. Same figures FAOC-Loss trains on, so the
    comparison is on the objective FAOC claims to optimize."""
    import torch

    from validation.kpi_suite import cost_weighted_recall, false_alarm_cost

    y = np.asarray(labels).astype(int)
    d = np.asarray(flags).astype(int)
    a = list(airports)
    dm = np.array([faoc_loss.multipliers.get(x, NATIONAL_COMPOSITE_MULTIPLIER) for x in a])
    miss_cost = float((faoc_loss.fn_weight * dm)[(y == 1) & (d == 0)].sum())
    fa_cost = false_alarm_cost(torch.tensor(d), torch.tensor(y), faoc_loss)
    return {
        "cost_per_airport_day": (miss_cost + fa_cost) / len(y),
        "miss_cost_share": miss_cost / max(miss_cost + fa_cost, 1e-9),
        "flag_rate": float(d.mean()),
        "cost_weighted_recall": cost_weighted_recall(torch.tensor(d), torch.tensor(y), a, faoc_loss),
    }


def results_table(rows: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).T
