"""
Person C (Planning tier) - Step 9 of the build sequence: six-metric KPI
suite. "Operational metrics beyond accuracy - false-alarm cost, lead-time
credit, regime-transition F1, and more." Shared utility (see TEAM_PLAN.md)
- all three tiers evaluate through this, once trained models exist to
evaluate. Only 3 of the 6 metrics were named in the build doc; the other 3
(cost-weighted recall, probability calibration error, forecast stability)
are this module's own addition, chosen to fill real gaps the named 3 leave
open - see each function's docstring for why it earns its place.

Deliberately NOT one unified interface: these six metrics need genuinely
different inputs (some need dollar figures, some need time-ordered
sequences, some need repeated predictions for the same target date). Forcing
a single call signature across all six would hide that difference rather
than represent it honestly - the same reasoning that kept SARIMA separate
from the two neural nets in models/planning/baselines.py.

Validated with small, hand-constructed examples where the correct answer
is known in advance (see validate_kpi_suite.py) - not real model
predictions, since no model has been trained yet (every prior step's
caveat applies here too).
"""

import numpy as np
import torch

from losses.faoc_loss import FAOCLoss, NATIONAL_COMPOSITE_MULTIPLIER


def false_alarm_cost(pred_labels: torch.Tensor, true_labels: torch.Tensor, faoc_loss: FAOCLoss) -> float:
    """Total dollar cost attributable to false positives, in this batch.
    Reuses `faoc_loss.fp_weight` directly rather than redefining the
    dollar figure here - if FAOC-Loss's cost assumptions get recalibrated
    (cost_figures.md §4), this metric follows automatically."""
    false_positives = (pred_labels == 1) & (true_labels == 0)
    return float(false_positives.sum()) * faoc_loss.fp_weight


def cost_weighted_recall(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    airports: list,
    faoc_loss: FAOCLoss,
) -> float:
    """What fraction of the TOTAL dollar-weighted disruption risk in this
    batch did the model actually catch? Complements false_alarm_cost (the
    FP side) by putting the FN side in the same dollar terms, rather than
    just counting missed events - catching 1 disruption at a high-cascade
    airport (see delay_propagation_multipliers.csv) matters more than
    catching 1 at a quiet one, and raw recall can't see that difference.
    """
    is_positive = true_labels == 1
    if not bool(is_positive.any()):
        raise ValueError("no true-positive examples in this batch - recall is undefined")

    dm = torch.tensor(
        [faoc_loss.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in airports],
        dtype=torch.float32,
    )
    at_risk_cost = (faoc_loss.fn_weight * dm)[is_positive]
    caught = (pred_labels == 1)[is_positive]
    return float(at_risk_cost[caught].sum() / at_risk_cost.sum())


def lead_time_credit(lead_time_days: torch.Tensor, max_horizon_days: int = 30) -> float:
    """Average credit for true positives, proportional to how much advance
    warning each one gave (lead_time_days: days between the as-of
    prediction date and the actual disruption date, one per true
    positive - computed by the caller, since it needs the real event date,
    not just the label). A 30-day-out catch scores 1.0; a 1-day-out catch
    scores ~0.033. Rewards early warning specifically, since a correct but
    last-minute alert is operationally far less useful than an early one -
    the KPI-suite equivalent of what avg_fn_hours tries to capture in
    FAOC-Loss, but for evaluation rather than training.
    """
    if len(lead_time_days) == 0:
        raise ValueError("no true positives to score - lead-time credit is undefined")
    return float((lead_time_days.clamp(0, max_horizon_days) / max_horizon_days).mean())


def regime_transition_f1(true_labels: torch.Tensor, pred_labels: torch.Tensor) -> float:
    """F1 computed on regime TRANSITIONS (label changes from one day to
    the next), not on every day's label directly. Time-ordered 1D
    sequences, one airport at a time. A naive daily F1 would be dominated
    by "nothing changed" days, since most days aren't the start of a new
    disruption regime - this isolates the specific, rarer, operationally
    critical event of a regime actually changing.
    """
    true_labels = true_labels.to(dtype=torch.float32)
    pred_labels = pred_labels.to(dtype=torch.float32)
    true_transition = torch.cat([torch.tensor([0.0]), (true_labels.diff() != 0).float()])
    pred_transition = torch.cat([torch.tensor([0.0]), (pred_labels.diff() != 0).float()])

    tp = float(((true_transition == 1) & (pred_transition == 1)).sum())
    fp = float(((true_transition == 0) & (pred_transition == 1)).sum())
    fn = float(((true_transition == 1) & (pred_transition == 0)).sum())

    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        raise ValueError("no transitions at all in this sequence - F1 is undefined")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calibration_error(pred_probs: torch.Tensor, true_labels: torch.Tensor, num_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): buckets predictions by predicted
    probability, and measures the average gap between "what the model
    claimed" and "what actually happened" in each bucket. A model that
    says "70% chance" should be right about 70% of the time across all the
    cases it said that - this matters specifically because this project is
    explicitly PROBABILISTIC forecasting (README.md), not just binary
    classification, so the probability itself needs to be trustworthy, not
    just the yes/no call at some threshold.
    """
    pred_probs = pred_probs.numpy() if isinstance(pred_probs, torch.Tensor) else np.asarray(pred_probs)
    true_labels = true_labels.numpy() if isinstance(true_labels, torch.Tensor) else np.asarray(true_labels)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.clip(np.digitize(pred_probs, bin_edges[1:-1]), 0, num_bins - 1)

    total_error = 0.0
    n = len(pred_probs)
    for b in range(num_bins):
        mask = bin_indices == b
        if not mask.any():
            continue
        bin_confidence = pred_probs[mask].mean()
        bin_accuracy = true_labels[mask].mean()
        total_error += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(total_error)


def forecast_stability(predictions_over_time: torch.Tensor) -> float:
    """predictions_over_time: 1D, time-ordered predicted probabilities for
    the SAME target date, made at successively later as-of dates (e.g. the
    30-day-out prediction, then 29-day-out, ... down to 1-day-out, all
    forecasting the same eventual day). Returns the mean absolute
    day-over-day change - lower means the model gives planners stable
    guidance to act on; a prediction that swings from 0.8 to 0.1 and back
    as new data trickles in is operationally costly even if it's eventually
    correct, since ATCSCC planning depends on guidance not flip-flopping
    day to day.
    """
    if len(predictions_over_time) < 2:
        raise ValueError("need at least 2 predictions over time to measure stability")
    return float(predictions_over_time.diff().abs().mean())
