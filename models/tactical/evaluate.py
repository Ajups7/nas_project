"""
Person A (Tactical tier) - Step 7 of the build sequence: the evaluation
stub. Wires TacticalDisruptNet's predictions into the shared six-metric
KPI suite (validation/kpi_suite.py, Person C's Step 9) rather than
building tier-specific metrics from scratch - the whole point of that
module is cross-tier comparability.

Reuses assign_tactical_split() (features/tactical_split.py, Step 4) to
pick which rows to evaluate on - never evaluate on 'train' or 'embargo'
rows.

Four of the six KPI-suite metrics translate directly to Tactical's
(airport, date, local_hour) granularity and 0-2h horizon:
  - false_alarm_cost, cost_weighted_recall: reuse a plain FAOCLoss()
    instance purely for its dollar figures (fp_weight/fn_weight/
    multipliers) - these model the cost of the disruption EVENT once it
    happens, not the prediction horizon, so they're not
    Planning-specific despite living in a Planning-authored loss module.
    Tactical still TRAINS with weighted BCE (Step 6), not FAOC-Loss -
    this is evaluation-only reuse of its cost figures, same as every
    other tier per kpi_suite.py's own docstring.
  - regime_transition_f1, calibration_error: no unit/horizon assumptions
    at all - direct reuse.

The other two are NOT wired up here, on purpose, not by oversight:
  - lead_time_credit needs the exact onset hour of each true disruption
    (to measure how many hours of advance warning a catch gave) -
    build_tactical_labels() only returns a binary "did EDCT activity
    occur anywhere in the next horizon_hours" via .any(), not which
    specific hour it first appeared in. Computing this honestly would
    require changing what that function returns, not just this module -
    flagged for a follow-up, not faked here with a wrong proxy.
  - forecast_stability needs several predictions for the SAME target
    hour made at successively later as-of times. With a 2-hour horizon
    that's at most 2 as-of points per target hour (vs. Planning's 30) -
    computable in principle, but no code yet assembles predictions this
    way. Left out rather than wired up on a single-point, near-
    meaningless "trajectory."

Every caveat every prior step in this build carries applies here too: no
model has actually been trained yet, so this is validated with small
hand-constructed examples where the right answer is known in advance -
same convention as kpi_suite_validation.md, not real predictions.
"""

import torch

from losses.faoc_loss import FAOCLoss
from validation.kpi_suite import (
    calibration_error,
    cost_weighted_recall,
    false_alarm_cost,
    regime_transition_f1,
)

TACTICAL_HORIZON_HOURS = 2  # matches features/tactical_label.py's DEFAULT_HORIZON_HOURS


def evaluate_tactical_predictions(
    pred_probs: torch.Tensor,
    true_labels: torch.Tensor,
    airports: list,
    faoc_loss: FAOCLoss = None,
    threshold: float = 0.5,
) -> dict:
    """Computes the four KPI-suite metrics that generalize cleanly to
    Tactical's granularity/horizon (see module docstring for the two that
    don't). `airports` is one code per row, same length as pred_probs -
    matches build_tactical_features()/build_tactical_labels()'s own
    'airport' column when evaluating multiple airports at once.

    Caller's responsibility: only pass rows already filtered to a single
    split bucket (features/tactical_split.py's assign_tactical_split) -
    this function has no opinion about train/validation/test, it just
    scores whatever predictions/labels you hand it."""
    if faoc_loss is None:
        faoc_loss = FAOCLoss()

    pred_labels = (pred_probs >= threshold).long()
    true_labels = true_labels.long()

    return {
        "false_alarm_cost": false_alarm_cost(pred_labels, true_labels, faoc_loss),
        "cost_weighted_recall": cost_weighted_recall(pred_labels, true_labels, airports, faoc_loss),
        "regime_transition_f1": regime_transition_f1(true_labels, pred_labels),
        "calibration_error": calibration_error(pred_probs, true_labels),
    }
