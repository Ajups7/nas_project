"""
Validates kpi_suite.py against small, hand-constructed examples where the
correct answer is worked out BEFORE running the code, not just "does it
run without error." Same philosophy as validate_faoc_loss.py's confusion-
matrix corners: known inputs, known expected outputs.

Run from the repo root:
    python -m validation.validate_kpi_suite
"""

import torch

from losses.faoc_loss import FAOCLoss
from validation.kpi_suite import (
    calibration_error,
    cost_weighted_recall,
    false_alarm_cost,
    forecast_stability,
    lead_time_credit,
    regime_transition_f1,
)


def check(name, actual, expected, tol=1e-6):
    ok = abs(actual - expected) < tol
    print(f"{name}: got {actual:.6f}, expected {expected:.6f}  [{'OK' if ok else 'MISMATCH'}]")


def validate_false_alarm_cost():
    print("=== false_alarm_cost ===")
    criterion = FAOCLoss()
    # 5 samples: exactly 2 false positives (pred=1, true=0), rest correct/other.
    pred = torch.tensor([1, 1, 0, 1, 0])
    true = torch.tensor([0, 1, 0, 0, 1])
    # false positives: index 0 (pred1,true0), index 3 (pred1,true0) -> 2 FPs
    expected = 2 * criterion.fp_weight
    actual = false_alarm_cost(pred, true, criterion)
    check("2 false positives", actual, expected)
    print()


def validate_cost_weighted_recall():
    print("=== cost_weighted_recall ===")
    criterion = FAOCLoss()
    # 3 true positives at 3 different airports; catch 2 of them.
    pred = torch.tensor([1, 0, 1, 0])  # 4th is a true negative, irrelevant
    true = torch.tensor([1, 1, 1, 0])
    airports = ["ORD", "ADK", "LGB", "ATL"]
    dm = {"ORD": criterion.multipliers["ORD"], "ADK": criterion.multipliers["ADK"], "LGB": criterion.multipliers["LGB"]}
    at_risk = criterion.fn_weight * (dm["ORD"] + dm["ADK"] + dm["LGB"])
    caught = criterion.fn_weight * (dm["ORD"] + dm["LGB"])  # ORD and LGB caught, ADK missed
    expected = caught / at_risk
    actual = cost_weighted_recall(pred, true, airports, criterion)
    check("catch 2 of 3, different airports", actual, expected)
    print()


def validate_lead_time_credit():
    print("=== lead_time_credit ===")
    # 3 true positives with 30, 15, 0 days of lead time -> average of (1.0, 0.5, 0.0) = 0.5
    lead_times = torch.tensor([30.0, 15.0, 0.0])
    expected = (1.0 + 0.5 + 0.0) / 3
    actual = lead_time_credit(lead_times, max_horizon_days=30)
    check("lead times [30, 15, 0] / 30-day horizon", actual, expected)
    print()


def validate_regime_transition_f1():
    print("=== regime_transition_f1 ===")
    # true:  0 0 1 1 0 0 1  -> transitions at index 2 (0->1), 4 (1->0), 6 (0->1) = 3 transitions
    # pred:  0 0 1 1 1 0 1  -> transitions at index 2 (0->1), 5 (1->0), 6 (0->1) = 3 transitions
    # true transition indices: {2, 4, 6}; pred transition indices: {2, 5, 6}
    # TP (both):    {2, 6} -> 2
    # FP (pred only): {5} -> 1
    # FN (true only): {4} -> 1
    # precision = 2/3, recall = 2/3, F1 = 2/3
    true = torch.tensor([0, 0, 1, 1, 0, 0, 1])
    pred = torch.tensor([0, 0, 1, 1, 1, 0, 1])
    expected = 2 / 3
    actual = regime_transition_f1(true, pred)
    check("hand-traced transition F1", actual, expected)
    print()


def validate_calibration_error():
    print("=== calibration_error ===")
    # Perfectly calibrated: predicted 0.2 for 5 examples, 1 of them (20%) is actually positive.
    pred_probs = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    true_labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # 1/5 = 0.2, matches exactly
    actual = calibration_error(pred_probs, true_labels, num_bins=10)
    check("perfectly calibrated bucket (0.2 pred, 0.2 actual)", actual, 0.0)

    # Badly miscalibrated: predicted 0.9 for 5 examples, none actually positive.
    pred_probs_bad = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])
    true_labels_bad = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    actual_bad = calibration_error(pred_probs_bad, true_labels_bad, num_bins=10)
    check("badly miscalibrated (0.9 pred, 0.0 actual)", actual_bad, 0.9)
    print()


def validate_forecast_stability():
    print("=== forecast_stability ===")
    # Predictions swing 0.1 -> 0.9 -> 0.2 -> 0.8 : abs diffs = 0.8, 0.7, 0.6 -> mean = 0.7
    swinging = torch.tensor([0.1, 0.9, 0.2, 0.8])
    expected_swinging = (0.8 + 0.7 + 0.6) / 3
    actual_swinging = forecast_stability(swinging)
    check("swinging predictions", actual_swinging, expected_swinging)

    # Stable predictions: barely move -> stability metric should be near 0
    stable = torch.tensor([0.70, 0.71, 0.69, 0.70])
    actual_stable = forecast_stability(stable)
    print(f"stable predictions: {actual_stable:.4f} (expect small, near 0)")
    print(f"swinging is correctly rated less stable than stable: {actual_swinging > actual_stable}")
    print()


def validate_error_handling():
    print("=== Error handling: undefined cases raise, not silently return garbage ===")
    try:
        cost_weighted_recall(torch.tensor([0]), torch.tensor([0]), ["ORD"], FAOCLoss())
        print("cost_weighted_recall: no error (unexpected)")
    except ValueError as e:
        print(f"cost_weighted_recall with no true positives raises: {e}")

    try:
        lead_time_credit(torch.tensor([]))
        print("lead_time_credit: no error (unexpected)")
    except ValueError as e:
        print(f"lead_time_credit with no true positives raises: {e}")

    try:
        forecast_stability(torch.tensor([0.5]))
        print("forecast_stability: no error (unexpected)")
    except ValueError as e:
        print(f"forecast_stability with <2 predictions raises: {e}")


if __name__ == "__main__":
    validate_false_alarm_cost()
    validate_cost_weighted_recall()
    validate_lead_time_credit()
    validate_regime_transition_f1()
    validate_calibration_error()
    validate_forecast_stability()
    validate_error_handling()
