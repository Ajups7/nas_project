"""
Validates shadow_mode.py's mechanics with a small synthetic example, then
runs the full shadow_mode_replay.py capstone integration against REAL
data - the only validation in this project that exercises features,
model, shadow log, real labels, and the KPI suite all in one run.

Run from the repo root:
    python -m validation.validate_shadow_mode
"""

import pandas as pd
import torch

from losses.faoc_loss import FAOCLoss
from models.planning.compare_losses import predict_from_checkpoint
from models.planning.data import load_planning_splits
from models.planning.train_disruptnet import CHECKPOINT_DIR
from validation.kpi_suite import calibration_error, cost_weighted_recall, false_alarm_cost
from validation.shadow_mode import SAFE_PUBLIC_METHODS, ShadowModeLog
from validation.shadow_mode_replay import run_shadow_mode_replay

AIRPORT = "ATL"
DATES = pd.date_range("2023-06-20", "2023-06-24")
CHECKPOINT = CHECKPOINT_DIR / "final_standard_seed0.pt"


def validate_api_surface():
    print("=== The 'no acting on predictions' contract: introspect the public API ===")
    public_methods = {
        name for name in dir(ShadowModeLog)
        if not name.startswith("_") and callable(getattr(ShadowModeLog, name))
    }
    print(f"public methods found: {sorted(public_methods)}")
    print(f"matches the declared safe surface exactly: {public_methods == SAFE_PUBLIC_METHODS}")
    print("(no method here could feed a prediction back into a real decision -")
    print(" this is checkable, not just claimed in a docstring)")
    print()


def validate_basic_mechanics():
    print("=== Basic mechanics: record -> resolve -> read ===")
    log = ShadowModeLog()
    log.record_prediction("2024-01-01", "ORD", 0.7)
    log.record_prediction("2024-01-02", "ORD", 0.3)
    print(f"unresolved count after 2 predictions, 0 resolutions: {log.unresolved_count()} (expect 2)")

    log.resolve_outcome("2024-01-01", "ORD", True)
    print(f"unresolved count after 1 resolution: {log.unresolved_count()} (expect 1)")
    print(f"resolved pairs: {log.resolved_pairs()} (expect [(0.7, True)])")

    print()
    print("=== Error handling: resolving a prediction that was never recorded ===")
    try:
        log.resolve_outcome("2024-01-05", "ORD", True)
        print("no error raised (unexpected)")
    except ValueError as e:
        print(f"raised ValueError as expected: {e}")
    print()


def validate_real_replay():
    print("=== CAPSTONE: replay of the frozen main model against real data ===")
    val = load_planning_splits(exclude_covid=True)["validation"]
    frame = val[(val["airport"] == AIRPORT) & val["date"].isin(DATES)]
    print(f"airport={AIRPORT}, dates={[d.date() for d in DATES]}")
    log, pending = run_shadow_mode_replay(frame, lambda rows: predict_from_checkpoint(CHECKPOINT, rows))

    print(f"dates replayed: {len(frame)}; unresolved at end of each simulated day: {pending}")
    print("(all pending: no 30-day window elapses inside a 5-day replay)")
    print(f"unresolved after the clock runs out: {log.unresolved_count()} (expect 0)")

    resolved = log.resolved_pairs()
    print()
    print("resolved (prediction, actual outcome) pairs:")
    for pred, outcome in resolved:
        print(f"  predicted={pred:.4f}  actual={outcome}")

    print()
    print("=== Scoring the shadow log with Step 9's KPI suite ===")
    pred_probs = torch.tensor([p for p, _ in resolved])
    true_labels = torch.tensor([float(o) for _, o in resolved])
    pred_labels = (pred_probs > 0.5).int()

    ece = calibration_error(pred_probs, true_labels, num_bins=5)
    print(f"calibration error: {ece:.4f}")

    criterion = FAOCLoss()
    fa_cost = false_alarm_cost(pred_labels, true_labels.int(), criterion)
    print(f"false-alarm cost: ${fa_cost:,.2f}")

    try:
        airports = [AIRPORT] * len(resolved)
        cwr = cost_weighted_recall(pred_labels, true_labels.int(), airports, criterion)
        print(f"cost-weighted recall: {cwr:.4f}")
    except ValueError as e:
        print(f"cost-weighted recall: not applicable ({e})")


if __name__ == "__main__":
    validate_api_surface()
    validate_basic_mechanics()
    validate_real_replay()
