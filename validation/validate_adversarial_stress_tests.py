"""
Runs the Step 11 stress battery against every architecture built so far:
MoERegimeRouter (Step 2), PlanningDisruptNet (Step 7), the two neural
baselines (Step 6), and the Regime Transformer (Step 8) - plus the
NaN-propagation, disconnected-graph, and FGSM-scaffolding tests specific to
one representative model each.

Run from the repo root:
    python -m validation.validate_adversarial_stress_tests
"""

import torch

from models.planning.baselines import Seq2SeqBaseline, TFTBaseline
from models.planning.moe_router import MoERegimeRouter
from models.planning.planning_disruptnet import PlanningDisruptNet
from validation.adversarial_stress_tests import (
    fgsm_perturb,
    probe_model_robustness,
    test_disconnected_graph_node,
    test_nan_propagates_visibly,
)

torch.manual_seed(0)

NUM_FEATURES = 6
BATCH_SIZE = 5
SEQ_LEN = 10


def print_results(name: str, results: dict):
    print(f"--- {name} ---")
    for scenario, outcome in results.items():
        if outcome.get("crashed"):
            print(f"  {scenario:22s} CRASHED: {outcome['error']}")
        else:
            status = "OK" if outcome["finite"] and outcome["valid_range"] else "PROBLEM"
            print(f"  {scenario:22s} finite={outcome['finite']}  valid_range={outcome['valid_range']}  [{status}]")


def main():
    print("=== Robustness battery: MoERegimeRouter (2D input) ===")
    router = MoERegimeRouter(input_dim=NUM_FEATURES)
    x_2d = torch.randn(BATCH_SIZE, NUM_FEATURES)
    print_results("MoERegimeRouter", probe_model_robustness("moe_router", lambda x: router(x), x_2d))

    print()
    print("=== Robustness battery: PlanningDisruptNet (2D input) ===")
    pdn = PlanningDisruptNet(input_dim=NUM_FEATURES)
    print_results("PlanningDisruptNet", probe_model_robustness("planning_disruptnet", lambda x: pdn(x), x_2d))

    print()
    print("=== Robustness battery: Seq2SeqBaseline (3D input) ===")
    seq2seq = Seq2SeqBaseline(num_features=NUM_FEATURES)
    x_3d = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    print_results("Seq2SeqBaseline", probe_model_robustness("seq2seq", lambda x: seq2seq(x), x_3d))

    print()
    print("=== Robustness battery: TFTBaseline (3D input) ===")
    tft = TFTBaseline(num_features=NUM_FEATURES)
    print_results("TFTBaseline", probe_model_robustness("tft", lambda x: tft(x), x_3d))

    print()
    print("=== NaN propagation: does a NaN input become a visible NaN output, or silently 'fine'? ===")
    nan_result = test_nan_propagates_visibly(pdn, x_2d)
    print(f"  output contains NaN: {nan_result['output_contains_nan']} (expect True - visible, not silent)")
    print(f"  NaN isolated to the affected example only: {nan_result['nan_isolated_to_affected_example']}")

    print()
    print("=== Disconnected graph node: every airport with zero rotation-chain edges ===")
    disconnected_result = test_disconnected_graph_node()
    print(f"  output finite (self-loop safeguard prevents all -inf softmax row): "
          f"{disconnected_result['output_finite']}")
    print(f"  output in valid probability range: {disconnected_result['output_valid_range']}")

    print()
    print("=== FGSM-style perturbation scaffolding (on UNTRAINED weights - proves the mechanism, not robustness) ===")
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    airports = ["ORD", "ADK", "LGB", "ATL", "ZZZ"]
    fgsm_result = fgsm_perturb(pdn, x_2d, labels, airports, epsilon=0.1)
    print(f"  predictions before: {[round(p, 4) for p in fgsm_result['pred_before']]}")
    print(f"  predictions after:  {[round(p, 4) for p in fgsm_result['pred_after']]}")
    print(f"  max absolute change: {fgsm_result['max_abs_change']:.4f}")
    print(f"  output still a valid, finite probability after perturbation: {fgsm_result['output_still_valid']}")


if __name__ == "__main__":
    main()
