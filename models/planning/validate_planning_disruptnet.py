"""
Validates planning_disruptnet.py with synthetic inputs - same philosophy
as the rest of this project's validations (see moe_router_validation.md,
baselines_validation.md): proves the wiring is correct, not that anything
has learned to predict real disruptions.

The key check here isn't just "does it run" - it's "does loss_type='standard'
actually route to a DIFFERENT loss than loss_type='faoc', or did something
get aliased together by accident." The differential test below proves it
two ways: the loss VALUES differ, and more importantly, 'standard' is
provably blind to airport identity while 'faoc' is provably sensitive to
it - which is exactly the conceptual difference the module docstring claims.

Run from the repo root:
    python -m models.planning.validate_planning_disruptnet
"""

import torch

from models.planning.planning_disruptnet import PlanningDisruptNet

torch.manual_seed(0)

BATCH_SIZE = 5
NUM_FEATURES = 6


def main():
    model = PlanningDisruptNet(input_dim=NUM_FEATURES)
    x = torch.randn(BATCH_SIZE, NUM_FEATURES)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

    print("=== Forward pass (MoE routing exposed unchanged) ===")
    pred_probs, gate_weights = model(x, return_gate_weights=True)
    print(f"pred_probs shape: {tuple(pred_probs.shape)} (expect ({BATCH_SIZE},))")
    print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")
    print(f"gate_weights shape: {tuple(gate_weights.shape)} (expect ({BATCH_SIZE}, 4))")
    print(f"gate_weights sum to 1: {bool(torch.allclose(gate_weights.sum(dim=-1), torch.ones(BATCH_SIZE)))}")

    print()
    print("=== Both loss types run without error, and produce DIFFERENT values ===")
    airports_a = ["ORD", "ADK", "LGB", "ATL", "ZZZ"]
    standard_loss = model.compute_loss(x, labels, airports_a, loss_type="standard")
    faoc_loss = model.compute_loss(x, labels, airports_a, loss_type="faoc")
    print(f"standard loss: {standard_loss.item():.4f}")
    print(f"faoc loss:     {faoc_loss.item():,.2f}")
    print(f"different scales, as expected (faoc is dollar-denominated, standard isn't): "
          f"{faoc_loss.item() > 100 * standard_loss.item()}")

    print()
    print("=== The differential test: is 'standard' really blind to airport, unlike 'faoc'? ===")
    airports_b = ["LGB", "LGB", "LGB", "LGB", "LGB"]  # all high-multiplier (2.00), same x/labels
    standard_loss_b = model.compute_loss(x, labels, airports_b, loss_type="standard")
    faoc_loss_b = model.compute_loss(x, labels, airports_b, loss_type="faoc")

    standard_unchanged = torch.allclose(standard_loss, standard_loss_b)
    faoc_changed = not torch.allclose(faoc_loss, faoc_loss_b)
    print(f"same x/labels, airports changed to all-LGB (multiplier 2.00):")
    print(f"  standard loss: {standard_loss.item():.4f} -> {standard_loss_b.item():.4f}  "
          f"(unchanged, as expected: {standard_unchanged})")
    print(f"  faoc loss:     {faoc_loss.item():,.2f} -> {faoc_loss_b.item():,.2f}  "
          f"(changed, as expected: {faoc_changed})")
    print(f"CONFIRMS: 'standard' and 'faoc' are genuinely different code paths, "
          f"not aliased together: {standard_unchanged and faoc_changed}")

    print()
    print("=== Error handling: unknown loss_type ===")
    try:
        model.compute_loss(x, labels, airports_a, loss_type="nonsense")
        print("no error raised (unexpected)")
    except ValueError as e:
        print(f"raised ValueError as expected: {e}")


if __name__ == "__main__":
    main()
