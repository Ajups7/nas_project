"""
Person C (Planning tier) - Step 11 of the build sequence: adversarial
stress testing. "Perturbation and edge-case scenarios probing failure
modes before deployment." Shared utility (TEAM_PLAN.md).

Honest scope: no model in this project has been trained yet (every prior
step's caveat applies here too), so "adversarial robustness" in the usual
sense - does a trained model's prediction flip under a small malicious
perturbation - isn't measurable yet. What IS measurable, and worth testing
regardless of training state, is whether the architectures built in Steps
2, 6, 7, 8 fail SAFELY under bad input: do they stay numerically finite
under extreme-magnitude input, does a single-example batch work (a classic
training-vs-deployment gotcha), does a malformed graph (a disconnected
node) avoid producing NaN, and - the test tied most directly to a real bug
already found in this project - does a NaN value in the input propagate
VISIBLY (as NaN in the output, easy to detect) rather than silently
becoming a plausible-looking wrong number.

That last point connects directly to features/planning.py's real finding:
load_weather() returns "M" (missing) as a string, and a careless caller
who forgot to coerce it to NaN could feed genuinely corrupted data into a
model. test_nan_propagates_visibly() below confirms that if a NaN DOES
reach a model (despite planning.py's fix), it fails loudly rather than
quietly, which is the safer of the two bad outcomes.

fgsm_perturb() is deliberately-named after the classic Fast Gradient Sign
Method adversarial attack, but built and validated here on UNTRAINED
weights - same "written, not run for real" scope as every architecture
stub in this project. It exists so the actual attack is ready to run the
moment a trained model exists, not so it can report anything about
today's random weights.
"""

import torch

from models.planning.moe_router import MoERegimeRouter
from models.planning.planning_disruptnet import PlanningDisruptNet
from models.planning.regime_transformer import SectorGraphAttentionRegimeTransformer

STRESS_SCENARIOS = {
    "large_magnitude_1e6": lambda x: x * 1e6,
    "tiny_magnitude_1e-6": lambda x: x * 1e-6,
    "all_zero": lambda x: torch.zeros_like(x),
    "single_example": lambda x: x[:1],
}


def probe_model_robustness(name: str, forward_fn, base_input: torch.Tensor) -> dict:
    """Runs `forward_fn` (a closure wrapping one model's forward pass, so
    different models' differing input shapes/extra args can each supply
    their own closure) against every scenario in STRESS_SCENARIOS. Reports
    whether it crashed, and if not, whether the output stayed finite and
    in a valid probability range."""
    results = {}
    for scenario_name, transform in STRESS_SCENARIOS.items():
        perturbed = transform(base_input)
        try:
            output = forward_fn(perturbed)
            results[scenario_name] = {
                "crashed": False,
                "finite": bool(torch.isfinite(output).all()),
                "valid_range": bool(((output >= 0) & (output <= 1)).all()),
            }
        except Exception as e:
            results[scenario_name] = {"crashed": True, "error": f"{type(e).__name__}: {e}"}
    return results


def test_nan_propagates_visibly(model: PlanningDisruptNet, x: torch.Tensor) -> dict:
    """If a NaN slips into the input (e.g. an unimputed missing weather
    value - see features/planning.py's real "M" sentinel finding), does
    the model's output visibly contain NaN (safe - easy to detect
    downstream) or does it silently produce a plausible-looking finite
    number (dangerous - a bad prediction nobody would think to question)?
    """
    x_with_nan = x.clone()
    x_with_nan[0, 0] = float("nan")
    output = model(x_with_nan)
    return {
        "output_contains_nan": bool(torch.isnan(output).any()),
        "nan_isolated_to_affected_example": bool(torch.isnan(output[0]) and not torch.isnan(output[1:]).any()),
    }


def test_disconnected_graph_node(num_features: int = 6, num_nodes: int = 4) -> dict:
    """A node with ZERO rotation-chain edges (an airport with no aircraft
    rotation connections in the graph) - regime_transformer.py's
    build_attention_mask() adds a self-loop specifically so this can't
    produce an all -inf softmax row (which would yield NaN). This is a
    direct test of that safeguard, in the exact scenario it exists for."""
    model = SectorGraphAttentionRegimeTransformer(num_features=num_features)
    x = torch.randn(1, num_nodes, 8, num_features)
    adjacency = torch.zeros(num_nodes, num_nodes)  # every node fully disconnected
    output = model(x, adjacency)
    return {
        "output_finite": bool(torch.isfinite(output).all()),
        "output_valid_range": bool(((output >= 0) & (output <= 1)).all()),
    }


def fgsm_perturb(
    model: PlanningDisruptNet,
    x: torch.Tensor,
    labels: torch.Tensor,
    airports: list,
    epsilon: float,
    loss_type: str = "faoc",
) -> dict:
    """Fast-Gradient-Sign-style perturbation: nudges `x` by `epsilon` in
    the direction that INCREASES the loss most, per input dimension, and
    reports how much the model's predictions moved. Built and validated
    here on untrained weights - proves the attack mechanism itself works,
    not that today's random weights are "vulnerable" (that claim is
    meaningless before training)."""
    x = x.clone().requires_grad_(True)
    loss = model.compute_loss(x, labels, airports, loss_type=loss_type)
    loss.backward()

    x_perturbed = (x + epsilon * x.grad.sign()).detach()
    with torch.no_grad():
        pred_before = model(x)
        pred_after = model(x_perturbed)

    return {
        "pred_before": pred_before.tolist(),
        "pred_after": pred_after.tolist(),
        "max_abs_change": float((pred_after - pred_before).abs().max()),
        "output_still_valid": bool(((pred_after >= 0) & (pred_after <= 1)).all() and torch.isfinite(pred_after).all()),
    }
