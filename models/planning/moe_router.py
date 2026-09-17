"""
Phase 3 / Person C shared utility - Step 2 of the Planning-tier build
sequence (see TEAM_PLAN.md), paired with FAOC-Loss (losses/faoc_loss.py).
Architecture stub: written and mechanically validated, not trained - no
model in this project has been trained yet (see PLAN.md, Phase 2).

What this is: instead of one network handling every situation the same
way, a Mixture-of-Experts (MoE) splits the work across several small
"expert" sub-networks - one per disruption regime - and a "gate" that looks
at the input and decides how much to trust each expert for that specific
case. The regimes are the three disruption types this project's own scope
already names (README.md, PLAN.md): Ground Delay Program (GDP), Ground
Stop, and Miles-in-Trail (MIT) - plus a fourth "normal" regime for when no
disruption is occurring.

This is a *soft* mixture: every expert always runs, and the gate produces a
weight per regime (summing to 1, via softmax) that blends their outputs.
That's the simplest form of MoE and the right default for a first stub -
sparse/top-k routing (only running the top 1-2 experts) adds load-balancing
complexity that isn't worth taking on before a single baseline model exists
to compare against.

Because the network is freshly initialized and untrained, the gate's
weights do not yet correspond to anything real - it has not learned to
recognize a GDP-like input from a Ground-Stop-like one. Validation here
only checks that the mechanism is wired correctly (shapes, softmax
normalization, output range, and that it composes cleanly with FAOC-Loss),
not that it has learned anything - see moe_router_validation.md.

Usage:
    from models.planning.moe_router import MoERegimeRouter

    router = MoERegimeRouter(input_dim=feature_count)
    pred_probs = router(features)                       # shape (N,), in (0, 1)
    pred_probs, gate_weights = router(features, return_gate_weights=True)
"""

import torch
import torch.nn as nn

REGIME_NAMES = ["normal", "gdp", "ground_stop", "mit"]


class _Expert(nn.Module):
    """One regime's sub-network: a small MLP producing a single disruption
    logit. Kept deliberately simple (one hidden layer) - this stub's job is
    to prove the routing mechanism works, not to be the final architecture.
    The sector-graph-attention Regime Transformer (Step 8) is the intended
    eventual replacement for what sits inside each expert."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (N,)


class MoERegimeRouter(nn.Module):
    """Regime-aware gating layer over one expert sub-network per regime.

    forward(x) returns a single blended disruption probability per sample,
    ready to hand to FAOCLoss alongside labels and airport codes - the two
    shared utilities are designed to compose directly, not just coexist.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.regime_names = REGIME_NAMES
        num_regimes = len(REGIME_NAMES)
        self.gate = nn.Linear(input_dim, num_regimes)
        self.experts = nn.ModuleList(
            [_Expert(input_dim, hidden_dim) for _ in range(num_regimes)]
        )

    def forward(self, x: torch.Tensor, return_gate_weights: bool = False):
        """
        x: feature tensor, shape (N, input_dim).
        Returns pred_probs, shape (N,), in (0, 1) - and, if requested,
        gate_weights, shape (N, num_regimes), each row summing to 1.
        """
        gate_weights = torch.softmax(self.gate(x), dim=-1)  # (N, num_regimes)
        expert_logits = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (N, num_regimes)
        combined_logit = (gate_weights * expert_logits).sum(dim=-1)  # (N,)
        pred_probs = torch.sigmoid(combined_logit)

        if return_gate_weights:
            return pred_probs, gate_weights
        return pred_probs
