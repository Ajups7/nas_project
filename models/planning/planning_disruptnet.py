"""
Person C (Planning tier) - Step 7 of the build sequence: the model stub.
"Planning DisruptNet tier wired for standard loss, FAOC-Loss, and MoE
routing." This doesn't invent a new architecture - it assembles pieces
already built and validated separately (Step 1's FAOCLoss, Step 2's
MoERegimeRouter) into one named, trainable unit with a pluggable loss.

Why "both losses," not just FAOC-Loss: if this model could only ever be
trained with FAOC-Loss, there'd be no clean way to tell "did FAOC-Loss
specifically help, or would any model have done about as well here?" By
keeping ONE fixed architecture (MoERegimeRouter) and making the loss a
runtime choice, a later experiment can hold the architecture constant and
vary only the loss - the controlled comparison an actual claim of
"FAOC-Loss improved things" needs.

Two conceptually different reasons a loss might be "weighted," both
present here for a reason:
- "standard" = weighted binary cross-entropy, weighted by POS_WEIGHT for
  class imbalance (disruptions are presumably rarer than non-disruption
  days) - a purely statistical concern, the same kind of loss Step 6's
  baselines and Person A/B's tiers train with (TEAM_PLAN.md).
- "faoc" = Step 1's FAOCLoss, weighted by real FAA/A4A dollar-per-flight-
  hour figures and per-airport delay propagation multipliers - an economic
  concern, unrelated to class frequency.
These are answering different questions, not doing the same thing twice:
"standard" asks "does this architecture even work" (fair comparison to the
Step 6 baselines); "faoc" asks "does dollar-cost-aware training change
what the model prioritizes." See planning_disruptnet_validation.md for a
concrete demonstration of that difference.

No training happens here - see baselines_validation.md's caveats, which
apply identically: this is architecture assembly, mechanically validated
with synthetic inputs, gated on the team-agreed temporal split
(TEAM_PLAN.md) before real training can start.
"""

import torch
import torch.nn as nn

from losses.faoc_loss import FAOCLoss
from models.planning.moe_router import MoERegimeRouter

DEFAULT_POS_WEIGHT = 1.0  # class-imbalance weight for the standard loss -
                           # a placeholder, same spirit as FAOCLoss's own
                           # avg_fn_hours/avg_fp_hours: meant to be
                           # recalibrated once real label prevalence is
                           # known (features/planning_label.py), not
                           # trusted as tuned.


def _weighted_bce(pred_probs: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Standard weighted binary cross-entropy - weighted by class
    imbalance (pos_weight), nothing to do with dollar cost or airport.
    Written manually rather than via nn.BCEWithLogitsLoss(pos_weight=...)
    because that class expects raw logits, and MoERegimeNet's forward()
    already applies sigmoid - see moe_router.py."""
    eps = 1e-7
    p = pred_probs.clamp(eps, 1 - eps)
    labels = labels.to(dtype=p.dtype)
    loss = -(pos_weight * labels * torch.log(p) + (1 - labels) * torch.log(1 - p))
    return loss.mean()


class PlanningDisruptNet(nn.Module):
    """The Planning tier's model. Architecture = MoERegimeRouter (Step 2).
    Loss is a runtime choice via compute_loss(..., loss_type=...): "standard"
    (weighted BCE) or "faoc" (FAOCLoss, Step 1)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        pos_weight: float = DEFAULT_POS_WEIGHT,
        faoc_loss: FAOCLoss | None = None,
    ):
        super().__init__()
        self.router = MoERegimeRouter(input_dim, hidden_dim)
        self.pos_weight = pos_weight
        self.faoc_loss_fn = faoc_loss if faoc_loss is not None else FAOCLoss()

    def forward(self, x: torch.Tensor, return_gate_weights: bool = False):
        """Passes straight through to MoERegimeRouter - Step 2's routing
        mechanism (including its optional gate weights) is fully exposed
        here, not hidden behind this wrapper."""
        return self.router(x, return_gate_weights=return_gate_weights)

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        airports: list,
        loss_type: str = "faoc",
    ) -> torch.Tensor:
        pred_probs = self.forward(x)
        if loss_type == "standard":
            return _weighted_bce(pred_probs, labels, self.pos_weight)
        elif loss_type == "faoc":
            return self.faoc_loss_fn(pred_probs, labels, airports)
        else:
            raise ValueError(f"loss_type must be 'standard' or 'faoc', got {loss_type!r}")
