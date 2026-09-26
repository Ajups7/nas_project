"""
Person A (Tactical tier) - Step 6 of the build sequence: the model stub.
"Tactical DisruptNet tier with weighted cross-entropy." Unlike Planning's
Step 7, this tier doesn't choose between two losses at runtime -
TEAM_PLAN.md assigns only "weighted cross-entropy" to Tactical, so
compute_loss() below offers exactly that, not a loss_type switch the way
PlanningDisruptNet does.

Architecture: TFTBaseline (Step 5, models/tactical/baselines.py - itself
reused from Person C's models/planning/baselines.py) - the most
expressive of this tier's three baseline architectures. No new
architecture is defined here, same "assemble already-built pieces" spirit
as PlanningDisruptNet: TEAM_PLAN.md doesn't assign Tactical a separate
novel-architecture step of its own (unlike Strategic's STGNN or Planning's
Regime Transformer) - the "hold short" line this tier converges at only
requires a baseline + a standard-loss stub, not a novel one.

Loss helper is a LOCAL copy of models/planning/planning_disruptnet.py's
_weighted_bce, not an import of it - that function's leading underscore
marks it module-private by convention, so importing it across tiers would
reach past a boundary Person C set deliberately. Duplicating this one
small, generic function is a smaller cost than depending on someone
else's private implementation detail.

No training happens here - same caveat as every prior architecture in
this project: mechanically validated, not trained, gated on the
team-agreed temporal split (still an unsigned-off proposal as of this
writing - see common/temporal_split.py).
"""

import torch
import torch.nn as nn

from models.tactical.baselines import TFTBaseline

DEFAULT_POS_WEIGHT = 1.0  # class-imbalance weight, placeholder - same
                           # spirit as PlanningDisruptNet's own
                           # DEFAULT_POS_WEIGHT: meant to be recalibrated
                           # once real label prevalence is known
                           # (features/tactical_label.py), not trusted as
                           # tuned. The smoke test earlier showed roughly
                           # 57%/43% for ATL/June 2024 - close enough to
                           # balanced that 1.0 isn't unreasonable as a
                           # starting point, but that's one airport, one
                           # month.


def _weighted_bce(pred_probs: torch.Tensor, labels: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Standard weighted binary cross-entropy - weighted by class
    imbalance (pos_weight), nothing to do with dollar cost or airport.
    Written manually rather than via nn.BCEWithLogitsLoss(pos_weight=...)
    because that class expects raw logits, and TFTBaseline's forward()
    already applies sigmoid - see models/tactical/baselines.py."""
    eps = 1e-7
    p = pred_probs.clamp(eps, 1 - eps)
    labels = labels.to(dtype=p.dtype)
    loss = -(pos_weight * labels * torch.log(p) + (1 - labels) * torch.log(1 - p))
    return loss.mean()


class TacticalDisruptNet(nn.Module):
    """The Tactical tier's model. Architecture = TFTBaseline (Step 5).
    Loss is fixed to weighted cross-entropy, per TEAM_PLAN.md - unlike
    PlanningDisruptNet, there is no second loss to choose between here."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        pos_weight: float = DEFAULT_POS_WEIGHT,
    ):
        super().__init__()
        self.tft = TFTBaseline(num_features=num_features, hidden_dim=hidden_dim, num_heads=num_heads)
        self.pos_weight = pos_weight

    def forward(self, x: torch.Tensor, return_variable_weights: bool = False):
        """Passes straight through to TFTBaseline - its variable-selection
        weights (Step 5) are fully exposed here, not hidden behind this
        wrapper, same transparency choice as PlanningDisruptNet.forward()."""
        return self.tft(x, return_variable_weights=return_variable_weights)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred_probs = self.forward(x)
        return _weighted_bce(pred_probs, labels, self.pos_weight)
