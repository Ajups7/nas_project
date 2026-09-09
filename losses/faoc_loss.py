"""
Phase 3 / Person C shared utility - Step 1 of the Planning-tier build
sequence (see TEAM_PLAN.md). Asymmetric, cost-calibrated loss: a missed
disruption (false negative) is penalized more heavily than a false alarm
(false positive), using the dollar figures sourced in cost_figures.md.

Two things this loss needs that plain weighted cross-entropy doesn't source
from anywhere public:

1. The FN:FP cost RATIO. No FAA/A4A source publishes "cost of a missed GDP
   prediction" vs. "cost of a false alarm" as distinct numbers - it depends
   on the operational response being modeled. Expressed below as two
   assumed exposure durations (avg_fn_hours, avg_fp_hours) rather than a
   bare ratio, so the reasoning stays inspectable and each can be
   recalibrated independently once Phase 2 baseline features exist (e.g.
   avg_fn_hours from the mean EDCT-derived disruption duration in the
   training set). See cost_figures.md §4.

2. Per-airport network effect. A missed disruption's cost is scaled by
   that airport's delay propagation multiplier (delay_propagation_
   multipliers.csv) - a miss at a high-multiplier airport cascades further
   than the same miss elsewhere. Applied to the FN term only: an
   unrealized false alarm doesn't cascade through the network the way a
   real disruption does.

Usage:
    from losses.faoc_loss import FAOCLoss

    criterion = FAOCLoss()
    loss = criterion(pred_probs, labels, airports)
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# A4A 2025 direct operating cost per block hour - cost_figures.md §1.
# Deliberately not the FAA's $9,359/hr fully-loaded figure (§2): that folds
# in overhead unrelated to the marginal cost of one delayed aircraft.
BASE_COST_PER_HOUR = 98.41 * 60  # $/block-minute * 60 = $5,904.60/hr

# FAA/MITRE national composite (cost_figures.md §3) - fallback for airports
# with no airport-specific multiplier (GA/reliever fields outside FAA's
# major-commercial-airport set, e.g. 7 of this project's 82 METAR airports).
NATIONAL_COMPOSITE_MULTIPLIER = 1.51

_MULTIPLIER_PATH = Path(__file__).resolve().parent / "delay_propagation_multipliers.csv"


def _load_multipliers(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["airport"], df["delay_propagation_multiplier"]))


class FAOCLoss(nn.Module):
    """Asymmetric binary cross-entropy for disruption prediction.

    False negatives are weighted as avg_fn_hours of block-hour cost, scaled
    by the missed airport's delay propagation multiplier. False positives
    are weighted as avg_fp_hours, unscaled. Both durations are placeholder
    modeling assumptions, not sourced figures - see module docstring and
    cost_figures.md §4.
    """

    def __init__(
        self,
        avg_fn_hours: float = 2.0,
        avg_fp_hours: float = 0.25,
        base_cost_per_hour: float = BASE_COST_PER_HOUR,
        multiplier_path: Path = _MULTIPLIER_PATH,
    ):
        super().__init__()
        if avg_fn_hours <= 0 or avg_fp_hours <= 0:
            raise ValueError("avg_fn_hours and avg_fp_hours must be positive")
        self.fn_weight = base_cost_per_hour * avg_fn_hours
        self.fp_weight = base_cost_per_hour * avg_fp_hours
        self.multipliers = _load_multipliers(multiplier_path)

    def _multiplier_for(self, airports, device, dtype) -> torch.Tensor:
        return torch.tensor(
            [self.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in airports],
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        pred_probs: torch.Tensor,
        labels: torch.Tensor,
        airports: list,
    ) -> torch.Tensor:
        """
        pred_probs: predicted P(disruption in next 1-30d), shape (N,), in (0, 1).
        labels: ground-truth binary labels, shape (N,).
        airports: length-N list of airport codes, one per sample.
        """
        if not (pred_probs.shape == labels.shape and len(airports) == pred_probs.shape[0]):
            raise ValueError("pred_probs, labels, and airports must all have length N")

        eps = 1e-7
        p = pred_probs.clamp(eps, 1 - eps)
        labels = labels.to(dtype=p.dtype)
        dm = self._multiplier_for(airports, device=p.device, dtype=p.dtype)

        fn_term = self.fn_weight * dm * labels * torch.log(p)
        fp_term = self.fp_weight * (1 - labels) * torch.log(1 - p)
        return -(fn_term + fp_term).mean()
