"""
Person C (Planning tier) - Step 12 of the build sequence: shadow-mode
deployment. "Runs alongside live ops without acting on predictions, to
validate before cutover." Shared utility (TEAM_PLAN.md) - the FINAL step,
and the build sequence's own "hold short" convergence point: "All three
tiers converge here: features built, labels defined, baselines and
novel-architecture stubs written, evaluation wired up. Who runs training,
and how, is decided later."

Honest scope: there is no live ops system for this project to shadow yet -
no real-time data feed, no existing production GDP-prediction process to
run alongside, and no trained model (every prior step's caveat still
applies). What's built here is a REPLAY-based shadow-mode simulator: walk
real historical dates in chronological order, record a prediction at each
one, and only reveal the real outcome afterward - mirroring the temporal
flow of a live deployment (predict now, learn the truth once the horizon
elapses), even though a replay technically already has the "future"
sitting in the historical data.

The one property that actually defines "shadow mode" is enforced by
construction, not by convention: ShadowModeLog has no method that could
feed a prediction back into anything. Predictions are recorded via
record_prediction() and never mutated; outcomes are stored in a SEPARATE
dict that can only grow, keyed independently, so there is no code path
anywhere in this class that could let a prediction influence a real
decision. validate_shadow_mode.py includes a test that introspects the
class's public API and confirms it's limited to exactly this safe surface.
"""

from dataclasses import dataclass

import pandas as pd

SAFE_PUBLIC_METHODS = {"record_prediction", "resolve_outcome", "resolved_pairs", "unresolved_count"}


@dataclass(frozen=True)
class ShadowPrediction:
    date: pd.Timestamp
    airport: str
    predicted_probability: float


class ShadowModeLog:
    """Append-only, purely observational record of shadow-mode
    predictions. Predictions are recorded once and never modified; outcomes
    live in a separate dict that can only be added to. There is no method
    here - and by design, none should ever be added - that lets a
    prediction affect anything beyond this log."""

    def __init__(self):
        self._predictions: list[ShadowPrediction] = []
        self._outcomes: dict[tuple, bool] = {}

    def record_prediction(self, date, airport: str, predicted_probability: float) -> None:
        self._predictions.append(ShadowPrediction(pd.Timestamp(date), airport, predicted_probability))

    def resolve_outcome(self, date, airport: str, outcome: bool) -> None:
        """Fills in the real outcome once it's knowable (i.e. once the
        label's forward-looking window - features/planning_label.py - has
        actually elapsed). Raises if there's no matching recorded
        prediction, rather than silently creating one out of thin air."""
        key = (pd.Timestamp(date), airport)
        if key not in {(p.date, p.airport) for p in self._predictions}:
            raise ValueError(f"no recorded prediction for ({airport}, {pd.Timestamp(date).date()})")
        self._outcomes[key] = outcome

    def resolved_pairs(self) -> list[tuple]:
        """(predicted_probability, outcome) for every prediction whose
        outcome is now known - the input format validation/kpi_suite.py's
        metrics expect."""
        return [
            (p.predicted_probability, self._outcomes[(p.date, p.airport)])
            for p in self._predictions
            if (p.date, p.airport) in self._outcomes
        ]

    def unresolved_count(self) -> int:
        return sum(1 for p in self._predictions if (p.date, p.airport) not in self._outcomes)
