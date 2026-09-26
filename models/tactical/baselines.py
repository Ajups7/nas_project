"""
Person A (Tactical tier) - Step 5 of the build sequence: baseline
architectures. "SARIMA, seq2seq, TFT - written, not run" (per
TEAM_PLAN.md). Same Phase 2 goal as Strategic/Planning: a working,
evaluable baseline before any novel component is layered in.

Same reuse pattern as features/strategic_label.py's reuse of Person C's
build_planning_labels, and models/strategic/baselines.py's reuse of
Person C's three baseline classes: SARIMABaseline, Seq2SeqBaseline, and
TFTBaseline (models/planning/baselines.py) are fully generic - no
tier-specific logic in any of them - so this tier reuses them directly
rather than maintaining a third, identical copy of that code.
make_sequence_windows() (models/strategic/baselines.py) is equally
generic and should be imported from there directly by any caller that
needs to turn features/tactical.py's output into model input windows -
not re-imported here, since this module doesn't call it itself.

CROSS-TIER DEPENDENCY this creates, same as Strategic's: if Person B or
Person C changes either reused module, this one picks up that change
automatically - the same "heads-up before changing" expectation
TEAM_PLAN.md sets for the officially shared utilities, applied here by
convention.

ONE Tactical-specific override, called out explicitly rather than left
implicit: SARIMABaseline's default seasonal_order=(1, 1, 1, 7) assumes a
DAILY-granularity series with weekly seasonality. Tactical's series is
HOURLY, where the dominant repeating cycle is a 24-hour day, not a
7-day week - build_tactical_baselines() below passes
seasonal_order=(1, 1, 1, 24) explicitly rather than inheriting the
daily-tier default, which would be actively wrong here.

No training happens here - same caveat as every prior architecture in
this project: mechanically validated, not trained against real data,
gated on the team-agreed temporal split (still an unsigned-off proposal
as of this writing - see common/temporal_split.py).
"""

from models.planning.baselines import SARIMABaseline, Seq2SeqBaseline, TFTBaseline

TACTICAL_SEASONAL_ORDER = (1, 1, 1, 24)  # 24-hour daily cycle, not the
                                           # daily-tier default's 7-day week


def build_tactical_baselines(num_features: int) -> dict:
    """Returns one freshly-initialized instance of each of the three
    shared baseline architectures, matching build_strategic_baselines's
    shape. SARIMABaseline gets an hourly-appropriate seasonal_order (see
    module docstring) rather than its daily-tuned default."""
    return {
        "sarima": SARIMABaseline(seasonal_order=TACTICAL_SEASONAL_ORDER),
        "seq2seq": Seq2SeqBaseline(num_features=num_features),
        "tft": TFTBaseline(num_features=num_features),
    }
