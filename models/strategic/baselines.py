"""
Person B (Strategic tier) - Step 4 of the build sequence: the Strategic
baseline model (see TEAM_PLAN.md). PLAN.md's Phase 2 goal is a working,
evaluable baseline before any novel component (the rotation-chain GNN,
Phase 3) gets layered in - "benchmarked against the seq2seq / Temporal
Fusion Transformer / SARIMA baselines named in the paper's Section 5."

No new architectures are defined here. models/planning/baselines.py's
SARIMABaseline, Seq2SeqBaseline, and TFTBaseline are fully generic - no
planning-specific logic anywhere in them, just num_features/hidden_dim/
decode_steps constructor parameters and a forward(x) -> (batch,)
probability interface - so this tier reuses them directly rather than
maintaining a second, identical copy of that PyTorch code. Confirmed with
the user as the preferred approach over duplicating into this directory.
This is a deliberate exception to TEAM_PLAN.md's "separate directories,
rare cross-imports" design, made specifically because the code being
reused is provably generic rather than tier-specific.

CROSS-TIER DEPENDENCY this creates, same as features/strategic_label.py's
reuse of Person C's build_planning_labels: if Person C changes
models/planning/baselines.py's classes, this module picks up that change
automatically - the same "heads-up before changing" expectation
TEAM_PLAN.md sets for the officially shared utilities (FAOC-Loss, rotation
graph, temporal split), applied here by convention even though
models/planning/ isn't on that official list.

What IS Strategic-specific: make_sequence_windows() below turns
features/strategic.py's (Step 2) daily feature table into the
(batch, seq_len, num_features) windows Seq2SeqBaseline/TFTBaseline expect
- something Person C's own validate_baselines.py didn't need, since it
validated purely with synthetic torch.randn tensors rather than a real
per-tier feature pipeline. See validate_baselines.py for this module's
real-data end-to-end check: build_strategic_features (Step 2) ->
make_sequence_windows -> Seq2SeqBaseline/TFTBaseline -> FAOCLoss, and
SARIMABaseline fit directly on ATL's real recent_disruption_rate series
rather than a synthetic sine wave.

No training happens here - same caveat as every prior architecture in this
project: mechanically validated, not trained against real data, gated on
the team-agreed temporal split (TEAM_PLAN.md).
"""

import numpy as np
import pandas as pd
import torch

from models.planning.baselines import SARIMABaseline, Seq2SeqBaseline, TFTBaseline


def make_sequence_windows(
    features: pd.DataFrame, feature_cols: list[str], seq_len: int
) -> torch.Tensor:
    """features: a table for ONE airport, already sorted by date with no
    gaps (e.g. features/strategic.py's build_strategic_features() output
    over a contiguous date range) - gaps/multiple airports aren't detected
    or handled here, since the caller controls both in this project's
    current usage.

    Returns (num_windows, seq_len, len(feature_cols)), window i covering
    rows [i, i+seq_len) - matches Seq2SeqBaseline/TFTBaseline's expected
    input shape. num_windows = len(features) - seq_len + 1; raises if
    `features` has fewer than seq_len rows.

    NaNs in `feature_cols` are filled with 0 before windowing - the only
    column expected to have any, in practice, is recent_gust (see
    strategic_features_validation.md): a missing gust observation means no
    gust event occurred that day, so 0 is the domain-correct fill, not an
    arbitrary one.
    """
    values = features[feature_cols].fillna(0).to_numpy(dtype="float32")
    num_windows = len(values) - seq_len + 1
    if num_windows < 1:
        raise ValueError(f"need at least {seq_len} rows, got {len(values)}")
    windows = np.stack([values[i:i + seq_len] for i in range(num_windows)])
    return torch.from_numpy(windows)


def build_strategic_baselines(num_features: int) -> dict:
    """Returns one freshly-initialized instance of each of the three
    shared baseline architectures. SARIMABaseline doesn't take
    num_features (it's univariate, fit per-airport on a single series) -
    included anyway so a caller can benchmark against all three baselines
    from one place, matching PLAN.md Phase 2's "benchmarked against
    seq2seq/TFT/SARIMA" framing."""
    return {
        "sarima": SARIMABaseline(),
        "seq2seq": Seq2SeqBaseline(num_features=num_features),
        "tft": TFTBaseline(num_features=num_features),
    }
