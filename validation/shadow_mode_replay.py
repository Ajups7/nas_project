"""
The capstone integration for the Planning tier: wires together every prior
step against REAL data. Step 3's features -> Step 7's model -> Step 12's
ShadowModeLog -> Step 4's real labels (revealed only after prediction) ->
Step 9's KPI suite. Nothing else in this project runs the full pipeline
end to end - every earlier validation exercised one piece at a time.

Still bounded by the same real limitation as every prior step: the model
is untrained, so predictions are meaningless as forecasts. What this
proves is that the PLUMBING connects correctly across all six pieces, on
real downloaded data, not synthetic inputs - the final proof point before
this tier's Phase 1-3 work "holds short" per the build sequence doc.
"""

import torch

from features.planning import build_planning_features
from features.planning_label import build_planning_labels
from models.planning.planning_disruptnet import PlanningDisruptNet
from validation.shadow_mode import ShadowModeLog

FEATURE_COLUMNS = [
    "day_of_year_sin", "day_of_year_cos",
    "climo_tmpf", "climo_sknt", "climo_vsby", "climo_disruption_rate",
]


def run_shadow_mode_replay(
    model: PlanningDisruptNet,
    airport: str,
    dates,
    edct_months_for_climatology: list,
    horizon_days: int = 30,
) -> ShadowModeLog:
    """Replays shadow-mode operation against real historical data for one
    airport. Builds real features and real labels, records a prediction
    per date WITHOUT ever exposing the label to the model, then resolves
    each outcome only afterward - mirroring the temporal flow a live
    deployment would follow.
    """
    feature_table = build_planning_features(airport, dates, edct_months_for_climatology)
    label_table = build_planning_labels(airport, dates, horizon_days)

    log = ShadowModeLog()

    with torch.no_grad():
        for _, row in feature_table.iterrows():
            x = torch.tensor([[row[c] for c in FEATURE_COLUMNS]], dtype=torch.float32)
            pred_prob = model(x).item()
            log.record_prediction(row["date"], airport, pred_prob)

    for _, row in label_table.iterrows():
        log.resolve_outcome(row["date"], airport, bool(row["label"]))

    return log
