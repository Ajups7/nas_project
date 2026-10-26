"""
Shadow-mode replay for the Planning tier: walks real dates in chronological
order as if live. On each simulated day it records that day's predictions
in a ShadowModeLog (validation/shadow_mode.py), and resolves an outcome
only once its label window has fully elapsed (as-of date + horizon_days).
Nothing a prediction does can reach back into the model or the data.

`predict_fn(rows) -> probabilities` wraps any frozen model with its own
preprocessing (e.g. models.planning.compare_losses.predict_from_checkpoint),
so the replay uses exactly what was evaluated.
"""

import numpy as np
import pandas as pd

from features.planning_label import DEFAULT_HORIZON_DAYS
from validation.shadow_mode import ShadowModeLog


def run_shadow_mode_replay(frame: pd.DataFrame, predict_fn, horizon_days: int = DEFAULT_HORIZON_DAYS):
    """frame: rows with airport, date, label and the model's features.
    Returns (log, pending): pending[i] = predictions still unresolved at the
    end of simulated day i, the live-deployment backlog."""
    log = ShadowModeLog()
    by_date = {d: g for d, g in frame.groupby("date")}
    days = sorted(by_date)

    def resolve(d):
        for _, r in by_date[d].iterrows():
            log.resolve_outcome(d, r["airport"], bool(r["label"]))

    pending, j = [], 0
    for day in days:
        rows = by_date[day]
        for (_, r), p in zip(rows.iterrows(), np.asarray(predict_fn(rows))):
            log.record_prediction(day, r["airport"], float(p))
        # Every as-of date whose window has elapsed by today - including ones
        # whose exact maturity day had no rows (data gaps).
        while j < len(days) and days[j] <= day - pd.Timedelta(days=horizon_days):
            resolve(days[j])
            j += 1
        pending.append(log.unresolved_count())

    # After the last day, let the clock run until every window has elapsed.
    for d in days[j:]:
        resolve(d)
    return log, pending
