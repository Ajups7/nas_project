"""
Shared definition of a "disruption" (GDP day), used by all three tiers'
labels so the word means the same thing everywhere.

GDP day: at least GDP_ARRIVAL_SHARE_THRESHOLD of the airport's arrivals that
day had an EDCT. Arrivals only: a GDP holds flights bound FOR the
constrained airport, so arrival EDCTs at X reflect a GDP at X, while
departure EDCTs at X reflect GDPs at X's destinations. The 10% threshold
sits at the trough of the train-period distribution of daily arrival-EDCT
share and is where days start showing multi-hour held arrivals - see
features/planning_label_validation.md, "Label redesign (2026-10)".

Values: 1.0 GDP day, 0.0 not, NaN unknown (no EDCT file for that month, or
the airport reported no arrivals that day).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from common.aspm_cache import load_edct_cache

GDP_ARRIVAL_SHARE_THRESHOLD = 0.10


@lru_cache(maxsize=1)
def daily_arrival_edct_share() -> pd.DataFrame:
    """Wide table: index = every calendar date covered by the cache,
    columns = airport, value = share of that day's arrivals with an EDCT."""
    e = load_edct_cache(["airport", "date", "arrivals_for_metric", "arrivals_with_edct"])
    g = e.groupby(["airport", "date"])[["arrivals_for_metric", "arrivals_with_edct"]].sum()
    share = g["arrivals_with_edct"] / g["arrivals_for_metric"].replace(0, np.nan)
    wide = share.unstack("airport")
    calendar = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(calendar)


def gdp_days(threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD) -> pd.DataFrame:
    share = daily_arrival_edct_share()
    return (share >= threshold).astype(float).where(share.notna())


# Hourly definition (Tactical tier). Among hours with >= 5 arrivals, the
# train-period share of arrivals with an EDCT falls from 10% to a trough at
# 30-50%, then rises to a second mode at 70-80% (hours inside a running
# program). At >= 50%, 97.5% of flagged hours fall on GDP days by the daily
# definition. At least 2 held arrivals, so a single held flight at a small
# airport is not a "program".
GDP_HOUR_SHARE_THRESHOLD = 0.50
GDP_HOUR_MIN_HELD = 2


def gdp_hours(share_threshold: float = GDP_HOUR_SHARE_THRESHOLD, min_held: int = GDP_HOUR_MIN_HELD) -> pd.DataFrame:
    """Long table (airport, date, local_hour, gdp_hour) for every airport-hour
    present in the EDCT report. Hours absent from a day the report covers had
    no arrivals and are not GDP hours; days the report doesn't cover at all
    (missing months) are unknown - callers decide how to treat both."""
    e = load_edct_cache(["airport", "date", "local_hour", "arrivals_for_metric", "arrivals_with_edct"])
    share = e["arrivals_with_edct"] / e["arrivals_for_metric"].replace(0, np.nan)
    e["gdp_hour"] = ((share >= share_threshold) & (e["arrivals_with_edct"] >= min_held)).astype(np.float32)
    return e[["airport", "date", "local_hour", "gdp_hour"]]
