"""
Person B (Strategic tier) - Step 3 of the build sequence: define the
strategic label, 2-24h horizon (see TEAM_PLAN.md).

Label definition: for a given (airport, date), label = 1 if EDCT activity
occurred at that airport on date+1 (the next calendar day), else 0 -
"will a disruption occur sometime in the next day," matching this tier's
own "day-ahead" framing the same way features/strategic.py (Step 2) uses
yesterday's actuals to predict tomorrow.

Deliberately NOT reimplemented from scratch:
features/planning_label.py's build_planning_labels(airport, dates,
horizon_days) already computes exactly "did EDCT activity occur in
(date+1, date+horizon_days]" - with horizon_days=1, that window collapses
to exactly (date+1, date+1], i.e. "did it occur on date+1," which is
precisely this tier's label. Reusing that function outright avoids
duplicating its EDCT-loading/windowing logic (including its handling of
the 2 known missing EDCT months - see README.md) as a second copy that
would need to be kept in sync by hand.

CROSS-TIER DEPENDENCY this creates, flagged the same way
models/planning/regime_transformer.py flags its dependency on this
person's rotation_graph.py: if Person C ever changes
build_planning_labels's signature or window semantics, this module breaks
silently unless they give a heads-up first - the same "shared code changes
rarely, and needs a heads-up first" rule TEAM_PLAN.md sets for the
officially shared utilities (FAOC-Loss, rotation graph, temporal split),
applied here by convention even though planning_label.py isn't on that
official list.

Known limitation, same as Planning's label (see that module's docstring
for the full explanation): EDCT only proxies Ground Delay Programs, not
Ground Stops or Miles-in-Trail restrictions, so this is a lower bound on
"any disruption occurred," not the true GDP/GS/MIT label PLAN.md
ultimately wants.
"""

import pandas as pd

from features.planning_label import build_planning_labels

DEFAULT_HORIZON_DAYS = 1


def build_strategic_labels(airport: str, dates: pd.Series) -> pd.DataFrame:
    """For each date in `dates`, label = 1 if EDCT activity occurred at
    `airport` on date+1 (the very next calendar day), else 0. Thin wrapper
    over features.planning_label.build_planning_labels with
    horizon_days=1 fixed - see module docstring for why that's exactly
    this tier's label rather than an arbitrary reuse.

    Returns a table keyed by (airport, date) - joins directly with
    features/strategic.py's build_strategic_features() output on those
    same keys."""
    return build_planning_labels(airport, dates, horizon_days=DEFAULT_HORIZON_DAYS)
