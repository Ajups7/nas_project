"""
Person C (Planning tier) - Step 4 of the build sequence: the planning label.

For a given (airport, date): label = 1 if a GDP day (common/disruption.py)
occurs at that airport on any day in (date+1, date+horizon_days], else 0.

Unknown days (missing EDCT month, or no arrivals reported) are handled
conservatively: a window containing any known GDP day is still a 1, but a
window with no GDP day AND at least one unknown day can't be called a 0, so
that row is dropped. Dates whose window runs past the end of the data are
dropped for the same reason. The returned table can therefore be shorter
than `dates` - join on (airport, date), don't assume row alignment.

features/strategic_label.py wraps this with horizon_days=1 ("GDP day
tomorrow"), so changes here change the Strategic label too.

History: until 2026-10 the label was "any EDCT, arrival or departure" - 90%
positive in train, 65 of 82 airports >= 95% positive. See
planning_label_validation.md, "Label redesign (2026-10)".

Known limitation: EDCT proxies Ground Delay Programs only, not Ground Stops
or Miles-in-Trail.
"""

import pandas as pd

from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD, gdp_days

DEFAULT_HORIZON_DAYS = 30


def _forward_window_max(s: pd.Series, horizon_days: int) -> pd.Series:
    """max(s[t+1 .. t+horizon_days]); NaN when the window runs off the end."""
    return s[::-1].rolling(horizon_days, min_periods=horizon_days).max()[::-1].shift(-1)


def build_planning_labels(
    airport: str,
    dates: pd.Series,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
) -> pd.DataFrame:
    """Returns (airport, date, label) for each labelable date in `dates`."""
    days = gdp_days(threshold)
    if airport not in days.columns or days[airport].isna().all():
        raise ValueError(f"no arrival EDCT data for airport {airport}")
    s = days[airport]

    any_gdp = _forward_window_max(s.fillna(0), horizon_days)
    any_unknown = _forward_window_max(s.isna().astype(float), horizon_days)
    label = any_gdp.where((any_gdp == 1) | (any_unknown == 0))

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    out = pd.DataFrame({"airport": airport, "date": dates.values,
                        "label": label.reindex(dates).values})
    return out.dropna(subset=["label"]).astype({"label": int}).reset_index(drop=True)
