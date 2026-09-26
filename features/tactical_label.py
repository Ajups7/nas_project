"""
Person A (Tactical tier) - Step 2 of the build sequence: define the
tactical label, 0-2h horizon (see TEAM_PLAN.md).

Label definition: for a given (airport, date, local_hour), label = 1 if
EDCT activity occurred at that airport in any hour in the window
(ts, ts + horizon_hours] - "will a disruption occur sometime in the next
0-2h" - else 0.

Uses the EXACT SAME "had_edct" criterion as Person C's
features/planning_label.py: arrivals_with_edct > 0 OR
departures_with_edct > 0. Deliberately reusing that same criterion, not
inventing a different one, so "disruption" means the same thing across
all three tiers - only the look-ahead WINDOW differs (hours here vs. days
for Strategic/Planning).

NOT a thin wrapper over build_planning_labels the way
features/strategic_label.py is: that function's internals assume
whole-day windows. This module reimplements the same had_edct logic at
airport-HOUR granularity instead.

IMPORTANT - keyed on LOCAL hour, not gmt_hour: an earlier version built
its lookup timestamp as `date + timedelta(hours=gmt_hour)`, which wraps
mid-day for any airport not at UTC+0 (see features/tactical.py's module
docstring for the full explanation - the same bug existed here). Fixed by
using `date + local_hour` instead, which correctly represents LOCAL
elapsed time (one real hour per row, monotonic across local midnight),
with the same small known DST-transition-day caveat as the features
module. This module's public signature now takes `local_hours`, not
`gmt_hours` - callers must update accordingly.

Known limitation, same as Planning's and Strategic's labels: EDCT only
proxies Ground Delay Programs, not Ground Stops or Miles-in-Trail
restrictions.
"""

import pandas as pd

from common.data_loader import DATA_DIR, load_edct

DEFAULT_HORIZON_HOURS = 2  # matches this tier's own 0-2h forecast horizon


def _local_sort_key(dates: pd.Series, local_hours: pd.Series) -> pd.Series:
    """Collapses (date, local_hour) into one sortable/lookupable value -
    same helper as features/tactical.py, duplicated here rather than
    imported so a label module never needs to depend on the features
    module. Monotonic by construction (see that module's docstring for
    why gmt_hour was wrong here)."""
    dates = pd.to_datetime(pd.Series(dates))
    return dates + pd.to_timedelta(pd.Series(local_hours), unit="h")


def _load_edct_hourly_activity(
    airport: str, year_months: list[tuple[int, int]]
) -> pd.Series:
    """Loads the given EDCT months once, filters to `airport`, and
    returns one boolean per airport-hour timestamp: did THIS hour have
    EDCT activity? Indexed by the hourly `_ts` timestamp (local-hour
    based - see module docstring)."""
    frames = []
    for year, month in year_months:
        path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
        if not path.exists():
            print(f"build_tactical_labels: no EDCT file for {year}-{month:02d}, skipping (known gap)")
            continue
        edct = load_edct(year, month)
        frames.append(edct[edct["airport"] == airport])

    if not frames:
        raise ValueError(f"no EDCT data available for airport {airport} in months {year_months}")

    combined = pd.concat(frames, ignore_index=True).copy()
    combined["_ts"] = _local_sort_key(combined["date"], combined["local_hour"])
    combined["had_edct"] = (
        (combined["arrivals_with_edct"] > 0) | (combined["departures_with_edct"] > 0)
    )
    return combined.groupby("_ts")["had_edct"].any()


def build_tactical_labels(
    airport: str,
    dates: pd.Series,
    local_hours: pd.Series,
    year_months: list[tuple[int, int]],
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> pd.DataFrame:
    """For each (date, local_hour) pair, label = 1 if EDCT activity
    occurred at `airport` in any hour in (ts, ts + horizon_hours], else 0.

    Returns a table keyed by (airport, date, local_hour) - joins with
    features/tactical.py's build_tactical_features() output on those
    keys."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    local_hours = pd.Series(local_hours).reset_index(drop=True)
    ts = _local_sort_key(dates, local_hours)

    hourly_activity = _load_edct_hourly_activity(airport, year_months)

    labels = []
    for t in ts:
        window = pd.date_range(
            t + pd.Timedelta(hours=1), t + pd.Timedelta(hours=horizon_hours), freq="h"
        )
        occurred = hourly_activity.reindex(window, fill_value=False).any()
        labels.append(int(occurred))

    return pd.DataFrame({
        "airport": airport,
        "date": dates.values,
        "local_hour": local_hours.values,
        "label": labels,
    })
