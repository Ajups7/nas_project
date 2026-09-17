"""
Person C (Planning tier) - Step 4 of the build sequence: define the
planning label. Tier-local work (see TEAM_PLAN.md) - a separate concern
from planning.py's features (Step 3), even though both read EDCT data.

Label definition: for a given (airport, date), label = 1 if EDCT activity
occurred at that airport on any day in the window (date+1, date+horizon_days]
inclusive, else 0 - "will a disruption occur sometime in the next 1-30
days," which is exactly what PLAN.md's Phase 1 item 4 (and this module's
name) commit to.

Why "any EDCT activity" and not a percentage threshold: an EDCT (Expect
Departure Clearance Time) is only assigned when the FAA is actively running
a Ground Delay Program - see schema.md, "GDP proxy label source." Its mere
presence at an airport-hour already means a GDP was in effect; there's no
need to invent an arbitrary "% of flights affected" cutoff on top of that,
the way FAOC-Loss's avg_fn_hours/avg_fp_hours had to (see
losses/cost_figures.md §4). This is a real advantage of using EDCT as a
label source over a made-up threshold.

Known limitation (see schema.md): EDCT only proxies Ground Delay Programs,
not Ground Stops or Miles-in-Trail restrictions - so this label is a lower
bound on "any disruption occurred," not the true GDP/GS/MIT label PLAN.md
ultimately wants. That real label needs ASPM Advisories access, still
blocked on FAA account approval (README.md).

Performance note (same root cause as planning.py's disruption climatology):
load_edct() takes ~13s/month. build_planning_labels() loads every month
needed to cover ALL requested dates' windows exactly ONCE up front, not
once per date - looping build_planning_label() below per-date would reload
overlapping months repeatedly and be far slower for anything but a single
date.
"""

import pandas as pd

from common.data_loader import DATA_DIR, load_edct

DEFAULT_HORIZON_DAYS = 30


def _months_spanned(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """Every (year, month) touched by [start, end], inclusive."""
    months = []
    cursor = start.to_period("M")
    end_period = end.to_period("M")
    while cursor <= end_period:
        months.append((cursor.year, cursor.month))
        cursor += 1
    return months


def _load_edct_daily_activity(airport: str, months: list[tuple[int, int]]) -> pd.Series:
    """Loads the given EDCT months once, filters to `airport`, and collapses
    to one boolean per date: did ANY hour that day have EDCT activity
    (arrivals_with_edct > 0 or departures_with_edct > 0)? Indexed by date.
    Missing months (see README.md, "2 known gaps") are skipped with a
    warning, same file-existence check as planning.py's disruption
    climatology - see that module's docstring for why a try/except around
    load_edct() itself isn't the right way to detect this."""
    frames = []
    for year, month in months:
        path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
        if not path.exists():
            print(f"build_planning_labels: no EDCT file for {year}-{month:02d}, skipping (known gap)")
            continue
        edct = load_edct(year, month)
        frames.append(edct[edct["airport"] == airport])

    if not frames:
        raise ValueError(f"no EDCT data available for airport {airport} in months {months}")

    combined = pd.concat(frames, ignore_index=True)
    combined["had_edct"] = (
        (combined["arrivals_with_edct"] > 0) | (combined["departures_with_edct"] > 0)
    )
    return combined.groupby("date")["had_edct"].any()


def build_planning_labels(
    airport: str,
    dates: pd.Series,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """For each date in `dates`, label = 1 if EDCT activity occurred at
    `airport` on any day in (date+1, date+horizon_days], else 0.

    Returns a table keyed by (airport, date) - joins directly with
    planning.py's build_planning_features() output on those same keys.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    window_starts = dates + pd.Timedelta(days=1)
    window_ends = dates + pd.Timedelta(days=horizon_days)

    months_needed: set[tuple[int, int]] = set()
    for start, end in zip(window_starts, window_ends):
        months_needed.update(_months_spanned(start, end))

    daily_activity = _load_edct_daily_activity(airport, sorted(months_needed))

    labels = []
    for start, end in zip(window_starts, window_ends):
        window = pd.date_range(start, end)
        occurred = daily_activity.reindex(window, fill_value=False).any()
        labels.append(int(occurred))

    return pd.DataFrame({"airport": airport, "date": dates.values, "label": labels})
