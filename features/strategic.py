"""
Person B (Strategic tier) - Step 2 of the build sequence: day-ahead
features for the 2-24h forecast horizon (see TEAM_PLAN.md).

Four feature groups, each answering "what would you actually know about
this airport, the day before the forecast day":

1. Calendar/seasonal signals - reused directly from
   features.planning.build_calendar_features rather than reimplemented:
   it's pure date arithmetic with no tier-specific logic in it (day of
   week, cyclical day-of-year), so duplicating it here would just be two
   copies of the same function to keep in sync.
2. Recent operational conditions - yesterday's actual ASPM Airport
   Analysis (capacity/demand, delays, taxi times), collapsed from
   airport-hour to one airport-day row.
3. Recent disruption activity - yesterday's actual EDCT rate (this
   project's GDP proxy - see schema.md), same underlying signal as
   features/planning.py's disruption climatology, but the ACTUAL prior day
   rather than a historical seasonal average.
4. Recent weather actuals - yesterday's actual METAR summary, not a
   forecast (METAR is observed - see schema.md) and not a climatology
   average either.

Why yesterday's actuals, not climatology (Planning tier) or a live nowcast
(Tactical tier): at a 2-24h horizon there's no real weather forecast to
lean on any more than Planning has, but there IS a much stronger near-term
signal available than "typical for this time of year" - short-term
persistence. A Ground Delay Program yesterday raises the odds of continued
disruption today (backlogged aircraft/crew, cascading rotation delays -
see losses/delay_propagation_multipliers.csv for the cost side of exactly
this phenomenon), and weather is itself autocorrelated day-to-day. Using
today's OWN actuals to predict today would be leakage; using yesterday's
is the last point that's both informative and genuinely knowable in
advance at this horizon.

Mechanically: each of groups 2-4 is built keyed by its own observation
date, then shifted forward by exactly one day before merging, so that the
row for forecast date D holds date (D-1)'s actual values - this keeps the
shift logic in one place (build_strategic_features) rather than every
helper needing to know it's really building "yesterday's" table.

Output tables are keyed by (airport, date), matching features/planning.py's
convention (see that module's docstring for why local_hour/gmt_hour are
correctly omitted from a daily-frequency table).
"""

import pandas as pd

from common.data_loader import DATA_DIR, load_airport_analysis, load_edct, load_weather
from features.planning import build_calendar_features

WEATHER_NUMERIC_COLS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "alti"]
OPERATIONAL_NUMERIC_COLS = [
    "pct_ontime_gate_dep", "pct_ontime_airport_dep", "pct_ontime_gate_arr",
    "avg_gate_dep_delay", "avg_taxi_out_time", "avg_taxi_out_delay",
    "avg_airport_dep_delay", "avg_airborne_delay", "avg_taxi_in_delay",
    "avg_block_delay", "avg_gate_arr_delay",
]


def build_recent_operational_features(
    airport: str, year_months: list[tuple[int, int]]
) -> pd.DataFrame:
    """One row per date, ASPM Airport Analysis's airport-hour rows
    collapsed to a daily mean, for `airport` across the given
    (year, month) list. Columns prefixed `recent_` - these describe the
    OWN observation date, not yet shifted forward; that shift happens in
    build_strategic_features, not here, so this function stays reusable
    for anyone who wants same-day (not day-ahead) operational aggregates."""
    frames = [load_airport_analysis(year, month) for year, month in year_months]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["airport"] == airport]
    daily = combined.groupby("date")[OPERATIONAL_NUMERIC_COLS].mean().reset_index()
    daily.columns = ["date"] + [f"recent_{c}" for c in OPERATIONAL_NUMERIC_COLS]
    daily.insert(0, "airport", airport)
    return daily


def build_recent_disruption_features(
    airport: str, year_months: list[tuple[int, int]]
) -> pd.DataFrame:
    """One row per date: `recent_disruption_rate`, the mean of
    pct_arrivals_with_edct and pct_departures_with_edct across that date's
    airport-hours - the same quantity as features/planning.py's
    disruption climatology, but per-date rather than averaged across years
    into a per-month climatology.

    Missing EDCT months (README.md, "2 known gaps") are skipped with a
    warning via a direct file-existence check, not a try/except around
    load_edct() - same reasoning as features/planning.py and
    features/planning_label.py's identical check: a missing .xls makes
    pandas.read_html fail with a generic "No tables found" ValueError that
    looks like a parsing bug, not a missing-file one.
    """
    frames = []
    for year, month in year_months:
        path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
        if not path.exists():
            print(f"build_recent_disruption_features: no EDCT file for {year}-{month:02d}, skipping (known gap)")
            continue
        edct = load_edct(year, month)
        frames.append(edct[edct["airport"] == airport])

    if not frames:
        raise ValueError(f"no EDCT data available for airport {airport} in months {year_months}")

    combined = pd.concat(frames, ignore_index=True)
    combined["disruption_rate"] = combined[
        ["pct_arrivals_with_edct", "pct_departures_with_edct"]
    ].mean(axis=1)
    daily = combined.groupby("date")["disruption_rate"].mean().reset_index()
    daily.columns = ["date", "recent_disruption_rate"]
    daily.insert(0, "airport", airport)
    return daily


def build_recent_weather_features(airport: str, dates: pd.Series) -> pd.DataFrame:
    """One row per date in `dates`, METAR observations collapsed to a
    daily mean, for `airport`. Coerces the numeric columns explicitly
    before aggregating - load_weather() returns them as object dtype
    because of IEM's "M" missing-observation sentinel (confirmed on real
    data - see features/planning_features_validation.md, Issue 1 - the
    same data_loader.py-level gap, worked around here the same way
    features/planning.py's build_weather_climatology() does it)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    date_range = (dates.min(), dates.max())

    wx = load_weather(airport).copy()
    wx = wx[(wx["date"] >= date_range[0]) & (wx["date"] <= date_range[1])]
    wx[WEATHER_NUMERIC_COLS] = wx[WEATHER_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    daily = wx.groupby("date")[WEATHER_NUMERIC_COLS].mean().reset_index()
    daily.columns = ["date"] + [f"recent_{c}" for c in WEATHER_NUMERIC_COLS]
    daily.insert(0, "airport", airport)
    return daily


def build_strategic_features(
    airport: str,
    dates: pd.Series,
    year_months: list[tuple[int, int]],
) -> pd.DataFrame:
    """Assembles all four feature groups into one table keyed by
    (airport, date): calendar features for date D itself, plus
    operational/disruption/weather actuals from date (D-1), shifted
    forward by one day so they join directly on D.

    `year_months` must cover every month touched by (min(dates) - 1) ..
    max(dates), inclusive, so the day-before lookup for the earliest
    requested date has data to find - the caller decides this window (see
    features/planning.py's build_disruption_climatology for the same
    caller-decides-the-window convention)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    calendar = build_calendar_features(dates)
    calendar.insert(0, "airport", airport)

    one_day = pd.Timedelta(days=1)
    operational = build_recent_operational_features(airport, year_months)
    operational["date"] = operational["date"] + one_day
    disruption = build_recent_disruption_features(airport, year_months)
    disruption["date"] = disruption["date"] + one_day
    weather = build_recent_weather_features(airport, dates - one_day)
    weather["date"] = weather["date"] + one_day

    out = calendar.merge(operational, on=["airport", "date"], how="left")
    out = out.merge(disruption, on=["airport", "date"], how="left")
    out = out.merge(weather, on=["airport", "date"], how="left")
    return out
