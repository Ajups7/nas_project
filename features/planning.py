"""
Person C (Planning tier) - Step 3 of the build sequence: long-horizon
features for the 1-30 day forecast horizon. Tier-local work (see
TEAM_PLAN.md) - not a shared utility like FAOC-Loss or MoE routing.

Three feature groups, each answering "what would you know about this
airport N days out, before any real weather forecast exists that far
ahead":

1. Calendar/seasonal signals - day-of-week, month, cyclical day-of-year -
   pure date arithmetic, no data loading required.
2. Historical disruption frequency ("disruption climatology") - how often
   has this airport, in this calendar month, actually had EDCT-flagged
   disruption activity historically? Built from common.data_loader.load_edct.
3. Weather climatology - typical (not actual/forecast) weather for this
   airport in this calendar month, built from common.data_loader.load_weather.

Why climatology and not nowcasts: at a 1-30 day horizon there is no real
weather forecast to feed the model (METAR is observed, not predicted - see
schema.md) and no reliable near-term EDCT signal either (that's Tactical/
Strategic's job, at their 0-2h/2-24h horizons). The only defensible signal
this far out is "what does this airport typically look like at this time
of year" - climatology - which is exactly what these functions compute.

Output tables are keyed by (airport, date) only, not the full 4-key
schema.md join (airport, date, local_hour, gmt_hour): these are daily-
frequency signals with no meaningful hourly variation, so local_hour/
gmt_hour would just be duplicated 24x for no benefit. When joining to an
hourly table downstream, broadcast this table's rows across all 24 hours of
their date rather than expecting an exact 4-key match.

Known cost (see planning_features_validation.md for measurements):
load_edct() parses one .xls file per call via pandas.read_html, at roughly
13 seconds/month measured. Building disruption climatology across the full
2016-2026 history (119 of 121 months - see the docstring below for the 2
known gaps) takes on the order of 25 minutes - a one-time batch job, not
something to re-run per training loop. Cache the result.
"""

import numpy as np
import pandas as pd

from common.data_loader import DATA_DIR, load_edct, load_weather

WEATHER_NUMERIC_COLS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "alti"]


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """dates: a Series of dates (any dtype pandas can coerce to datetime).
    Returns calendar/seasonal features, one row per input date, cyclically
    encoded so e.g. Dec 31 and Jan 1 are numerically close together instead
    of maximally far apart."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    day_of_year = dates.dt.dayofyear
    days_in_year = np.where(dates.dt.is_leap_year, 366, 365)

    return pd.DataFrame({
        "date": dates.values,
        "month": dates.dt.month,
        "day_of_week": dates.dt.dayofweek,  # 0=Monday
        "is_weekend": dates.dt.dayofweek.isin([5, 6]),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / days_in_year),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / days_in_year),
    })


def build_weather_climatology(airport: str) -> pd.DataFrame:
    """Historical average weather conditions by calendar month, from every
    year of downloaded METAR history for this airport. One row per month
    (1-12) - not per (year, month), since climatology is defined as
    "typical for this time of year," collapsed across all available years.

    load_weather() returns these columns as object/string dtype, not
    numeric: the raw METAR files use IEM's "M" sentinel for a missing
    observation (routine for `gust`, which only reports when a gust
    actually occurs - ~89% "M" in ATL's history), and a single non-numeric
    value forces pandas to infer the whole column as object dtype. Coerced
    to numeric here (invalid values, i.e. "M", become NaN and are excluded
    from the mean) - this is a data_loader.py-level gap that would affect
    anyone doing numeric work on load_weather()'s output, not something
    specific to this module. Worth raising with the team rather than
    silently working around it everywhere it's used.
    """
    wx = load_weather(airport).copy()
    wx["month"] = wx["date"].dt.month
    wx[WEATHER_NUMERIC_COLS] = wx[WEATHER_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    clim = wx.groupby("month")[WEATHER_NUMERIC_COLS].mean().reset_index()
    clim.columns = ["month"] + [f"climo_{c}" for c in WEATHER_NUMERIC_COLS]
    clim.insert(0, "airport", airport)
    return clim


def build_disruption_climatology(
    airport: str, months: list[tuple[int, int]]
) -> pd.DataFrame:
    """Historical disruption frequency by calendar month, from EDCT data.
    `months` is the list of (year, month) pairs to pull - the caller
    decides the window (ideally full history; see module docstring for why
    that's slow and should be cached, not recomputed per call).

    "Disruption frequency" here = the mean of pct_arrivals_with_edct and
    pct_departures_with_edct across all airport-hours in that calendar
    month, across every year requested - i.e. what fraction of flights,
    typically, needed an EDCT (this project's GDP proxy - see schema.md)
    during this time of year at this airport.

    2 of the 121 expected EDCT months are missing entirely (download
    timeouts that were never recovered - see README.md, "2 known gaps," and
    data/aspm_edct_report/download_log_2016_retry1.txt for 2016-07
    specifically). Checked for directly with a file-existence test, not a
    try/except around load_edct(): when the .xls is missing, pandas'
    read_html falls through to treating the path string itself as literal
    HTML and raises a generic ValueError("No tables found") rather than
    anything file-specific - too easy to accidentally swallow a real
    parsing failure if caught by exception type instead. Skipped here with
    a warning, since a caller requesting the full history shouldn't have to
    know about this gap in advance.
    """
    frames = []
    for year, month in months:
        path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
        if not path.exists():
            print(f"build_disruption_climatology: no EDCT file for {year}-{month:02d}, skipping (known gap)")
            continue
        edct = load_edct(year, month)
        frames.append(edct[edct["airport"] == airport])
    if not frames:
        raise ValueError(f"no EDCT data available for any of {months}")
    combined = pd.concat(frames, ignore_index=True)
    combined["month"] = combined["date"].dt.month
    combined["disruption_rate"] = combined[
        ["pct_arrivals_with_edct", "pct_departures_with_edct"]
    ].mean(axis=1)
    clim = combined.groupby("month")["disruption_rate"].mean().reset_index()
    clim.columns = ["month", "climo_disruption_rate"]
    clim.insert(0, "airport", airport)
    return clim


def build_planning_features(
    airport: str,
    dates: pd.Series,
    edct_months: list[tuple[int, int]],
) -> pd.DataFrame:
    """Assembles all three feature groups into one table, keyed by
    (airport, date) - see module docstring for why local_hour/gmt_hour are
    intentionally omitted."""
    calendar = build_calendar_features(dates)
    calendar.insert(0, "airport", airport)

    weather_clim = build_weather_climatology(airport)
    disruption_clim = build_disruption_climatology(airport, edct_months)

    out = calendar.merge(weather_clim, on=["airport", "month"], how="left")
    out = out.merge(disruption_clim, on=["airport", "month"], how="left")
    return out
