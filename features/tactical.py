"""
Person A (Tactical tier) - Steps 1 and 3 of the build sequence: near-term
features and a lightweight rotation signal for the 0-2h forecast horizon
(see TEAM_PLAN.md).

Unlike Strategic/Planning's once-a-day feature rows, Tactical operates at
(airport, date, local_hour, gmt_hour) granularity - a 0-2h horizon needs
to know what's true RIGHT NOW, not yesterday's actuals or a seasonal
average.

Each near-term source (ASPM Airport Analysis, EDCT, METAR) contributes
both a rolling MEAN over the trailing WINDOW_HOURS and a rolling DELTA
(this window's mean minus the preceding window's mean) - the delta is
what actually signals "getting worse right now" versus "has been like
this all day," which a plain rolling mean can't distinguish.

Weather is joined on `gmt_hour` (confirmed present in load_weather()'s
output - schema.md only documents `local_hour_approx`, but the loader
exposes an exact `gmt_hour` too), never `local_hour_approx` - at this
tier's hour-level granularity, the approximation schema.md warns about
would actually matter, unlike for Strategic/Planning's day-level joins.

NOTE: common/temporal_split.py is still an unsigned-off PROPOSAL as of
this writing (see that file's own docstring) - Step 4 of this build
sequence is blocked on Person A/Person B actually agreeing to it.
"""

import pandas as pd

from common.data_loader import DATA_DIR, load_airport_analysis, load_edct, load_weather
from features.strategic import OPERATIONAL_NUMERIC_COLS, WEATHER_NUMERIC_COLS

WINDOW_HOURS = 3  # matches TEAM_PLAN.md's "last 1-3h" near-term window

EDCT_NUMERIC_COLS = [
    "arrivals_with_edct", "pct_arrivals_with_edct", "avg_edct_all_arrivals",
    "avg_edct_arrivals_gt0", "arr_edct_late15_count", "arr_edct_late15_pct",
    "departures_with_edct", "pct_departures_with_edct", "avg_edct_all_departures",
    "avg_edct_departures_gt0", "dep_edct_late15_count", "dep_edct_late15_pct",
]


def _airport_hour_index(dates: pd.Series, gmt_hours: pd.Series) -> pd.Series:
    """Collapses (date, gmt_hour) into one sortable UTC timestamp, used
    internally for rolling windows - never exposed in output tables,
    which keep date/local_hour/gmt_hour as separate columns per
    schema.md's convention."""
    dates = pd.to_datetime(pd.Series(dates))
    return dates + pd.to_timedelta(pd.Series(gmt_hours), unit="h")


def _rolling_mean_and_delta(
    frame: pd.DataFrame, value_cols: list[str], window_hours: int
) -> pd.DataFrame:
    """For a frame sorted by an hourly `_ts` column, computes per value
    column: `{col}_rolling_mean` (trailing window_hours, current hour
    included) and `{col}_rolling_delta` (that mean minus the mean of the
    PRECEDING window of the same width) - the delta flags "getting worse
    right now"."""
    frame = frame.sort_values("_ts").reset_index(drop=True)
    roll = frame[value_cols].rolling(window=window_hours, min_periods=1)
    rolling_mean = roll.mean()
    prev_rolling_mean = rolling_mean.shift(window_hours)
    out = pd.DataFrame(index=frame.index)
    for col in value_cols:
        out[f"{col}_rolling_mean"] = rolling_mean[col]
        out[f"{col}_rolling_delta"] = rolling_mean[col] - prev_rolling_mean[col]
    return out


def build_near_term_operational_features(
    airport: str, year_months: list[tuple[int, int]], window_hours: int = WINDOW_HOURS
) -> pd.DataFrame:
    """One row per (date, local_hour, gmt_hour): ASPM Airport Analysis's
    own airport-hour rows plus their rolling mean/delta over the trailing
    `window_hours`."""
    frames = [load_airport_analysis(year, month) for year, month in year_months]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["airport"] == airport].copy()
    combined["_ts"] = _airport_hour_index(combined["date"], combined["gmt_hour"])
    rolled = _rolling_mean_and_delta(combined, OPERATIONAL_NUMERIC_COLS, window_hours)
    out = combined[["airport", "date", "local_hour", "gmt_hour"]].reset_index(drop=True)
    return pd.concat([out, rolled], axis=1)


def build_near_term_edct_features(
    airport: str, year_months: list[tuple[int, int]], window_hours: int = WINDOW_HOURS
) -> pd.DataFrame:
    """Same shape as build_near_term_operational_features, over EDCT
    counts. Missing EDCT months are skipped with a warning, same
    known-gap handling as features/strategic.py and features/planning.py."""
    frames = []
    for year, month in year_months:
        path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
        if not path.exists():
            print(f"build_near_term_edct_features: no EDCT file for {year}-{month:02d}, skipping (known gap)")
            continue
        edct = load_edct(year, month)
        frames.append(edct[edct["airport"] == airport])

    if not frames:
        raise ValueError(f"no EDCT data available for airport {airport} in months {year_months}")

    combined = pd.concat(frames, ignore_index=True).copy()
    combined["_ts"] = _airport_hour_index(combined["date"], combined["gmt_hour"])
    rolled = _rolling_mean_and_delta(combined, EDCT_NUMERIC_COLS, window_hours)
    out = combined[["airport", "date", "local_hour", "gmt_hour"]].reset_index(drop=True)
    return pd.concat([out, rolled], axis=1)


def build_near_term_weather_features(
    airport: str, window_hours: int = WINDOW_HOURS
) -> pd.DataFrame:
    """One row per (date, gmt_hour): METAR observations plus their
    rolling mean/delta over the trailing `window_hours`. Coerces the
    numeric columns explicitly first, same fix as features/strategic.py's
    build_recent_weather_features (IEM's "M" missing-observation sentinel
    - see features/planning_features_validation.md, Issue 1)."""
    wx = load_weather(airport).copy()
    wx[WEATHER_NUMERIC_COLS] = wx[WEATHER_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    wx["_ts"] = _airport_hour_index(wx["date"], wx["gmt_hour"])
    rolled = _rolling_mean_and_delta(wx, WEATHER_NUMERIC_COLS, window_hours)
    out = wx[["airport", "date", "gmt_hour"]].reset_index(drop=True)
    return pd.concat([out, rolled], axis=1)


def build_lightweight_rotation_signal(airport: str, year_months: list[tuple[int, int]]):
    """PENDING - Step 3. Mean arrival delay of flights that landed at
    `airport` (BTS's `Dest`) in the trailing WINDOW_HOURS, as a fast
    proxy for "is inbound traffic currently backed up" - deliberately
    NOT the full rotation-chain graph (features/rotation_graph.py is
    Person B's shared utility for that).

    Not yet implemented: BTS's arrival columns (CRSArrTime, ArrTime,
    WheelsOn) aren't pre-parsed into (date, gmt_hour) the way ASPM/EDCT
    are - needs its own timestamp-parsing step first."""
    raise NotImplementedError("BTS arrival timestamps need parsing into (date, gmt_hour) first")


def build_tactical_features(
    airport: str,
    year_months: list[tuple[int, int]],
    window_hours: int = WINDOW_HOURS,
) -> pd.DataFrame:
    """Assembles the near-term operational, EDCT, and weather feature
    groups into one table keyed by (airport, date, local_hour, gmt_hour).
    Weather is joined on (airport, date, gmt_hour) only, since
    load_weather() doesn't expose a local_hour column - local_hour in the
    output comes from the operational table.

    The lightweight rotation signal (step 3) is NOT yet included - see
    build_lightweight_rotation_signal's docstring."""
    operational = build_near_term_operational_features(airport, year_months, window_hours)
    edct = build_near_term_edct_features(airport, year_months, window_hours)
    weather = build_near_term_weather_features(airport, window_hours)

    out = operational.merge(
        edct, on=["airport", "date", "local_hour", "gmt_hour"], how="left", suffixes=("", "_edct")
    )
    out = out.merge(
        weather, on=["airport", "date", "gmt_hour"], how="left", suffixes=("", "_wx")
    )
    return out
