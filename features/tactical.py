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
exposes an exact `gmt_hour` too), never `local_hour_approx`.

The lightweight rotation signal (step 3) is mean inbound ArrDelay from
BTS, NOT the full rotation-chain graph (features/rotation_graph.py is
Person B's shared utility for that - see TEAM_PLAN.md, "tier-local work"
vs "shared utility"). It's joined on `local_hour`, not `gmt_hour` - BTS's
ArrTime is already local to the Dest airport (no cross-airport timezone
conversion needed here, unlike weather), but BTS gives no gmt_hour
equivalent directly.

NOTE: common/temporal_split.py is still an unsigned-off PROPOSAL as of
this writing (see that file's own docstring) - Step 4 of this build
sequence is blocked on Person A/Person B actually agreeing to it.
"""

import pandas as pd

from common.data_loader import DATA_DIR, load_airport_analysis, load_bts, load_edct, load_weather
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


def _hhmm_to_hour(series: pd.Series) -> pd.Series:
    """Converts a BTS hhmm-formatted time column (e.g. 1345, or 2400 for
    midnight) into an integer local hour (0-23). Missing values
    (cancelled/diverted flights have no ArrTime) come out as NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric // 100) % 24


def build_lightweight_rotation_signal(
    airport: str, year_months: list[tuple[int, int]], window_hours: int = WINDOW_HOURS
) -> pd.DataFrame:
    """One row per (date, local_hour): mean ArrDelay of flights that
    landed at `airport` (BTS's `Dest`) in that local hour, plus a rolling
    mean/delta over the trailing `window_hours` - a fast proxy for "is
    inbound traffic currently backed up," deliberately NOT the full
    rotation-chain graph (features/rotation_graph.py is Person B's shared
    utility for that).

    Cancelled and diverted flights are excluded (no real ArrDelay to
    measure). Known approximation: BTS's FlightDate is the scheduled
    DEPARTURE date, so a flight landing just after local midnight is
    still bucketed under its departure day here rather than its true
    arrival day - BTS exposes no separate arrival-date field. Same class
    of approximation as schema.md's own local_hour_approx note for
    weather - acceptable for a fast, lightweight signal, not a
    load-bearing exact join."""
    frames = [load_bts(year, month) for year, month in year_months]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        (combined["Dest"] == airport)
        & (combined["Cancelled"] != 1)
        & (combined["Diverted"] != 1)
    ].copy()

    combined["local_hour"] = _hhmm_to_hour(combined["ArrTime"])
    combined = combined.dropna(subset=["local_hour", "ArrDelay"])
    combined["local_hour"] = combined["local_hour"].astype(int)
    combined["date"] = pd.to_datetime(combined["FlightDate"])

    hourly = (
        combined.groupby(["date", "local_hour"])["ArrDelay"]
        .mean()
        .reset_index()
        .rename(columns={"ArrDelay": "inbound_arr_delay"})
    )
    hourly["_ts"] = hourly["date"] + pd.to_timedelta(hourly["local_hour"], unit="h")
    rolled = _rolling_mean_and_delta(hourly, ["inbound_arr_delay"], window_hours)
    out = hourly[["date", "local_hour"]].reset_index(drop=True)
    out.insert(0, "airport", airport)
    return pd.concat([out, rolled], axis=1)


def build_tactical_features(
    airport: str,
    year_months: list[tuple[int, int]],
    window_hours: int = WINDOW_HOURS,
) -> pd.DataFrame:
    """Assembles the near-term operational, EDCT, weather, and rotation
    feature groups into one table keyed by (airport, date, local_hour,
    gmt_hour). Weather is joined on (airport, date, gmt_hour) only, since
    load_weather() doesn't expose local_hour. The rotation signal is
    joined on (airport, date, local_hour) instead - BTS gives no
    gmt_hour equivalent directly - so rows where BTS has no inbound
    traffic that hour come out with NaN rotation features, which is
    correct (not missing data, just genuinely no inbound flights)."""
    operational = build_near_term_operational_features(airport, year_months, window_hours)
    edct = build_near_term_edct_features(airport, year_months, window_hours)
    weather = build_near_term_weather_features(airport, window_hours)
    rotation = build_lightweight_rotation_signal(airport, year_months, window_hours)

    out = operational.merge(
        edct, on=["airport", "date", "local_hour", "gmt_hour"], how="left", suffixes=("", "_edct")
    )
    out = out.merge(
        weather, on=["airport", "date", "gmt_hour"], how="left", suffixes=("", "_wx")
    )
    out = out.merge(
        rotation, on=["airport", "date", "local_hour"], how="left", suffixes=("", "_rot")
    )
    return out
