"""
Person C (Planning tier) - Step 3 of the build sequence: features for the
1-30 day horizon. One row per (airport, date); the prediction is made at the
end of `date` about the window (date+1, date+30].

Feature groups:
1. Calendar - cyclical day-of-year.
2. Climatology over the forecast window - for each of the next 30 days, the
   train-period average for that airport and calendar month, then averaged
   across the window. So a June 25 prediction sees mostly July's climate.
   - weather: METAR means (tmpf, dwpf, drct, sknt, gust, vsby, alti)
   - disruption: fraction of GDP days (common/disruption.py)
   Computed from train dates only (<= TRAIN_END): averaging over all years
   would put validation/test GDPs into their own features.
3. Recent history - the airport's GDP-day rate over the past 7 and 30 days
   (through `date` itself). Known at prediction time, so not leakage; it
   captures persistent regimes (construction, a stormy stretch) that
   climatology can't. Train with and without it (RECENT_COLUMNS) to measure
   what it adds.
4. Recent anomaly (ANOMALY_COLUMNS) - recent rate minus the airport's
   train-period climatology for those same past days. The raw recent rate
   mostly re-encodes "busy airport"; the anomaly keeps only "worse/better
   than usual here."

All-airport output is cached by build_planning_dataset().
"""

import numpy as np
import pandas as pd

from common.data_loader import DATA_DIR, load_weather
from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD, gdp_days
from common.temporal_split import TRAIN_END, VALIDATION_END, assign_temporal_split
from features.planning_label import DEFAULT_HORIZON_DAYS, build_planning_labels

WEATHER_NUMERIC_COLS = ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "alti"]
CALENDAR_COLUMNS = ["day_of_year_sin", "day_of_year_cos"]
CLIMO_COLUMNS = [f"climo_{c}" for c in WEATHER_NUMERIC_COLS] + ["climo_gdp_day_rate"]
RECENT_COLUMNS = ["recent_gdp_rate_7d", "recent_gdp_rate_30d"]
ANOMALY_COLUMNS = ["recent_gdp_anom_7d", "recent_gdp_anom_30d"]
FEATURE_COLUMNS = CALENDAR_COLUMNS + CLIMO_COLUMNS + RECENT_COLUMNS

DATASET_PATH = DATA_DIR / "cache" / "planning_dataset.parquet"


def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    day_of_year = dates.dt.dayofyear
    days_in_year = np.where(dates.dt.is_leap_year, 366, 365)
    return pd.DataFrame({
        "date": dates.values,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / days_in_year),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / days_in_year),
    })


def build_weather_climatology(airport: str, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Mean METAR conditions per calendar month (1-12), train dates only.
    IEM's "M" missing-value sentinel is coerced to NaN and excluded."""
    wx = load_weather(airport)
    wx = wx[wx["date"] <= pd.Timestamp(train_end)].copy()
    wx[WEATHER_NUMERIC_COLS] = wx[WEATHER_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    clim = wx.groupby(wx["date"].dt.month)[WEATHER_NUMERIC_COLS].mean()
    clim.columns = [f"climo_{c}" for c in WEATHER_NUMERIC_COLS]
    return clim.reindex(range(1, 13))


def build_disruption_climatology(
    airport: str, train_end: str = TRAIN_END, threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD
) -> pd.Series:
    """Fraction of known train-period days that were GDP days, per calendar month."""
    s = gdp_days(threshold)[airport].loc[:train_end]
    return s.groupby(s.index.month).mean().reindex(range(1, 13)).rename("climo_gdp_day_rate")


def _forward_window_mean(daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """mean(daily[t+1 .. t+horizon_days]); NaN when the window runs off the end."""
    return daily[::-1].rolling(horizon_days, min_periods=horizon_days).mean()[::-1].shift(-1)


def build_planning_features(
    airport: str,
    dates: pd.Series,
    train_end: str = TRAIN_END,
    threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Feature table keyed by (airport, date)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    gdp = gdp_days(threshold)[airport]

    # Daily calendar extended past the last date so forward windows near the
    # end of the data still resolve - climatology is known for any future date.
    span = pd.date_range(min(dates.min(), gdp.index.min()),
                         max(dates.max(), gdp.index.max()) + pd.Timedelta(days=horizon_days))
    monthly = build_weather_climatology(airport, train_end).join(
        build_disruption_climatology(airport, train_end, threshold))
    daily_climo = monthly.reindex(span.month).set_index(span)
    climo = _forward_window_mean(daily_climo, horizon_days)

    known = gdp.reindex(span)
    past_climo = daily_climo["climo_gdp_day_rate"]
    recent = pd.DataFrame({
        "recent_gdp_rate_7d": known.rolling(7, min_periods=1).mean(),
        "recent_gdp_rate_30d": known.rolling(30, min_periods=1).mean(),
    })
    # Recent rate minus the airport's normal rate for those same past days:
    # "worse than usual for here, at this time of year", without the
    # "this is a busy airport" component climatology already carries.
    recent["recent_gdp_anom_7d"] = recent["recent_gdp_rate_7d"] - past_climo.rolling(7, min_periods=1).mean()
    recent["recent_gdp_anom_30d"] = recent["recent_gdp_rate_30d"] - past_climo.rolling(30, min_periods=1).mean()

    out = build_calendar_features(dates)
    out.insert(0, "airport", airport)
    out = out.join(climo.reindex(dates).reset_index(drop=True))
    out = out.join(recent.reindex(dates).reset_index(drop=True))
    return out


def build_planning_dataset(
    airports: list[str],
    threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Features + label + split for every (airport, date) in the EDCT range.
    Rows without a confirmable label are dropped (see planning_label.py)."""
    all_dates = gdp_days(threshold).index.to_series().reset_index(drop=True)
    frames = []
    for airport in airports:
        feats = build_planning_features(airport, all_dates, threshold=threshold, horizon_days=horizon_days)
        labels = build_planning_labels(airport, all_dates, horizon_days=horizon_days, threshold=threshold)
        frames.append(feats.merge(labels, on=["airport", "date"], how="inner"))
    df = pd.concat(frames, ignore_index=True)
    df["split"] = assign_temporal_split(df["date"], TRAIN_END, VALIDATION_END).values
    return df


MIN_TRAIN_COVERAGE = 0.90


def eligible_airports(threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD) -> list[str]:
    """Airports with a known GDP-day value on >= 90% of train days. Excludes
    APA, ASE, BJC, BOI, CMH (in ASPM only from mid-2023) and OXR, GYY (patchy
    arrival data: dropping unconfirmable 0s leaves OXR 95% positive on 4% of
    its rows - a biased label, not a real one)."""
    coverage = gdp_days(threshold).loc[:TRAIN_END].notna().mean()
    return sorted(coverage[coverage >= MIN_TRAIN_COVERAGE].index)


if __name__ == "__main__":
    import time

    t = time.time()
    airports = eligible_airports()
    df = build_planning_dataset(airports)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    print(f"{len(airports)} airports, {len(df):,} rows in {time.time() - t:.0f}s -> {DATASET_PATH}")
