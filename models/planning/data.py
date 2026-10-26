"""
Loads the cached Planning dataset (features/planning.py) into train /
validation / test frames, ready for any model in this tier.

- Embargo rows are dropped (never used by any split).
- Missing recent-history values (windows inside the 2016-07 EDCT gap) are
  filled with the airport's forward-window climatology - the natural "no
  recent information" estimate; missing anomalies with 0 ("normal").
- One-hot airport columns (AIRPORT_COLUMNS) are added; models opt in by
  listing them in their feature columns.
- exclude_covid drops COVID_START..COVID_END from train only: traffic
  collapsed and GDPs nearly vanished (2020 label prevalence 14% vs 36-38%
  in 2016-2019), which is not the regime validation/test operate in.
- Standardizer is fit on train rows only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features.planning import ANOMALY_COLUMNS, DATASET_PATH, FEATURE_COLUMNS, RECENT_COLUMNS, eligible_airports

COVID_START = "2020-03-01"
COVID_END = "2021-06-30"
SPLITS = ("train", "validation", "test")


AIRPORT_COLUMNS = [f"ap_{a}" for a in eligible_airports()]


def add_airport_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot airport identity. Without it, the ~10-22 quiet airports look
    alike (GDP climatology ~0, recent rate 0) and PlanningDisruptNet predicts
    their pooled ~5-9% rate - right at the ~7% decision threshold
    (models/planning/fix_probability_floor.py)."""
    onehot = pd.DataFrame({c: (df["airport"] == c[3:]).astype(np.float32) for c in AIRPORT_COLUMNS},
                          index=df.index)
    return pd.concat([df, onehot], axis=1)


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in RECENT_COLUMNS:
        df[c] = df[c].fillna(df["climo_gdp_day_rate"])
    for c in ANOMALY_COLUMNS:
        df[c] = df[c].fillna(0.0)
    return df


def in_covid(dates: pd.Series) -> pd.Series:
    return dates.between(COVID_START, COVID_END)


def load_planning_splits(exclude_covid: bool = False, dataset_path=DATASET_PATH) -> dict[str, pd.DataFrame]:
    """dataset_path: the cached 10%-threshold dataset by default; the label-
    threshold sensitivity check passes its own variant."""
    df = pd.read_parquet(dataset_path)
    df = add_airport_onehot(fill_missing(df[df["split"] != "embargo"]))
    if exclude_covid:
        df = df[~(in_covid(df["date"]) & (df["split"] == "train"))]
    return {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}


class Standardizer:
    def __init__(self, feature_columns: list[str] = FEATURE_COLUMNS):
        self.feature_columns = list(feature_columns)
        self.scaler = StandardScaler()

    def fit(self, train: pd.DataFrame) -> "Standardizer":
        self.scaler.fit(train[self.feature_columns].to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.feature_columns].to_numpy()).astype(np.float32)
