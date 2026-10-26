"""
Loads the cached Strategic dataset (features/strategic.py) into
train / validation / test frames - same conventions as the Planning tier
(models/planning/data.py): embargo dropped, COVID excluded from train,
one-hot airport identity added.

Missing values: a missing max gust means no gust was reported, so it is set
to that day's max wind; every other gap is filled with the column's
median over train-period days (COVID excluded) - never with
validation/test statistics.
"""

import pandas as pd

from common.temporal_split import TRAIN_END
from features.strategic import DAILY_PATH, FEATURE_COLUMNS, LABELS_PATH
from models.planning.data import add_airport_onehot, in_covid

SPLITS = ("train", "validation", "test")


def load_daily(path=DAILY_PATH) -> pd.DataFrame:
    daily = pd.read_parquet(path)
    daily["wx_max_gust"] = daily["wx_max_gust"].fillna(daily["wx_max_sknt"])
    train_days = (daily["date"] <= TRAIN_END) & ~in_covid(daily["date"])
    medians = daily.loc[train_days, FEATURE_COLUMNS].median()
    daily[FEATURE_COLUMNS] = daily[FEATURE_COLUMNS].fillna(medians)
    return daily


def load_strategic_splits(exclude_covid: bool = True, daily: pd.DataFrame | None = None,
                          labels_path=LABELS_PATH) -> dict[str, pd.DataFrame]:
    daily = load_daily() if daily is None else daily
    labels = pd.read_parquet(labels_path)
    df = labels[labels["split"] != "embargo"].merge(daily, on=["airport", "date"], how="left")
    if exclude_covid:
        df = df[~(in_covid(df["date"]) & (df["split"] == "train"))]
    df = add_airport_onehot(df)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
