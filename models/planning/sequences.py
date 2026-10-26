"""
Sequence inputs for the Seq2Seq / TFT baselines (models/planning/baselines.py):
for each labeled (airport, date), the past SEQ_LEN days of features ending
on `date`, shape (seq_len, num_features).

Each day carries the 12 Planning features (features/planning.py) plus that
day's own GDP-day indicator (common/disruption.py) - the raw history a
sequence model exists to exploit. Unknown GDP days are filled with the
airport's train-period climatology for that day (the expected value).

Labels, splits and COVID exclusion come from the cached Planning dataset,
so every model is scored on exactly the same rows.
"""

import numpy as np
import pandas as pd

from common.disruption import GDP_ARRIVAL_SHARE_THRESHOLD, gdp_days
from common.temporal_split import TRAIN_END
from features.planning import FEATURE_COLUMNS, build_planning_features, eligible_airports
from models.planning.data import fill_missing, load_planning_splits

SEQ_LEN = 30
SEQUENCE_COLUMNS = FEATURE_COLUMNS + ["gdp_day_today"]


def _daily_feature_matrix(
    airport: str, calendar: pd.DatetimeIndex, train_end: str = TRAIN_END,
    threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD,
) -> np.ndarray:
    feats = fill_missing(build_planning_features(
        airport, calendar.to_series().reset_index(drop=True), train_end=train_end, threshold=threshold))
    gdp = gdp_days(threshold)[airport].reindex(calendar)
    train_gdp = gdp.loc[:train_end]
    month_rate = train_gdp.groupby(train_gdp.index.month).mean()
    feats["gdp_day_today"] = gdp.fillna(pd.Series(calendar.month.map(month_rate), index=calendar)).to_numpy()
    return feats[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)


def daily_matrices(
    train_end: str = TRAIN_END, threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD, airport_onehot: bool = False,
) -> dict:
    """{airport: (days, features)}; airport_onehot appends a constant
    one-hot airport-identity block to every day."""
    calendar = gdp_days(threshold).index
    airports = eligible_airports(threshold)
    out = {a: _daily_feature_matrix(a, calendar, train_end, threshold) for a in airports}
    if airport_onehot:
        eye = np.eye(len(airports), dtype=np.float32)
        out = {a: np.hstack([m, np.repeat(eye[i][None, :], len(m), axis=0)])
               for i, (a, m) in enumerate(out.items())}
    return out


def build_sequence_splits(
    exclude_covid: bool = True, seq_len: int = SEQ_LEN, airport_onehot: bool = False,
    threshold: float = GDP_ARRIVAL_SHARE_THRESHOLD, dataset_path=None,
) -> dict:
    """Returns {split: (X [n, seq_len, n_feat], y [n], airports [n], frame)}.
    Rows whose window would start before the first calendar day are dropped."""
    kwargs = {} if dataset_path is None else {"dataset_path": dataset_path}
    frames = load_planning_splits(exclude_covid=exclude_covid, **kwargs)
    return windows_for_frames(frames, daily_matrices(threshold=threshold, airport_onehot=airport_onehot), seq_len)


def windows_for_frames(frames: dict, matrices: dict, seq_len: int = SEQ_LEN) -> dict:
    calendar = gdp_days().index
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    out = {}
    for name, df in frames.items():
        idx = pos.reindex(df["date"]).to_numpy()
        df = df[idx >= seq_len - 1].reset_index(drop=True)
        idx = idx[idx >= seq_len - 1]
        width = next(iter(matrices.values())).shape[1]
        X = np.empty((len(df), seq_len, width), dtype=np.float32)
        for airport, rows in df.groupby("airport").indices.items():
            m = matrices[airport]
            ends = idx[rows]
            window = ends[:, None] + np.arange(-seq_len + 1, 1)[None, :]
            X[rows] = m[window]
        out[name] = (X, df["label"].to_numpy(), df["airport"].to_numpy(), df)
    return out


def standardize(splits: dict) -> dict:
    """Per-feature mean/std from train (all timesteps), applied to every split."""
    X_tr = splits["train"][0]
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6
    return {k: ((X - mean) / std, y, a, df) for k, (X, y, a, df) in splits.items()}
