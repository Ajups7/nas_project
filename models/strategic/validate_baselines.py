"""
Validates models/strategic/baselines.py two ways:

1. End-to-end with REAL Strategic-tier data (unlike
   models/planning/validate_baselines.py, which used purely synthetic
   torch.randn tensors): features/strategic.py's build_strategic_features()
   -> make_sequence_windows() -> Seq2SeqBaseline/TFTBaseline -> composes
   with losses/faoc_loss.py's FAOCLoss, and SARIMABaseline fit directly on
   ATL's real recent_disruption_rate series.
2. The same mechanical checks models/planning/validate_baselines.py runs
   (output shape, range, input-sensitivity, order-sensitivity for TFT) -
   these architectures are freshly initialized and untrained, so this
   proves the mechanisms are wired correctly for THIS tier's real feature
   width, not that either model has learned anything.

Run from the repo root:
    python -m models.strategic.validate_baselines
"""

import numpy as np
import pandas as pd
import torch

from features.strategic import build_strategic_features
from features.strategic_label import build_strategic_labels
from losses.faoc_loss import FAOCLoss
from models.planning.baselines import SARIMABaseline
from models.strategic.baselines import build_strategic_baselines, make_sequence_windows

torch.manual_seed(0)

AIRPORT = "ATL"
DATES = pd.date_range("2023-06-01", "2023-06-25")
YEAR_MONTHS = [(2023, 5), (2023, 6)]
SEQ_LEN = 10


def main():
    print(f"=== Real data: build_strategic_features + build_strategic_labels, {AIRPORT}, "
          f"{DATES.min().date()}-{DATES.max().date()} ===")
    features = build_strategic_features(AIRPORT, DATES, YEAR_MONTHS)
    labels = build_strategic_labels(AIRPORT, DATES)
    feature_cols = [c for c in features.columns if c not in ("airport", "date")]
    num_features = len(feature_cols)
    print(f"{len(features)} days, {num_features} feature columns: {feature_cols}")
    nan_counts = features[feature_cols].isna().sum()
    print(f"NaN columns before fill: {nan_counts[nan_counts > 0].to_dict()}")

    windows = make_sequence_windows(features, feature_cols, seq_len=SEQ_LEN)
    window_labels = torch.tensor(labels["label"].to_numpy()[SEQ_LEN - 1:], dtype=torch.float32)
    print(f"windows shape: {tuple(windows.shape)} "
          f"(expect ({len(features) - SEQ_LEN + 1}, {SEQ_LEN}, {num_features}))")
    print(f"window_labels shape: {tuple(window_labels.shape)} (must match windows.shape[0])")
    assert windows.shape[0] == window_labels.shape[0]

    baselines = build_strategic_baselines(num_features)
    criterion = FAOCLoss()
    airports = [AIRPORT] * windows.shape[0]

    for name in ("seq2seq", "tft"):
        print()
        print(f"=== {name} on real Strategic windows ===")
        model = baselines[name]
        pred_probs = model(windows)
        print(f"output shape: {tuple(pred_probs.shape)} (expect ({windows.shape[0]},))")
        print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")

        shuffled = windows[:, torch.randperm(SEQ_LEN), :]
        pred_probs_shuffled = model(shuffled)
        print(f"shuffling the time axis changes the output "
              f"(model is order-sensitive): {not torch.allclose(pred_probs, pred_probs_shuffled)}")

        loss = criterion(pred_probs, window_labels, airports)
        print(f"composes with FAOCLoss on real labels, no error: loss={loss.item():,.2f}")

    print()
    print("=== SARIMABaseline on ATL's real recent_disruption_rate series ===")
    series = features.set_index("date")["recent_disruption_rate"].dropna()
    print(f"series length (after dropping any NaN rows): {len(series)}")
    print("Using a lighter order than SARIMABaseline's default (1,1,1)/(1,1,1,7) -")
    print("that default double-differences and needs more than this window's 25")
    print("points to estimate seasonal parameters (confirmed: it emits statsmodels'")
    print("'too few observations' EstimationWarning on this series). Same override")
    print("models/planning/validate_baselines.py applies to its own short synthetic")
    print("series - SARIMABaseline's own docstring already says its defaults are a")
    print("guess to be tuned against real data, not trusted blindly.")
    model = SARIMABaseline(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))
    model.fit(series)
    forecast = model.forecast(steps=7)
    print(f"forecast length: {len(forecast)} (expect 7)")
    print(f"forecast values finite (no NaN/inf): {bool(np.isfinite(forecast).all())}")
    print(forecast.to_string())


if __name__ == "__main__":
    main()
