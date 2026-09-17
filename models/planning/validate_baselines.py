"""
Validates baselines.py with synthetic inputs - same philosophy as
validate_moe_router.py (Step 2): these architectures are freshly
initialized and untrained, so this proves the mechanisms are wired
correctly, not that any of them has learned anything. No real feature data
is used - that's gated on the team-agreed temporal split (TEAM_PLAN.md).

Run from the repo root:
    python -m models.planning.validate_baselines
"""

import numpy as np
import pandas as pd
import torch

from losses.faoc_loss import FAOCLoss
from models.planning.baselines import SARIMABaseline, Seq2SeqBaseline, TFTBaseline

torch.manual_seed(0)

BATCH_SIZE = 5
SEQ_LEN = 10
NUM_FEATURES = 6


def validate_seq2seq():
    print("=== Seq2SeqBaseline ===")
    model = Seq2SeqBaseline(num_features=NUM_FEATURES)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    pred_probs = model(x)
    print(f"output shape: {tuple(pred_probs.shape)} (expect ({BATCH_SIZE},))")
    print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")

    x2 = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES) * 5
    pred_probs2 = model(x2)
    print(f"different inputs -> different outputs: {not torch.allclose(pred_probs, pred_probs2)}")

    criterion = FAOCLoss()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    airports = ["ORD", "ADK", "LGB", "ATL", "ZZZ"]
    loss = criterion(pred_probs, labels, airports)
    print(f"composes with FAOCLoss, no error: loss={loss.item():,.2f}")
    print()


def validate_tft():
    print("=== TFTBaseline ===")
    model = TFTBaseline(num_features=NUM_FEATURES)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    pred_probs, weights = model(x, return_variable_weights=True)
    print(f"output shape: {tuple(pred_probs.shape)} (expect ({BATCH_SIZE},))")
    print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")
    print(f"variable-selection weights shape: {tuple(weights.shape)} (expect ({BATCH_SIZE}, {SEQ_LEN}, {NUM_FEATURES}))")

    row_sums = weights.sum(dim=-1)
    print(f"weights sum to 1 per (batch, timestep): "
          f"{bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))}")

    x2 = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES) * 5
    pred_probs2 = model(x2)
    print(f"different inputs -> different outputs: {not torch.allclose(pred_probs, pred_probs2)}")

    # attention should make the model sensitive to sequence ORDER, unlike
    # a model that only pooled features - shuffle the time axis and check
    # the output actually changes
    shuffled = x[:, torch.randperm(SEQ_LEN), :]
    pred_probs_shuffled = model(shuffled)
    print(f"shuffling the time axis changes the output "
          f"(model is order-sensitive): {not torch.allclose(pred_probs, pred_probs_shuffled)}")

    criterion = FAOCLoss()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    airports = ["ORD", "ADK", "LGB", "ATL", "ZZZ"]
    loss = criterion(pred_probs, labels, airports)
    print(f"composes with FAOCLoss, no error: loss={loss.item():,.2f}")
    print()


def validate_sarima():
    print("=== SARIMABaseline ===")
    print("Synthetic series only (sine wave + noise) - proves the fit()/forecast()")
    print("plumbing works, not that SARIMA has learned anything about real disruptions.")
    rng = np.random.default_rng(0)
    t = np.arange(60)
    synthetic = pd.Series(
        0.5 + 0.3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.05, size=60)
    )

    model = SARIMABaseline(order=(1, 0, 0), seasonal_order=(1, 0, 0, 7))
    model.fit(synthetic)
    forecast = model.forecast(steps=7)
    print(f"forecast length: {len(forecast)} (expect 7)")
    print(f"forecast values finite (no NaN/inf): {bool(np.isfinite(forecast).all())}")

    try:
        SARIMABaseline().forecast(steps=5)
        print("no error on forecast-before-fit (unexpected)")
    except RuntimeError as e:
        print(f"forecast-before-fit correctly raises: {e}")
    print()


if __name__ == "__main__":
    validate_seq2seq()
    validate_tft()
    validate_sarima()
