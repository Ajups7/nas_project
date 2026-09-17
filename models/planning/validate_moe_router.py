"""
Reproduces the checks in moe_router_validation.md. Not a unit test suite
(no assertions) - a diagnostic script confirming MoERegimeRouter is wired
correctly (shapes, softmax normalization, output range) and that it
composes cleanly with FAOCLoss, since the two are meant to be used
together. The router is freshly initialized and untrained, so this proves
the mechanism works - not that it has learned anything.

Run from the repo root:
    python -m models.planning.validate_moe_router
"""

import torch

from losses.faoc_loss import FAOCLoss
from models.planning.moe_router import REGIME_NAMES, MoERegimeRouter

torch.manual_seed(0)

INPUT_DIM = 6
BATCH_SIZE = 5


def main():
    router = MoERegimeRouter(input_dim=INPUT_DIM)
    features = torch.randn(BATCH_SIZE, INPUT_DIM)

    print("=== Shape and range checks ===")
    pred_probs, gate_weights = router(features, return_gate_weights=True)
    print(f"pred_probs shape: {tuple(pred_probs.shape)} (expect ({BATCH_SIZE},))")
    print(f"gate_weights shape: {tuple(gate_weights.shape)} (expect ({BATCH_SIZE}, {len(REGIME_NAMES)}))")
    print(f"pred_probs in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")

    print()
    print("=== Gate weights sum to 1 per sample (softmax property) ===")
    row_sums = gate_weights.sum(dim=-1)
    for i, s in enumerate(row_sums.tolist()):
        print(f"  sample {i}: sum={s:.6f}")

    print()
    print("=== Gate weights by regime, per sample ===")
    header = "sample  " + "  ".join(f"{name:>12s}" for name in REGIME_NAMES)
    print(header)
    for i, row in enumerate(gate_weights.tolist()):
        cells = "  ".join(f"{v:12.4f}" for v in row)
        print(f"{i:6d}  {cells}")

    print()
    print("=== Different inputs actually produce different routing ===")
    calm = torch.zeros(1, INPUT_DIM)
    stormy = torch.full((1, INPUT_DIM), 5.0)
    _, gate_calm = router(calm, return_gate_weights=True)
    _, gate_stormy = router(stormy, return_gate_weights=True)
    print(f"gate for all-zero input:   {gate_calm.tolist()[0]}")
    print(f"gate for all-5.0 input:    {gate_stormy.tolist()[0]}")
    print(f"identical routing for both inputs: {torch.allclose(gate_calm, gate_stormy)} (expect False)")

    print()
    print("=== Integration check: MoERegimeRouter output -> FAOCLoss ===")
    criterion = FAOCLoss()
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    airports = ["ORD", "ADK", "LGB", "ATL", "ZZZ"]
    loss = criterion(pred_probs, labels, airports)
    print(f"pred_probs: {[round(p, 4) for p in pred_probs.tolist()]}")
    print(f"labels:     {labels.tolist()}")
    print(f"airports:   {airports}")
    print(f"FAOCLoss on router output: {loss.item():,.2f}")
    print("(no error raised -> the two shared utilities compose end to end)")


if __name__ == "__main__":
    main()
