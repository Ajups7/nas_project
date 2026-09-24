"""
Validates models/strategic/rotation_gnn.py two ways:

1. Synthetic mechanism check, same placeholder graph as
   models/planning/regime_transformer_validation.md (a 6-node chain
   0-1-2-3-4-5 plus one extra edge 0-5) for direct comparability - but
   extended with a SECOND test regime_transformer.py's single-shot
   attention couldn't have: since this architecture interleaves graph
   attention with a GRU across seq_len timesteps, a perturbation at an
   EARLY timestep should be able to reach further than 1 hop by the final
   output, unlike a model that only attends over space once.
2. Real data: features/rotation_graph.py's build_rotation_adjacency()
   (Step 1) on real BTS-derived rotation-chain edges among 6 major hubs +
   features/strategic.py's real Strategic-tier features for those same 6
   airports. models/planning/regime_transformer_validation.md says
   explicitly this kind of check "can only be tested once
   features/rotation_graph.py exists" - it now does, so this is that test.

Run from the repo root:
    python -m models.strategic.validate_rotation_gnn
"""

import numpy as np
import pandas as pd
import torch

from features.rotation_graph import build_rotation_adjacency
from features.strategic import build_strategic_features
from features.strategic_label import build_strategic_labels
from losses.faoc_loss import FAOCLoss
from models.strategic.baselines import make_sequence_windows
from models.strategic.rotation_gnn import RotationChainSpatioTemporalGNN

HUB_AIRPORTS = ["ATL", "ORD", "DFW", "LAX", "JFK", "DEN"]
ROTATION_MONTHS = [(2019, 6), (2022, 6), (2023, 6)]  # same sample as validate_rotation_graph.py
DATES = pd.date_range("2023-06-01", "2023-06-25")
YEAR_MONTHS = [(2023, 5), (2023, 6)]
SEQ_LEN = 10


def validate_synthetic_mask_respect():
    print("=== Synthetic mechanism check: does the model respect the graph? ===")
    print("Same placeholder graph as regime_transformer_validation.md, for direct comparability:")
    adjacency = torch.tensor([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
    ], dtype=torch.float32)
    num_nodes, seq_len, num_features = 6, 4, 5

    torch.manual_seed(0)
    model = RotationChainSpatioTemporalGNN(num_features=num_features)
    x = torch.randn(1, num_nodes, seq_len, num_features)

    with torch.no_grad():
        base = model(x, adjacency)

        # Test A: perturb node 3 ONLY at the LAST timestep - isolates a
        # single spatial hop, directly comparable to regime_transformer.py's
        # own perturbation check.
        x_a = x.clone()
        x_a[:, 3, -1, :] += 10.0
        out_a = model(x_a, adjacency)

        # Test B: perturb node 3 at the FIRST timestep instead, leaving 3
        # more GRU-update + spatial-mixing steps to run before the output -
        # the actual "spatio-temporal" claim: can the effect reach further
        # than 1 hop by propagating across time as well as space?
        x_b = x.clone()
        x_b[:, 3, 0, :] += 10.0
        out_b = model(x_b, adjacency)

    changed_a = ~torch.isclose(base, out_a, atol=1e-6)
    print("Test A - perturb node 3 at the LAST timestep only (single spatial hop):")
    for i in range(num_nodes):
        expected = "neighbor/self" if i in (2, 3, 4) else "unaffected"
        print(f"  node {i}: changed={bool(changed_a[0, i])}  (expected: {expected})")
    print(f"  unconnected nodes (0, 1, 5) truly unaffected: {bool(not changed_a[0, [0, 1, 5]].any())}")
    print(f"  node 3 and its real neighbors (2, 4) did change: {bool(changed_a[0, [2, 3, 4]].all())}")

    changed_b = ~torch.isclose(base, out_b, atol=1e-6)
    print()
    print("Test B - perturb node 3 at the FIRST timestep instead (3 more steps to propagate):")
    hop_distance = {3: 0, 2: 1, 4: 1, 1: 2, 5: 2, 0: 3}
    for i in range(num_nodes):
        print(f"  node {i}: changed={bool(changed_b[0, i])}  (graph distance from node 3: {hop_distance[i]})")
    print(f"  reaches further than the single-hop case (node 1 and/or node 5 now changed): "
          f"{bool(changed_b[0, [1, 5]].any())}")
    print()


def validate_real_data():
    print(f"=== Real data: rotation-chain adjacency + Strategic features, {HUB_AIRPORTS} ===")
    adjacency_np, airport_order = build_rotation_adjacency(ROTATION_MONTHS, airports=HUB_AIRPORTS)
    print(f"real rotation-chain adjacency (features/rotation_graph.py, Step 1), airport order: {airport_order}")
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32)

    per_airport_windows, per_airport_labels = [], []
    for airport in airport_order:
        features = build_strategic_features(airport, DATES, YEAR_MONTHS)
        labels = build_strategic_labels(airport, DATES)
        feature_cols = [c for c in features.columns if c not in ("airport", "date")]
        windows = make_sequence_windows(features, feature_cols, seq_len=SEQ_LEN)
        per_airport_windows.append(windows)
        per_airport_labels.append(labels["label"].to_numpy()[SEQ_LEN - 1:])

    x = torch.stack(per_airport_windows, dim=1)  # (num_windows, num_nodes, seq_len, num_features)
    y = torch.tensor(np.stack(per_airport_labels, axis=1), dtype=torch.float32)  # (num_windows, num_nodes)
    num_windows, num_nodes, seq_len, num_features = x.shape
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")

    torch.manual_seed(0)
    model = RotationChainSpatioTemporalGNN(num_features=num_features)
    pred_probs, attn_weights = model(x, adjacency, return_attention_weights=True)
    print(f"pred_probs shape: {tuple(pred_probs.shape)} (expect ({num_windows}, {num_nodes}))")
    print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")
    row_sums = attn_weights.sum(dim=-1)
    print(f"attention weights sum to 1 per (window, node): "
          f"{bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))}")

    print()
    print("Checking no node attends to a real non-neighbor with nonzero weight:")
    zero_edges = (adjacency == 0)
    leaks = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if zero_edges[i, j] and i != j:
                w = attn_weights[:, i, j]
                if not torch.allclose(w, torch.zeros_like(w), atol=1e-6):
                    leaks += 1
                    print(f"  LEAK: {airport_order[i]} attends to non-neighbor {airport_order[j]}, max weight {w.max().item():.4f}")
    print(f"  leaks found: {leaks} (expect 0 - these 6 hubs are fully connected in the real "
          f"data, per rotation_graph_validation.md, so this mainly confirms no non-edges exist to leak into)")

    criterion = FAOCLoss()
    pred_flat = pred_probs.reshape(-1)
    labels_flat = y.reshape(-1)
    airports_flat = airport_order * num_windows  # matches reshape(-1)'s window-major order
    loss = criterion(pred_flat, labels_flat, airports_flat)
    print()
    print(f"composes with FAOCLoss on real labels, no error: loss={loss.item():,.2f}")


if __name__ == "__main__":
    validate_synthetic_mask_respect()
    validate_real_data()
