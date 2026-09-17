"""
Validates regime_transformer.py with synthetic inputs AND a synthetic,
made-up adjacency matrix - NOT Person B's real rotation-chain graph, which
doesn't exist yet (see regime_transformer.py's module docstring).

The check that matters most here isn't shape/range - it's whether "graph
attention" is doing anything structurally different from ordinary
attention that just ignores its mask. Proven by perturbing one node's
input and checking WHICH other nodes' outputs change: only the perturbed
node itself and its actual graph neighbors should move; an unconnected
node's output should be untouched, down to floating-point precision.

Run from the repo root:
    python -m models.planning.validate_regime_transformer
"""

import torch

from models.planning.regime_transformer import SectorGraphAttentionRegimeTransformer

torch.manual_seed(0)

BATCH_SIZE = 2
NUM_NODES = 6
SEQ_LEN = 10
NUM_FEATURES = 6

# Synthetic placeholder graph (NOT real rotation-chain data): a simple
# chain 0-1-2-3-4-5, plus one extra edge 0-5 to make it non-trivial.
# Node indices are arbitrary stand-ins for airports here.
ADJACENCY = torch.zeros(NUM_NODES, NUM_NODES)
for i in range(NUM_NODES - 1):
    ADJACENCY[i, i + 1] = 1
    ADJACENCY[i + 1, i] = 1
ADJACENCY[0, 5] = 1
ADJACENCY[5, 0] = 1


def main():
    model = SectorGraphAttentionRegimeTransformer(num_features=NUM_FEATURES)
    x = torch.randn(BATCH_SIZE, NUM_NODES, SEQ_LEN, NUM_FEATURES)

    print("=== Shape and range checks ===")
    pred_probs, graph_attn, regime_attn = model(x, ADJACENCY, return_attention_weights=True)
    print(f"pred_probs shape: {tuple(pred_probs.shape)} (expect ({BATCH_SIZE}, {NUM_NODES}))")
    print(f"in (0, 1): {bool(((pred_probs > 0) & (pred_probs < 1)).all())}")
    print(f"regime attention weights sum to 1 per node: "
          f"{bool(torch.allclose(regime_attn.sum(dim=-1), torch.ones(BATCH_SIZE, NUM_NODES), atol=1e-5))}")

    print()
    print("=== Does graph attention respect adjacency, or is the mask being ignored? ===")
    print(f"adjacency (synthetic placeholder, chain 0-1-2-3-4-5 + edge 0-5):")
    print(ADJACENCY.int().tolist())

    x_perturbed = x.clone()
    x_perturbed[:, 3, :, :] += 10.0  # large perturbation to node 3 only

    probs_before = model(x, ADJACENCY)
    probs_after = model(x_perturbed, ADJACENCY)
    changed = ~torch.isclose(probs_before, probs_after, atol=1e-6)

    print()
    print("Perturbing node 3's input only. Node 3's graph-chain neighbors are {2, 4}.")
    print("Expected to change: node 3 itself, and nodes 2 and 4 (direct neighbors).")
    print("Expected UNCHANGED: nodes 0, 1, 5 (no edge to node 3).")
    for node in range(NUM_NODES):
        is_neighbor = bool(ADJACENCY[3, node]) or node == 3
        print(f"  node {node}: changed={bool(changed[0, node].item())}  "
              f"{'(expected: neighbor/self)' if is_neighbor else '(expected: unaffected)'}")

    correctly_isolated = not changed[0, 0] and not changed[0, 1] and not changed[0, 5]
    correctly_propagated = changed[0, 2] and changed[0, 3] and changed[0, 4]
    print()
    print(f"Unconnected nodes (0, 1, 5) truly unaffected: {bool(correctly_isolated)}")
    print(f"Node 3 and its real neighbors (2, 4) did change: {bool(correctly_propagated)}")
    print(f"CONFIRMS graph attention structurally respects the adjacency mask "
          f"(not disguised dense attention): {bool(correctly_isolated and correctly_propagated)}")


if __name__ == "__main__":
    main()
