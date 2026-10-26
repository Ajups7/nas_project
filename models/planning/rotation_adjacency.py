"""
Rotation-chain adjacency for the Regime Transformer, from Person B's
features/rotation_graph.py over the 75 Planning airports.

Source: all 12 months of 2019 BTS (normal pre-COVID operations, inside the
train period - no leakage). Among these airports ~48% of pairs have direct
flights (median airport: 37 partners), so "any flights" barely restricts
attention. Variants:
- "top10": each airport linked to its 10 strongest partners by flight
  count (edge if either side has the other in its top 10)
- "full":  any flights between the pair
- "none":  no edges (self-loops only, added by the model)

Build once (~3.5 min):
    python -m models.planning.rotation_adjacency
"""

import numpy as np

from common.data_loader import DATA_DIR
from features.planning import eligible_airports
from features.rotation_graph import build_rotation_adjacency

CACHE_PATH = DATA_DIR / "cache" / "rotation_adjacency_2019.npz"
TOP_K = 10


def load_counts() -> tuple[np.ndarray, list[str]]:
    z = np.load(CACHE_PATH, allow_pickle=False)
    return z["counts"], list(z["airports"])


def adjacency(variant: str) -> np.ndarray:
    counts, _ = load_counts()
    counts = counts.copy()
    np.fill_diagonal(counts, 0)
    if variant == "full":
        return (counts > 0).astype(np.float32)
    if variant == "none":
        return np.zeros_like(counts, dtype=np.float32)
    if variant == "top10":
        top = np.zeros_like(counts, dtype=bool)
        for i in range(len(counts)):
            nz = np.flatnonzero(counts[i])
            keep = nz[np.argsort(counts[i, nz])[::-1][:TOP_K]]
            top[i, keep] = True
        return (top | top.T).astype(np.float32)
    raise ValueError(f"unknown adjacency variant {variant!r}")


if __name__ == "__main__":
    airports = eligible_airports()
    counts, order = build_rotation_adjacency([(2019, m) for m in range(1, 13)], airports)
    assert order == airports
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, counts=counts, airports=np.array(order))
    for v in ("top10", "full"):
        a = adjacency(v)
        deg = a.sum(1)
        print(f"{v}: edges {int(a.sum() / 2)}, neighbors per airport min {int(deg.min())} "
              f"median {int(np.median(deg))} max {int(deg.max())}")
    top = adjacency("top10")
    for ap in ("EWR", "ATL", "DEN", "SEA"):
        i = order.index(ap)
        print(f"  {ap} top-10 neighbors: {[order[j] for j in np.flatnonzero(top[i])]}")
