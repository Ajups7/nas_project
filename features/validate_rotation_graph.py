"""
Validates rotation_graph.py against REAL downloaded BTS data (like
validate_planning_features.py, not synthetic inputs - the data already
exists locally).

Uses 3 real months, restricted to 6 major hub airports for a readable
printed matrix, plus a separate full-network run (no airport restriction)
to report real node/edge counts at scale. See
rotation_graph_validation.md for the resulting numbers and discussion.

Run from the repo root:
    python -m features.validate_rotation_graph
"""

import time

import numpy as np
import pandas as pd

from common.data_loader import load_bts
from features.rotation_graph import build_rotation_adjacency, build_tail_number_chains

SAMPLE_MONTHS = [(2019, 6), (2022, 6), (2023, 6)]
HUB_AIRPORTS = ["ATL", "ORD", "DFW", "LAX", "JFK", "DEN"]


def main():
    print(f"=== Tail-number chain continuity, {SAMPLE_MONTHS[-1]} only ===")
    bts = load_bts(*SAMPLE_MONTHS[-1])
    total_rows = len(bts)
    null_tail = bts["Tail_Number"].isna().sum()
    print(f"total rows: {total_rows}, null Tail_Number: {null_tail} ({null_tail / total_rows:.1%})")

    chains = build_tail_number_chains(bts)
    mismatch_rate = 1 - chains["chain_continuous"].mean()
    print(f"consecutive-leg pairs built: {len(chains)}")
    print(f"chain discontinuity rate (dest != next_origin): {mismatch_rate:.1%}")
    print("sample discontinuous pairs:")
    print(chains[~chains["chain_continuous"]].head(5).to_string(index=False))

    print()
    print(f"=== Adjacency, {len(HUB_AIRPORTS)} hub airports, months {SAMPLE_MONTHS} ===")
    t0 = time.time()
    adjacency, airport_order = build_rotation_adjacency(SAMPLE_MONTHS, airports=HUB_AIRPORTS)
    elapsed = time.time() - t0
    print(f"(computed in {elapsed:.1f}s)")
    print(f"shape: {adjacency.shape}, symmetric: {np.allclose(adjacency, adjacency.T)}")
    print(pd.DataFrame(adjacency, index=airport_order, columns=airport_order).to_string())

    print()
    print("Top-weighted pairs among these hubs:")
    pairs = []
    for i in range(len(airport_order)):
        for j in range(i + 1, len(airport_order)):
            if adjacency[i, j] > 0:
                pairs.append((airport_order[i], airport_order[j], adjacency[i, j]))
    pairs.sort(key=lambda p: -p[2])
    for a, b, w in pairs[:5]:
        print(f"  {a}-{b}: {w}")

    print()
    print(f"=== Full-network adjacency (no airport restriction), months {SAMPLE_MONTHS} ===")
    t0 = time.time()
    full_adjacency, full_order = build_rotation_adjacency(SAMPLE_MONTHS)
    elapsed = time.time() - t0
    print(f"(computed in {elapsed:.1f}s)")
    print(f"nodes: {len(full_order)}, total edge weight: {full_adjacency.sum() // 2}")
    print(f"non-zero edges (unordered pairs): {(np.triu(full_adjacency, k=1) > 0).sum()}")


if __name__ == "__main__":
    main()
