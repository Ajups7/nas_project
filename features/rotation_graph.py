"""
Person B (Strategic tier) - shared utility (see TEAM_PLAN.md): the
rotation-chain graph, built from BTS tail-number sequencing. Exposed for
the other two tiers to import, the same way common/temporal_split.py and
losses/faoc_loss.py are shared.

CROSS-TEAM DEPENDENCY this unblocks: models/planning/regime_transformer.py
(already merged) needs "a graph of rotation-chain edges between airports"
and, lacking one, takes a plain `adjacency` tensor argument, validated only
against a synthetic, made-up matrix in the meantime (see that module's
docstring). build_rotation_adjacency() below produces the real thing in
exactly the shape it expects: (num_airports, num_airports), nonzero where
an edge exists.

What "rotation-chain graph" means here, precisely - nodes are airports, and
an edge exists where at least one tracked aircraft flew directly between
two airports during the requested window:

1. Sequencing/continuity check (the actual value of joining by tail number,
   as opposed to just tallying raw Origin/Dest pairs): sort each tail
   number's flights chronologically (FlightDate, then CRSDepTime) and check
   whether each flight's Dest matches the *next* flight's Origin for that
   same aircraft. A mismatch is a real data-quality signal - an unrecorded
   leg, a tail number reused across different physical aircraft, or a
   genuine base change - not a bug in this module. See
   rotation_graph_validation.md for the measured mismatch rate.
2. Edge weights: counts, per unordered airport pair, aggregated across
   every non-cancelled, non-diverted flight leg in the requested months.
   A cancelled flight never moved an aircraft anywhere; a diverted flight's
   BTS `Dest` is its *scheduled* destination, not where it actually landed
   (BTS doesn't record the actual diversion airport) - both excluded, since
   an edge here specifically means "this aircraft was actually at both
   airports."
3. Undirected, weighted, unfiltered: A->B and B->A roll into one symmetric
   (A, B) entry, and low-count edges are NOT thresholded away here -
   regime_transformer.py's build_attention_mask() already treats any
   adjacency > 0 as a valid edge with no directionality or self-loop
   handling expected from this module (self-loops are added on the
   consumer side); a caller wanting a sparser graph can threshold the
   returned counts by whatever cutoff suits their use case.

Naming/scope note (mirrors regime_transformer.py's own "naming note" about
"sector-graph-attention" not being literal ATC sectors): the adjacency
matrix is a same-aircraft-connectivity graph, validated by the tail-number
chain-continuity check, not a literal moment-by-moment rotation sequence.
That finer-grained sequence (build_tail_number_chains()'s output) is kept
as a first-class result in its own right, both because it's the mechanism
that makes the continuity check possible and because it's a natural
building block for later delay-propagation-along-a-chain features (a
plausible next Strategic-tier feature - see losses/delay_propagation_multipliers.csv,
already in the repo, which such a feature would eventually join against;
not built here).
"""

import numpy as np
import pandas as pd

from common.data_loader import load_bts


def build_tail_number_chains(bts: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive (leg_i, leg_i+1) pair flown by the same
    tail number, sorted by FlightDate then CRSDepTime. Columns:
    tail_number, origin, dest, next_origin, next_dest, chain_continuous
    (dest == next_origin).

    Rows with a missing Tail_Number are dropped first (checked against real
    2023-06 BTS data: ~0.7% of rows, all NaN rather than a blank string -
    mostly small/regional carriers not required to report a tail number).
    A tail number appearing only once in `bts` contributes no rows here (no
    "next" leg to pair it with), which is expected, not a bug.
    """
    df = bts.dropna(subset=["Tail_Number"]).copy()
    df = df.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])

    grouped = df.groupby("Tail_Number", sort=False)
    next_origin = grouped["Origin"].shift(-1)
    next_dest = grouped["Dest"].shift(-1)

    chains = pd.DataFrame({
        "tail_number": df["Tail_Number"],
        "origin": df["Origin"],
        "dest": df["Dest"],
        "next_origin": next_origin,
        "next_dest": next_dest,
    })
    chains = chains.dropna(subset=["next_origin"])
    chains["chain_continuous"] = chains["dest"] == chains["next_origin"]
    return chains.reset_index(drop=True)


def build_rotation_adjacency(
    months: list[tuple[int, int]],
    airports: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Loads BTS for each (year, month) in `months` via
    common.data_loader.load_bts, concatenates, drops Cancelled==1 and
    Diverted==1 rows (see module docstring for why), and aggregates into a
    symmetric (num_airports, num_airports) count matrix - entry (i, j) is
    the number of valid flight legs between airport_order[i] and
    airport_order[j], in either direction, summed across all requested
    months.

    airports: optional allowlist restricting the node set (e.g. the 82
    weather-covered airports from schema.md, for cross-tier consistency).
    None = every airport seen in Origin/Dest across the requested months.
    Rows with either endpoint outside this allowlist are dropped before
    aggregation, not silently folded into a bucket.

    Returns (adjacency, airport_order): airport_order is the row/column
    label list a caller needs to know which row is which airport code
    (e.g. models/planning/regime_transformer.py's `adjacency` argument).
    """
    frames = [load_bts(year, month) for year, month in months]
    bts = pd.concat(frames, ignore_index=True)
    bts = bts[(bts["Cancelled"] != 1) & (bts["Diverted"] != 1)]

    if airports is not None:
        allowed = set(airports)
        bts = bts[bts["Origin"].isin(allowed) & bts["Dest"].isin(allowed)]
        airport_order = sorted(allowed)
    else:
        airport_order = sorted(set(bts["Origin"]) | set(bts["Dest"]))

    index = {code: i for i, code in enumerate(airport_order)}
    n = len(airport_order)
    adjacency = np.zeros((n, n), dtype=np.int64)

    pair_counts = (
        bts.groupby(["Origin", "Dest"]).size().reset_index(name="count")
    )
    for origin, dest, count in pair_counts.itertuples(index=False):
        i, j = index[origin], index[dest]
        adjacency[i, j] += count
        adjacency[j, i] += count if i != j else 0

    return adjacency, airport_order
