# Rotation-chain graph: validation results and explanation

Validates `rotation_graph.py` against **real downloaded BTS data**
(`data/bts_ontime/`), the same approach as
`planning_features_validation.md` and for the same reason: the data already
exists locally, and this is the first Strategic-tier code touching it.

## What this is building

`TEAM_PLAN.md` assigns Person B one shared utility: an airport-to-airport
graph built from BTS tail-number sequencing, for the other tiers to import.
It's a real cross-team dependency, not just a backlog item —
`models/planning/regime_transformer.py` (already merged) needs exactly this
shape of adjacency matrix and, lacking one, was validated only against a
synthetic, made-up matrix in the meantime (see that module's docstring).
`build_rotation_adjacency()` below produces the real thing.

Nodes are airports; an edge exists where at least one tracked aircraft flew
directly between two airports during the requested window. Two pieces:

1. `build_tail_number_chains()` — sorts each tail number's flights
   chronologically and checks whether each flight's `Dest` matches the
   *next* flight's `Origin` for that same aircraft (the real value of
   joining by tail number, over just tallying raw Origin/Dest pairs).
2. `build_rotation_adjacency()` — aggregates non-cancelled, non-diverted
   flight legs into a symmetric, weighted `(num_airports, num_airports)`
   count matrix, plus the row/column airport ordering.

## How to reproduce this

```bash
cd nas_project
python -m features.validate_rotation_graph
```

Uses 3 real months (2019-06, 2022-06, 2023-06 — chosen a few years apart to
avoid over-fitting the sanity check to one specific month's schedule) and 6
major hub airports (ATL, ORD, DFW, LAX, JFK, DEN) for a readable printed
matrix, plus one full-network run with no airport restriction.

## Results

### 1. Chain continuity — a small, real minority, not a majority

```
total rows: 577262, null Tail_Number: 3986 (0.7%)
consecutive-leg pairs built: 567846
chain discontinuity rate (dest != next_origin): 2.5%
```

0.7% of June 2023's BTS rows have no `Tail_Number` at all (confirmed `NaN`,
not a blank string) — mostly small/regional carriers not required to
report one. These are dropped before chain-building, not counted as
discontinuities.

2.5% of consecutive same-tail-number leg pairs have `dest != next_origin`.
This is a real, small minority — exactly what a legitimate discontinuity
rate should look like (not e.g. 50%+, which would mean the sort/group-by
was wrong). Spot-checking the sample rows printed by the validation script,
the discontinuities cluster on tail numbers with irregular, sparse
itineraries touching small-charter-style airports (e.g. `PGD`, `USA`,
`SFB`) rather than being scattered evenly across major carriers — consistent
with genuine ferry flights / off-schedule repositioning / occasional tail
number reuse, not a bug in the sort key.

### 2. Hub adjacency — matches real air-travel network structure

```
      ATL   DEN   DFW   JFK   LAX   ORD
ATL     0  3061  3127  1710  2802  3212
DEN  3061     0  2796   739  3447  3297
DFW  3127  2796     0   897  3436  3093
JFK  1710   739   897     0  5393  1347
LAX  2802  3447  3436  5393     0  3642
ORD  3212  3297  3093  1347  3642     0

Top-weighted pairs among these hubs:
  JFK-LAX: 5393
  LAX-ORD: 3642
  DEN-LAX: 3447
  DFW-LAX: 3436
  DEN-ORD: 3297
```

This is a sanity-checkable result, the same way Planning's ATL/July
climatology was: JFK-LAX is the single heaviest edge among these six hubs —
a well-known, high-frequency transcontinental route flown by widebody and
premium narrowbody aircraft with high utilization, so it makes sense it
would also show heavy tail-number-tracked rotation traffic across 3 sampled
months. No pair, including the lightest (JFK-DEN, 739), came out at zero or
negative — the matrix is fully connected among major hubs, as expected. The
matrix printed as `symmetric: True`, confirmed programmatically via
`np.allclose(adjacency, adjacency.T)`, not just by eye.

### 3. Full network — plausible scale

```
nodes: 375, total edge weight: 1741444
non-zero edges (unordered pairs): 3455
```

375 distinct airports appear as an Origin or Dest across the 3 sampled
months — plausible for BTS, which covers all US commercial airports
reporting carriers serve, not just the 82 weather-covered airports in
`schema.md`. 3,455 distinct connected airport pairs out of a possible
375×374/2 ≈ 70,125 (~4.9% density) — consistent with air travel's real
hub-and-spoke structure (most airport pairs never have a direct flight;
connectivity concentrates around a much smaller set of hubs), not a dense,
unstructured graph.

## Design decisions worth flagging

- **Cancelled/diverted flights excluded from edge counts.** A cancelled
  flight never moved an aircraft; a diverted flight's BTS `Dest` is its
  *scheduled* destination, not where it actually landed (BTS doesn't record
  the actual diversion airport). Both would misrepresent "this aircraft was
  actually at both airports" if counted. Verified this actually removes
  rows on real data (June 2023: 12,219 cancelled + 2,239 diverted out of
  577,262 total rows).
- **Adjacency is undirected and unfiltered by count.** A→B and B→A roll
  into one symmetric entry, and no minimum-count threshold is applied —
  `regime_transformer.py`'s `build_attention_mask()` already treats any
  `adjacency > 0` as an edge with no directionality assumption, and a
  future consumer wanting a sparser graph (e.g. Phase 3's actual GNN work)
  can threshold the returned counts however suits that use case, rather
  than have an arbitrary cutoff baked in here.
- **`airports=None` pulls in all 375 network airports, not just the 82
  weather-covered ones.** Left as an explicit caller choice (via the
  `airports` allowlist parameter) rather than defaulting to the 82-airport
  set, since restricting silently would hide real network structure (e.g.
  a hub's connectivity to smaller non-weather-covered spokes) without the
  caller asking for it.

## Practical cost, measured (not estimated)

Building the 6-hub adjacency across 3 months took **26.0s**; the
unrestricted full-network version across the same 3 months took **32.8s**
— both dominated by `load_bts()`'s CSV parsing (3 months × one ~500K-row
CSV each), not by the aggregation itself. Scaling to the full 121-month
history would be on the order of **20 minutes** for a full-network
adjacency — a one-time batch job, not something to recompute per training
run, the same recommendation `planning_features_validation.md` made for
disruption climatology.

## What this validation does *not* cover

- Only 3 months (spread across different years) and, for the readable
  printed matrix, 6 airports were checked in detail. The full 121-month
  history hasn't been run, and no cached full-history adjacency exists yet
  — a reasonable next step once this is needed for actual model training,
  not before (same "cache it, don't recompute per training loop"
  recommendation as `planning_features_validation.md`).
- No claim that a 2.5% discontinuity rate is "acceptable" for any
  particular downstream use — only that it's small enough to look like
  genuine, expected air-travel irregularity rather than a bug in this
  module's sorting/grouping logic.
- Doesn't yet build the delay-propagation-along-a-chain feature that
  `build_tail_number_chains()`'s output would enable (mentioned in
  `rotation_graph.py`'s module docstring as a plausible next step) — only
  the chain table and the aggregated adjacency matrix are delivered here.
