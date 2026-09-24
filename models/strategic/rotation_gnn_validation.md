# Spatio-temporal GNN: validation results and explanation

Validates `models/strategic/rotation_gnn.py` two ways: the same synthetic
mechanism check every architecture in this project starts with, **plus a
real-data check `models/planning/regime_transformer_validation.md`
explicitly says isn't possible yet** — because this tier's shared
rotation-chain graph (Step 1) now exists.

## What Step 5 is, and why it's not blocked

This is the Strategic tier's Phase 3 novel architecture (`PLAN.md` item
6), analogous to Person C's `SectorGraphAttentionRegimeTransformer` for
Planning. `TEAM_PLAN.md` frames it as: "Strategic baseline model, then the
spatio-temporal GNN work in Phase 3 (rotation-chain edges fit naturally at
this horizon, where next-day network effects matter most)."

Unlike `regime_transformer.py`, which had to validate against a synthetic,
made-up adjacency matrix because `features/rotation_graph.py` didn't exist
yet, this module's real dependency (Step 1, already built and validated
against real BTS data) is available now. That earlier validation doc says
outright: *"this can only be tested [with a real graph] once
`features/rotation_graph.py` exists."* It does now — the second half of
this validation is exactly that test.

**Design difference from `regime_transformer.py`, on purpose**:
`regime_transformer.py` fully encodes each airport's time series first (an
LSTM run to completion), then does ONE spatial attention pass afterward —
space and time are two separate, sequential stages. `rotation_gnn.py`
interleaves them: at every one of the `seq_len` daily timesteps, each
node's hidden state is updated from that day's features via a GRU cell,
*then* mixed with its rotation-chain neighbors via masked graph attention,
and that mixed state feeds the *next* timestep's GRU update. This is what
makes it genuinely "spatio-temporal" rather than "temporal-then-spatial" —
a disruption signal can propagate outward across the graph *and* forward
in time within the same pass, which is the literal "network effects"
`TEAM_PLAN.md` calls out as the reason rotation-chain edges matter at this
horizon.

## How to reproduce this

```bash
cd nas_project
python -m models.strategic.validate_rotation_gnn
```

## Results

### 1. Synthetic mechanism check — same graph as `regime_transformer_validation.md`, extended

Same placeholder 6-node graph (chain `0-1-2-3-4-5` plus edge `0-5`, forming
a cycle), for direct comparability.

**Test A — single-hop (directly comparable to `regime_transformer.py`'s own check)**: perturb node 3's features only at the *last* timestep, so
there's exactly one spatial-mixing step left before the output.

```
node 0: changed=False  (expected: unaffected)
node 1: changed=False  (expected: unaffected)
node 2: changed=True   (expected: neighbor/self)
node 3: changed=True   (expected: neighbor/self)
node 4: changed=True   (expected: neighbor/self)
node 5: changed=False  (expected: unaffected)
unconnected nodes (0, 1, 5) truly unaffected: True
node 3 and its real neighbors (2, 4) did change: True
```

Identical result pattern to `regime_transformer_validation.md`'s own
single-hop check: exactly the nodes with a real edge to node 3 moved;
everything else was genuinely unchanged (not "moved a little" — bitwise
unchanged within floating-point tolerance). Confirms the attention mask is
actually zeroing out disallowed connections, not just present and ignored.

**Test B — the actual spatio-temporal claim**: perturb node 3's features
only at the *first* timestep instead, leaving 3 more GRU-update +
spatial-mixing steps to run before the output.

```
node 0: changed=True  (graph distance from node 3: 3)
node 1: changed=True  (graph distance from node 3: 2)
node 2: changed=True  (graph distance from node 3: 1)
node 3: changed=True  (graph distance from node 3: 0)
node 4: changed=True  (graph distance from node 3: 1)
node 5: changed=True  (graph distance from node 3: 2)
reaches further than the single-hop case (node 1 and/or node 5 now changed): True
```

Every node changed, including nodes 2-3 hops away from node 3 in the
graph — something a single-shot spatial attention pass (like
`regime_transformer.py`'s) structurally cannot do, since it only ever
looks one hop out from each node's direct neighbors. This is the concrete
evidence for the module docstring's claim: interleaving graph attention
with a recurrent temporal update lets a signal propagate further than one
hop, across timesteps, not just across space in one shot.

### 2. Real data — the check `regime_transformer_validation.md` said wasn't possible yet

Real rotation-chain adjacency (Step 1, `build_rotation_adjacency`) on 6
major hubs, and real Strategic-tier features (Step 2) for those same 6
airports, June 2023:

```
real rotation-chain adjacency, airport order: ['ATL', 'DEN', 'DFW', 'JFK', 'LAX', 'ORD']
x shape: (16, 6, 10, 24), y shape: (16, 6)
pred_probs shape: (16, 6) (expect (16, 6))
in (0, 1): True
attention weights sum to 1 per (window, node): True

Checking no node attends to a real non-neighbor with nonzero weight:
  leaks found: 0

composes with FAOCLoss on real labels, no error: loss=14,535.58
```

16 real sliding windows × 6 real airports, correctly shaped output, valid
probabilities, and a real softmax attention distribution over neighbors
per node. The "no leaks" check trivially passes here because these 6 hubs
happen to be fully connected in the real data (confirmed in
`rotation_graph_validation.md` — every pair of these 6 hubs has a nonzero
rotation-chain edge weight), so it doesn't independently prove masking
correctness the way Test A above does — it mainly confirms there's no
accidental attention leak into a genuinely nonexistent edge, which there
isn't because none exist among this particular airport set. Test A (the
synthetic sparse graph) remains the real proof that masking works;
this real-data run's contribution is proving the *real* graph and *real*
features actually plug into the architecture end-to-end without shape
mismatches, NaNs, or silent breakage — composing cleanly with `FAOCLoss`
on real labels, same as `models/strategic/baselines_validation.md`
established for the baseline architectures.

## Design decisions worth flagging

- **Reused `build_attention_mask` from `models/planning/regime_transformer.py`**
  rather than reimplementing it — it's a pure, tier-agnostic function
  (adjacency in, additive mask out), same reuse-over-duplication reasoning
  as `models/strategic/baselines.py`'s import of the shared baseline
  architectures. Cross-tier dependency, same as that file's.
- **Graph attention, not a literal GCN** — same choice `regime_transformer.py`
  made, for the same reason: masked softmax attention handles "restrict to
  neighbors" without needing separate degree-normalization a raw
  adjacency-weighted convolution would require.
- **The 6-hub real-data test can't independently prove masking correctness**
  because these particular hubs are fully connected in the real graph —
  flagged explicitly above rather than presented as if it were an
  additional masking proof. A future test on a wider, sparser airport set
  would be a better candidate for that specific check on real data.

## What this validation does *not* cover

- No training, anywhere — every check is mechanical (shapes, ranges, mask
  behavior, propagation reach), gated on the team-agreed temporal split
  (`TEAM_PLAN.md`) before any real training run, same caveat as every
  prior architecture in this project.
- Only 6 major, densely-connected hub airports were used for the real-data
  check — doesn't exercise a sparser, more realistic full-network graph
  (375 airports, per `rotation_graph_validation.md`), which would be a
  heavier but more representative test before any actual training.
- The GRU and attention weights are randomly initialized — this confirms
  the mechanism is wired correctly, not that it has learned anything about
  real disruption propagation.
