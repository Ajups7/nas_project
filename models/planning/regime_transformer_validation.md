# Regime Transformer stub: validation results and explanation

Same philosophy as every prior validation: synthetic inputs, no training,
no real data — with one extra caveat specific to this step, stated up
front and repeated below because it matters: **the graph structure used
here is also synthetic, not Person B's real rotation-chain graph, which
doesn't exist in this repo yet.**

## What Step 8 is, and the dependency it's blocked on

This is the project's actual Phase 3 novel architecture for the Planning
tier (`PLAN.md` item 6) — distinct from Step 2's MoE regime routing (item
7); the two are separate contributions, not one built on the other.

`PLAN.md` describes it as representing *"disruption propagation over
scheduled aircraft rotations rather than geographic proximity alone."*
That means it needs a real graph of which airports are connected by
aircraft rotation chains (built from BTS tail-number sequencing) — and
that graph is explicitly **Person B's** shared utility
(`features/rotation_graph.py`, per `TEAM_PLAN.md`), confirmed **not to
exist anywhere in this repo** (checked directly before writing this).

Rather than block Step 8 entirely on that, `regime_transformer.py` takes
`adjacency` as a plain input argument — any `(num_airports, num_airports)`
matrix. Person B's real module, whenever it exists, only needs to produce
that shape; nothing else here needs to change. **Everything validated below
uses a made-up placeholder graph** (a simple 6-node chain, `0-1-2-3-4-5`,
plus one extra edge `0-5`), not real rotation-chain structure.

## Two attention mechanisms, two different jobs

1. **Graph attention** — each airport's time-series encoding attends only
   to airports it has a rotation-chain edge to (plus itself). This is the
   "propagation over rotations, not geography" part.
2. **Regime attention** — each airport's post-graph representation attends
   over 4 learned regime embeddings (`normal`/`gdp`/`ground_stop`/`mit` —
   the same 4 regimes as Step 2's `MoERegimeRouter`, reused directly rather
   than re-invented under new names).

**Naming note**: `PLAN.md` calls this "sector-graph-attention," but the
actual edges are airport-to-airport rotation-chain edges, not literal ATC
airspace sectors — kept as the paper's own name for the architecture rather
than silently renamed.

## How to reproduce this

```bash
cd nas_project
python -m models.planning.validate_regime_transformer
```

## Results

### 1. Shapes and range

```
pred_probs shape: (2, 6) (expect (2, 6))
in (0, 1): True
regime attention weights sum to 1 per node: True
```

One probability per airport per example, all valid. Regime attention
weights are a genuine softmax distribution over the 4 regimes for every
node — same kind of check as the MoE router's gate weights in Step 2.

### 2. The check that actually matters: does graph attention respect the graph?

This is the one worth dwelling on, because it's the difference between "a
graph attention layer" and "an attention layer that happens to have a mask
argument nobody's checked works." The synthetic placeholder graph:

```
adjacency (chain 0-1-2-3-4-5 + edge 0-5):
[[0,1,0,0,0,1], [1,0,1,0,0,0], [0,1,0,1,0,0],
 [0,0,1,0,1,0], [0,0,0,1,0,1], [1,0,0,0,1,0]]
```

Node 3's only real neighbors are nodes 2 and 4. The test perturbed **only**
node 3's input features (added a large constant) and checked which nodes'
outputs moved:

```
node 0: changed=False  (expected: unaffected)
node 1: changed=False  (expected: unaffected)
node 2: changed=True   (expected: neighbor/self)
node 3: changed=True   (expected: neighbor/self)
node 4: changed=True   (expected: neighbor/self)
node 5: changed=False  (expected: unaffected)

Unconnected nodes (0, 1, 5) truly unaffected: True
Node 3 and its real neighbors (2, 4) did change: True
```

Exactly the nodes with a real edge to node 3 changed; every node without
one didn't move at all — not "moved a little," genuinely unchanged, down to
floating-point precision. That confirms the attention mask is actually
zeroing out disallowed connections in the softmax, not just being passed
in and ignored. If node 0 or node 5 had shifted here, that would mean
information was leaking across the graph regardless of the adjacency
matrix — i.e., the "graph" part of "graph attention" wouldn't actually be
doing anything.

## What this validation does *not* cover

- **The graph itself is fake.** This proves the attention mechanism
  correctly respects *whatever* adjacency matrix it's given — it says
  nothing about whether real rotation-chain edges (once Person B's module
  exists) would produce sensible predictions. That can only be tested once
  `features/rotation_graph.py` exists and the interface documented in
  `regime_transformer.py` gets real data instead of a synthetic chain.
- No real feature data, no training — same caveats as every architecture
  validated so far in this project.
- The regime embeddings are randomly initialized and, same as Step 2's MoE
  gate, don't yet correspond to anything real — this only confirms the
  mechanism is wired correctly, not that it's learned to recognize an
  actual GDP-like pattern.
