# Team Work Split (3 people)

Split along the paper's own three-tier hierarchical framing (Tactical /
Strategic / Planning) rather than by phase — each person owns a full,
independent vertical end-to-end, so day-to-day work stays in separate
directories and merge conflicts stay rare. Each person also owns one small,
well-defined shared utility that the other two import, instead of a phase
being "owned" by nobody and colliding.

## Before splitting off: one shared decision, made together

All three of you should agree on the **feature table schema** up front —
column names, join keys (airport code, date, hour), and file format for
the unified per-airport-hour table that BTS + ASPM Airport Analysis + EDCT
+ weather get joined into. This is the one thing that truly needs
synchronous agreement, because all three tiers read from it. Everything
else below can proceed independently once this is settled.

Suggested join keys: `(airport, date, local_hour, gmt_hour)` — matches the
grouping already used across all four downloaded datasets.

## Person A — Tactical tier (0-2h) + Baseline Models

- Feature engineering at hourly granularity for the tactical horizon
  (recent conditions: current weather, current capacity/demand, EDCT
  activity in the last few hours).
- Tactical baseline model using a standard loss (weighted cross-entropy),
  benchmarked against seq2seq / TFT / SARIMA (Phase 2).
- Later: tactical-tier model upgrades in Phase 3.
- Owns: `models/tactical/`, `features/tactical.py`

## Person B — Strategic tier (2-24h) + Rotation-Chain Graph (shared utility)

- Feature engineering at day-ahead granularity for the strategic horizon.
- **Shared utility**: build the rotation-chain graph from BTS tail-number
  sequencing (aircraft rotation edges between airports) — exposed as a
  reusable module the others can import if their tier needs network
  structure too.
- Strategic baseline model, then the spatio-temporal GNN work in Phase 3
  (rotation-chain edges fit naturally at this horizon, where next-day
  network effects matter most).
- Owns: `models/strategic/`, `features/rotation_graph.py` (shared)

## Person C — Planning tier (1-30d) + FAOC-Loss & Validation Framework (shared)

- Feature engineering at longer lookback / seasonal granularity for the
  planning horizon.
- **Shared utility**: implement FAOC-Loss (asymmetric, cost-sensitive loss),
  calibrated using the FAA/A4A cost-per-flight-hour figures already sourced
  (see PLAN.md) — exposed as a shared loss module all three tiers plug into
  once their baselines work.
- Planning baseline model, then the sector-graph-attention Regime
  Transformer in Phase 3.
- Also coordinates the Phase 4 validation framework (KPI suite, temporal
  blocked CV with regime stratification, adversarial stress tests) — since
  evaluating across all three tiers naturally needs one person pulling
  everyone's results together.
- Owns: `models/planning/`, `losses/faoc_loss.py` (shared),
  `validation/`

## Workflow

- Each person works on their own branch, in their own tier's directory.
- Shared utilities (`rotation_graph.py`, `faoc_loss.py`, the feature schema)
  change rarely once agreed — treat changes to them as a heads-up to the
  other two before merging.
- Regular short syncs (even async, e.g. a shared doc or channel) to confirm
  the shared feature schema hasn't silently drifted between tiers.
- PRs reviewed by at least one other teammate before merging to main.
