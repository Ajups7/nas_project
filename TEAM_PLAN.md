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

## Temporal split — PROPOSED, needs Person A + Person B sign-off

The second shared Phase 0 decision: fixed train/validation/test date
boundaries, used identically by all three tiers (so results are comparable
across tiers later, in the Phase 4 validation framework). **Not yet
agreed** — the below is Person C's concrete proposal, implemented as a
reusable function so it can be checked mechanically rather than agreed on
by eyeballing a calendar: [`common/temporal_split.py`](common/temporal_split.py)'s
`assign_temporal_split()`, validated in
[`common/validate_temporal_split.py`](common/validate_temporal_split.py).

**Why this needs an embargo gap, not just three adjacent date ranges**: the
Planning tier's label looks forward up to 30 days from any date. Without a
gap between splits, a training example near the boundary would have its
label computed from data that's actually inside the validation set — a real
leak. The embargo is a dead zone between splits, sized to at least the
longest horizon of any tier (**30 days — Planning's own**, which makes it
the binding constraint for all three of you, even though Tactical/Strategic
have much shorter horizons).

**Lesson already learned building this**: an earlier draft of this proposal
picked embargo boundaries by hand, rounding to "the rest of the calendar
month." That was actually wrong — checked mechanically, a validation date of
2024-01-31 has a 30-day label window reaching to 2024-03-01, but the
hand-rounded proposal started the test set on exactly that date — a 1-day
leak, caused by February 2024 being a leap year (29 days: one short of the
required 30). `assign_temporal_split()` computes the exact day-based gap
instead of rounding to calendar months, so this class of mistake can't
happen again.

**Concrete proposed boundaries** (`assign_temporal_split(dates, train_end="2022-12-31", validation_end="2024-01-30", embargo_days=30)`):

| Bucket | Range | Days |
|---|---|---|
| train | 2016-01-01 → 2022-12-31 | 2,557 (~7 yr) |
| embargo | 2023-01-01 → 2023-01-30 | 30 |
| validation | 2023-01-31 → 2024-01-30 | 365 (~1 yr) |
| embargo | 2024-01-31 → 2024-02-29 | 30 |
| test | 2024-03-01 → 2026-01-31 | 702 (~1.9 yr) |

Test is deliberately ~2 years, not a short holdout: Planning-tier features
are climatology-based (seasonal patterns — see `features/planning.py`), so
a short test window would only ever validate against one season.

Known data gaps (2016-07 and 2025-09 EDCT — see README.md) fall inside
train and test respectively; neither lands on a boundary, so they don't
interact with the embargo logic.

**Next step**: Person A and Person B review the function + boundaries above
and confirm, adjust `train_end`/`validation_end`, or raise concerns — once
agreed, this becomes locked shared infrastructure like `common/data_loader.py`.

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
- Shared utilities (`rotation_graph.py`, `faoc_loss.py`, `temporal_split.py`,
  the feature schema) change rarely once agreed — treat changes to them as a
  heads-up to the other two before merging.
- Regular short syncs (even async, e.g. a shared doc or channel) to confirm
  the shared feature schema hasn't silently drifted between tiers.
- PRs reviewed by at least one other teammate before merging to main.
