# Shadow-mode deployment: validation results and explanation

This is the **final step** of the Planning-tier build sequence, and the
only validation in this whole project that wires every prior piece
together in one run: Step 3's features → Step 7's model → this step's
`ShadowModeLog` → Step 4's real labels → Step 9's KPI suite. Every earlier
validation tested one piece in isolation; this one proves they actually
connect.

## What "shadow mode" means here, honestly

The real meaning — run a candidate model alongside a live production
system, without letting its predictions affect anything, to build
confidence before cutover — isn't fully realizable yet: there's no live
ops system for this project to shadow, no real-time data feed, and no
trained model (same caveat as every prior step). What's built instead is a
**replay-based simulator**: walk real historical dates in order, record a
prediction, and only reveal the real outcome afterward — mirroring the
temporal flow a live deployment would follow, even though a replay
technically already has the "future" sitting in the historical data being
loaded.

**The one property that actually matters is enforced by construction, not
convention**: `ShadowModeLog` has no method that could feed a prediction
back into a decision. Predictions are recorded once and never mutated;
outcomes live in a separate dict that can only grow. There's no code path
anywhere for a prediction to *act* on anything.

## How to reproduce this

```bash
cd nas_project
python -m validation.validate_shadow_mode
```

## Results

### 1. The "no acting on predictions" contract — checked, not just claimed

```
public methods found: ['record_prediction', 'resolve_outcome', 'resolved_pairs', 'unresolved_count']
matches the declared safe surface exactly: True
```

Introspected the class's actual public API and confirmed it's limited to
exactly the 4 methods the module docstring claims — record, resolve, read
resolved, count unresolved. Nothing else exists. This makes "shadow mode
can't act on anything" a falsifiable, checkable fact about the code, not
just a promise in a comment that could quietly stop being true as the
class grows.

### 2. Basic mechanics

```
unresolved count after 2 predictions, 0 resolutions: 2 (expect 2)
unresolved count after 1 resolution: 1 (expect 1)
resolved pairs: [(0.7, True)] (expect [(0.7, True)])

raised ValueError as expected: no recorded prediction for (ORD, 2024-01-05)
```

Record/resolve/read all behave exactly as specified, and resolving an
outcome for a prediction that was never recorded fails loudly instead of
silently creating a phantom entry.

### 3. The capstone: full replay against real data

```
airport=ATL, dates=[2019-06-20 .. 2019-06-24]
dates replayed: 5
unresolved after replay: 0

resolved (prediction, actual outcome) pairs:
  predicted=0.0001  actual=True   (×5)
```

All 5 dates resolved — no gaps, no leftover unresolved predictions. ATL
came back `True` for all 5, consistent with every prior finding about this
airport (Steps 4 and 10): it's a busy hub with near-constant EDCT activity.

### 4. Scoring the shadow log with the real KPI suite

```
calibration error: 0.9999
false-alarm cost: $0.00
cost-weighted recall: 0.0000
```

All three numbers are *bad* — and correctly so, for a fully expected
reason: the model is completely untrained (random weights), so there's no
reason to expect good predictions. `false-alarm cost` is $0 not because the
model did well, but because there were zero negative-labeled examples in
this 5-day window (ATL was `True` every day) — there was no way to have a
false positive at all. `cost-weighted recall` of 0 means the model caught
none of the real disruption risk in this window — again expected from
random weights, not a sign of a bug.

## A real practical finding worth flagging for whoever trains this for real

Every prediction came out essentially `0.0001` — the model saturated hard
toward zero. This isn't a bug in the pipeline; it's a direct consequence of
feeding **unnormalized, wildly different-scale features** into a freshly
initialized network: `day_of_year_sin`/`day_of_year_cos` sit in `[-1, 1]`,
while `climo_tmpf` for June/July at ATL sits around 78–81 (`°F`) — nearly
two orders of magnitude larger. With random initial weights, that scale
mismatch is enough to push the model's internal logits strongly negative,
saturating the sigmoid near 0 regardless of the actual input pattern.

**This is the same kind of practical, easy-to-miss finding as Step 1's
loss-scale/learning-rate note**: whoever wires up real training in Phase 2
will need to normalize/standardize `features/planning.py`'s output columns
(e.g. z-score each feature) before feeding them to any of these
architectures — otherwise training will start from a badly saturated
regime and may struggle to learn at all, independent of whether the
architecture or loss function themselves are sound.

## What this validation does *not* cover

- The model is untrained — every prediction here is noise from random
  initialization, not a real forecast. This proves the six pieces connect
  correctly, not that any of them works.
- Only 5 dates, 1 airport, and a 2-month climatology sample were used, to
  keep real EDCT loading cost manageable (see
  `features/planning_features_validation.md`'s ~13s/month measurement) —
  not a claim that this is a representative evaluation window.
- Real shadow-mode deployment (running continuously alongside actual live
  operations) still requires a live data feed and a trained model, neither
  of which exists yet.

## Where the Planning tier stands now

This closes Steps 1–12 of the Planning-tier build sequence. Per the build
sequence doc's own framing, this is the **"hold short" convergence point**:
features are built, the label is defined, baselines and the novel
architecture are written and validated, and evaluation (KPI suite,
temporal blocked CV, stress testing, shadow mode) is wired up end to end.
Real training — and the feature-normalization fix flagged above — comes
next, once the team-agreed temporal split (`TEAM_PLAN.md`, still pending
Person A/B sign-off) is actually settled and Person B's rotation-chain
graph exists for the Regime Transformer to use for real.
