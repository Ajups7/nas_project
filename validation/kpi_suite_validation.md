# Six-metric KPI suite: validation results and explanation

Different validation style from most of this project's prior steps: instead
of "does the mechanism behave sensibly" (Steps 1, 2, 6, 7, 8), each metric
here is checked against a **hand-traced example with a known correct
answer worked out before running any code** — closer to a textbook unit
test than an exploratory sanity check, because these are formulas with
exact right answers, not architectures whose behavior is judged relatively.

## What Step 9 is, and which 3 metrics are and aren't from the build doc

The build sequence names 3 metrics explicitly: *"false-alarm cost,
lead-time credit, regime-transition F1, and more."* The other 3 —
**cost-weighted recall, probability calibration error, forecast
stability** — are this module's own addition, chosen to cover gaps the
named 3 leave open (see `kpi_suite.py`'s docstrings for the reasoning
behind each). All six exist specifically because `PLAN.md` frames this
project's evaluation as needing to go "beyond generic statistical
accuracy" toward real FAA/ATCSCC operational concerns.

## How to reproduce this

```bash
cd nas_project
python -m validation.validate_kpi_suite
```

## Results

### 1. False-alarm cost

```
2 false positives: got 2952.300000, expected 2952.300000  [OK]
```

5 hand-picked predictions with exactly 2 false positives. Expected value
was computed by hand as `2 × fp_weight` (reusing `FAOCLoss`'s own dollar
figure, not a separate constant) — matched exactly.

### 2. Cost-weighted recall

```
catch 2 of 3, different airports: got 0.745182, expected 0.745182  [OK]
```

3 true positives at 3 different airports (ORD, ADK, LGB — different delay
propagation multipliers); the model catches 2 of the 3 (ORD and LGB, misses
ADK). Expected value hand-computed as `(caught dollar-weighted risk) /
(total dollar-weighted risk)`. **This is the metric earning its keep**: a
*plain* recall here would say `2/3 = 0.667` — treating the missed ADK
disruption the same as if it had been the missed ORD or LGB one. But ADK
has the lowest delay-propagation multiplier in the whole dataset (1.19),
so missing it is actually the *cheapest possible miss* — cost-weighted
recall correctly comes out *higher* (0.745) than plain recall would,
because the dollars actually at risk were mostly caught even though one
event was missed.

### 3. Lead-time credit

```
lead times [30, 15, 0] / 30-day horizon: got 0.500000, expected 0.500000  [OK]
```

3 true positives with 30, 15, and 0 days of advance warning → credits of
1.0, 0.5, 0.0 → average 0.5. Confirms the linear lead-time-to-credit
mapping works as designed.

### 4. Regime-transition F1

```
hand-traced transition F1: got 0.666667, expected 0.666667  [OK]
```

A 7-day true/predicted sequence with 3 real transitions and 3 predicted
transitions, only 2 of which actually line up (worked out by hand: TP=2,
FP=1, FN=1 → precision=recall=2/3 → F1=2/3). Matched exactly — confirms the
function is finding transitions (label changes), not just comparing raw
labels day by day.

### 5. Probability calibration error

```
perfectly calibrated bucket (0.2 pred, 0.2 actual): got 0.000000, expected 0.000000  [OK]
badly miscalibrated (0.9 pred, 0.0 actual): got 0.900000, expected 0.900000  [OK]
```

Two extremes, both by construction: 5 examples all predicted at 0.2, with
exactly 1 in 5 (20%) actually positive — a perfectly calibrated bucket,
correctly scored 0. Then 5 examples all predicted at 0.9 with *zero*
actually positive — as miscalibrated as this metric can detect in one
bucket, correctly scored 0.9 (the full gap between claimed and actual
confidence).

### 6. Forecast stability

```
swinging predictions: got 0.700000, expected 0.700000  [OK]
stable predictions: 0.0133 (expect small, near 0)
swinging is correctly rated less stable than stable: True
```

A sequence of predictions bouncing between 0.1 and 0.9 for the same target
date scored 0.7 (large day-over-day swings, hand-computed exactly). A
sequence hovering near 0.70 scored 0.0133 — correctly rated far more
stable. This is the metric with no equivalent among the other 5: it says
nothing about whether the model is *right*, only whether its guidance
holds steady long enough for a planner to act on it without it flip-
flopping.

### Error handling

```
cost_weighted_recall with no true positives raises: no true-positive examples in this batch - recall is undefined
lead_time_credit with no true positives raises: no true positives to score - lead-time credit is undefined
forecast_stability with <2 predictions raises: need at least 2 predictions over time to measure stability
```

Metrics that are mathematically undefined for degenerate inputs (e.g.
recall with zero positive examples) fail loudly instead of silently
returning `0` or `NaN`, which could easily be misread as "the model
scored zero" rather than "this metric doesn't apply here."

## What this validation does *not* cover

- Every input here is hand-constructed specifically to have a known
  answer — none of it is real model output, since no model has been
  trained yet (same caveat as every architecture step in this project).
- **`lead_time_credit` and `forecast_stability` both need data this
  project doesn't produce yet**: `lead_time_days` requires knowing the
  actual event date relative to when a prediction was made (a downstream
  computation once real rolling predictions exist), and
  `forecast_stability` needs repeated predictions for the same target date
  made at different as-of times — neither exists until Phase 2 training
  and inference are actually running.
- These 6 metrics are not wired into a single end-to-end evaluation report
  yet — that assembly, plus stratifying by regime, is Step 10's job
  (temporal blocked CV).
