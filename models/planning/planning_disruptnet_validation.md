# Planning DisruptNet model stub: validation results and explanation

Same philosophy as every prior validation in this project: synthetic
inputs, no training, no real data. This proves the wiring is correct —
that `compute_loss(loss_type=...)` actually routes to two genuinely
different loss functions, not that anything here can predict a real
disruption.

## What Step 7 actually assembles

Nothing new was invented. This wraps two already-validated, separately-
built pieces into one named, trainable unit:

- **Architecture** = `MoERegimeRouter` (Step 2) — used exactly as-is, no
  changes.
- **Loss** = a runtime choice: `"standard"` (weighted binary cross-entropy)
  or `"faoc"` (`FAOCLoss`, Step 1).

**Why a choice, not just FAOC-Loss:** if this model could only ever train
with FAOC-Loss, there'd be no way to isolate whether FAOC-Loss itself
helped, or whether any reasonable model would do about as well. Keeping the
architecture fixed and only swapping the loss is what makes a later claim
of "FAOC-Loss specifically improved things" a fair, controlled comparison
— rather than a vague "our whole system is better."

**Two different reasons a loss can be "weighted," not the same idea twice:**
`"standard"` weights by *class imbalance* — a purely statistical concern
(disruptions are presumably rarer than non-disruption days). `"faoc"`
weights by *real dollar cost* per class and per airport — an economic
concern, with no relationship to how often each class occurs. The
validation below exists specifically to prove these are actually different
mechanisms, not the same math wearing two names.

## How to reproduce this

```bash
cd nas_project
python -m models.planning.validate_planning_disruptnet
```

## Results

### 1. Forward pass — MoE routing passes through unchanged

```
pred_probs shape: (5,) (expect (5,))
in (0, 1): True
gate_weights shape: (5, 4) (expect (5, 4))
gate_weights sum to 1: True
```

Confirms wrapping `MoERegimeRouter` inside `PlanningDisruptNet` didn't
break or hide anything from Step 2 — the gate weights are still directly
accessible and still a valid softmax distribution, exactly as validated
before.

### 2. Both loss types run, and land on very different scales

```
standard loss: 0.6831
faoc loss:     8,971.52
```

Expected, and consistent with every FAOC-Loss number seen in this project
so far (`losses/faoc_loss_validation.md`): `"standard"` is a plain,
unitless cross-entropy value; `"faoc"` is dollar-scaled. Different scales
alone don't prove much though — the next check is the one that actually
matters.

### 3. The check that proves these are genuinely different code paths

```
same x/labels, airports changed to all-LGB (multiplier 2.00):
  standard loss: 0.6831 -> 0.6831  (unchanged, as expected: True)
  faoc loss:     8,971.52 -> 10,679.40  (changed, as expected: True)
```

Same predictions, same labels — only the airport list changed, to five
copies of `LGB` (this project's *highest* delay-propagation multiplier,
2.00 — see `losses/delay_propagation_multipliers.csv`).

- **`"standard"` didn't move at all.** It has no concept of airport, so it
  shouldn't — and it didn't.
- **`"faoc"` went up** (8,971.52 → 10,679.40), because more of the batch's
  disruption-labeled examples are now attributed to a higher-cascade
  airport.

This is the real confirmation: it's not just that the two loss types
*produce different numbers* (they'd do that trivially even if one had a
typo'd weight) — it's that one is **provably blind** to a variable the
other is **provably sensitive** to, exactly matching what each loss is
supposed to care about. If `"standard"` had also changed here, that would
mean airport info was leaking into the class-imbalance path by mistake — a
real, hard-to-notice bug this test is specifically designed to catch.

### 4. Error handling

```
raised ValueError as expected: loss_type must be 'standard' or 'faoc', got 'nonsense'
```

An invalid `loss_type` fails loudly and immediately rather than silently
falling back to one of the two real options.

## What this validation does *not* cover

- No real feature data or real labels — synthetic tensors only, same
  caveat as every prior step.
- No training happened — `PlanningDisruptNet` is exactly as untrained as
  `MoERegimeRouter` was in Step 2.
- `pos_weight` (the standard loss's class-imbalance weight) is a
  placeholder default of `1.0` — unweighted — flagged in
  `planning_disruptnet.py`'s docstring as needing recalibration once real
  label prevalence is known from `features/planning_label.py`, the same
  way FAOC-Loss's own duration assumptions do.
