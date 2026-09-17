# Adversarial stress testing: validation results and explanation

## What "adversarial" means here, honestly

The usual meaning — does a small malicious perturbation flip a *trained*
model's prediction — isn't measurable yet, because no model in this
project has been trained (same caveat as every prior step). What **is**
measurable regardless of training state: do these architectures **fail
safely** under extreme, malformed, or corrupted input? A model that
produces `NaN` when fed `NaN` is behaving correctly (loudly broken, easy to
catch); a model that produces a plausible-looking wrong number instead
would be far more dangerous, silently.

## How to reproduce this

```bash
cd nas_project
python -m validation.validate_adversarial_stress_tests
```

## Results

### 1. Robustness battery — 4 architectures × 4 stress scenarios, all clean

```
MoERegimeRouter, PlanningDisruptNet, Seq2SeqBaseline, TFTBaseline:
  large_magnitude_1e6    finite=True  valid_range=True  [OK]
  tiny_magnitude_1e-6    finite=True  valid_range=True  [OK]
  all_zero               finite=True  valid_range=True  [OK]
  single_example         finite=True  valid_range=True  [OK]
```

Every architecture built in Steps 2, 6, and 7 stayed numerically finite and
produced a valid probability under: inputs scaled up a million-fold, scaled
down a million-fold, an all-zero input, and a batch of size 1 (a classic
training-vs-single-example-inference gotcha — code that silently assumes
`batch_size > 1` is a common real deployment bug this specifically checks
for). No crashes, no `NaN`, no out-of-range probabilities anywhere.

### 2. The test tied directly to a real bug found earlier in this project

```
output contains NaN: True (expect True - visible, not silent)
NaN isolated to the affected example only: True
```

This connects straight back to `features/planning.py`'s real finding: raw
METAR data uses `"M"` for missing values, and a caller who forgot to coerce
it (the bug that was actually caught and fixed in Step 3) could feed a real
`NaN` into a model. This test confirms what happens if one *does* slip
through anyway: the model's output for that one example correctly becomes
`NaN` — visible, not a plausible-looking wrong number — and, just as
importantly, **the corruption stayed isolated to only the affected
example**. The other 4 examples in the same batch came back completely
unaffected. That's a meaningful confirmation that one bad row of data in a
batch can't silently poison predictions for the rest of the batch.

### 3. Disconnected graph node — a designed safeguard, tested in the exact scenario it exists for

```
output finite (self-loop safeguard prevents all -inf softmax row): True
output in valid probability range: True
```

`regime_transformer.py`'s `build_attention_mask()` adds a self-loop to
every node specifically so an airport with **zero** rotation-chain edges
doesn't end up with an all-`-inf` attention row (which would produce `NaN`
after softmax — dividing by zero, effectively). This test builds exactly
that worst case — every node in the graph fully disconnected — and confirms
the safeguard holds. This isn't a hypothetical edge case either: plenty of
real airports likely have few or no rotation-chain connections once Person
B's real graph exists, so this scenario is expected to occur in practice,
not just in a stress test.

### 4. FGSM-style perturbation scaffolding

```
predictions before: [0.4686, 0.5307, 0.5053, 0.4502, 0.4889]
predictions after:  [0.4638, 0.54, 0.4965, 0.4618, 0.4811]
max absolute change: 0.0116
output still a valid, finite probability after perturbation: True
```

Built the actual gradient-based perturbation mechanism (nudge the input in
the direction that most increases the loss, per the classic Fast Gradient
Sign Method) and ran it against `PlanningDisruptNet`'s current, **untrained**
weights. The small, bounded change (max 0.0116) and the fact that outputs
stayed valid confirms the *mechanism* works correctly — gradients flow,
the perturbation applies, outputs remain sane. **This says nothing about
whether the eventual trained model will be robust or fragile** — that
question is meaningless to ask of random weights. What matters is that this
scaffolding is now sitting ready to run the actual robustness test the
moment a real trained model exists.

## What this validation does *not* cover

- No trained model exists, so no claim is made about real adversarial
  robustness — only that the testing infrastructure itself works.
- The stress scenarios tested are numerical/structural edge cases, not an
  exhaustive adversarial search — a real pre-deployment audit would want a
  broader sweep (e.g. varying `epsilon` in `fgsm_perturb`, testing against
  real feature distributions rather than random noise) once real data and
  a trained model exist.
- `test_nan_propagates_visibly` and `test_disconnected_graph_node` each
  test one representative model (`PlanningDisruptNet`, the Regime
  Transformer) rather than every architecture — the same failure modes are
  plausible in the others too, just not individually re-tested here.
