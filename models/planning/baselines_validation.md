# Baseline architectures: validation results and explanation

Same philosophy as `moe_router_validation.md` (Step 2): synthetic inputs,
no training, no real data. This proves the three baseline architectures are
wired correctly — not that any of them can predict a real disruption. Real
training is gated on the team-agreed temporal split (`TEAM_PLAN.md`'s
proposed section) actually existing.

## What Step 6 is

Per the build sequence: *"Baseline architectures — SARIMA, seq2seq, TFT —
written, not run."* The point of this step (per `PLAN.md`'s own scope
note) is a working, evaluable baseline to benchmark against *before* any of
this project's novel components — FAOC-Loss, MoE routing, the eventual
Regime Transformer — get credit for outperforming anything. Skipping
straight to the "interesting" architecture without a baseline is explicitly
called out in `PLAN.md` as the most common way projects like this stall.

## Two different kinds of "baseline," on purpose

- **SARIMA** — a classical statistical model (`statsmodels`), fits a single
  time series directly via `fit()`/`forecast()`. It does **not** share an
  interface with the two neural baselines below — forcing a 1960s-era
  statistical method into a `forward(x)` tensor interface would misrepresent
  what it actually is.
- **Seq2Seq** and **TFT** — PyTorch modules, both `forward(x) → probability`,
  where `x` is a `(batch, seq_len, num_features)` window of daily feature
  vectors. Same output shape as Step 2's `MoERegimeRouter`, so both plug
  into `FAOCLoss` exactly the same way.

## How to reproduce this

```bash
cd nas_project
python -m models.planning.validate_baselines
```

## Results

### Seq2Seq

```
output shape: (5,) (expect (5,))
in (0, 1): True
different inputs -> different outputs: True
composes with FAOCLoss, no error: loss=8,329.83
```

Correct output shape and valid probability range. Confirmed the model
actually depends on its input (not silently ignoring it), and confirmed it
composes with `FAOCLoss` from Step 1 with no glue code — same integration
check used for the MoE router in Step 2.

### TFT (simplified)

```
output shape: (5,) (expect (5,))
in (0, 1): True
variable-selection weights shape: (5, 10, 6) (expect (5, 10, 6))
weights sum to 1 per (batch, timestep): True
different inputs -> different outputs: True
shuffling the time axis changes the output (model is order-sensitive): True
composes with FAOCLoss, no error: loss=9,420.07
```

Same basic checks as Seq2Seq, plus two TFT-specific ones:

- **Variable-selection weights sum to 1** at every timestep, for every
  example — confirming the gate is a genuine softmax distribution over
  features (same kind of check as the MoE router's regime gate in Step 2),
  not arbitrary numbers.
- **Shuffling the order of the input sequence changes the output.** This
  matters specifically because TFT's whole premise is attention *over
  time* — if scrambling which day comes first, second, etc. didn't change
  anything, the attention mechanism wouldn't actually be using temporal
  order, and the model would be no different from one that just pools
  features with no regard for sequence. Confirming order-sensitivity is
  confirming the architecture is actually doing what "temporal" in its name
  claims.

### SARIMA

```
forecast length: 7 (expect 7)
forecast values finite (no NaN/inf): True
forecast-before-fit correctly raises: call fit() before forecast()
```

Fit on a synthetic sine wave + noise (not real disruption data — this
checks the `statsmodels` wrapper's plumbing, nothing about real seasonal
disruption patterns). Forecast came back the requested length with no
`NaN`/`inf` values, and calling `forecast()` before `fit()` fails loudly
with a clear error instead of silently returning garbage.

## What this validation does *not* cover

- No real feature data (`features/planning.py`'s output) was fed into any
  of these — only random noise or a synthetic sine wave. This confirms the
  architectures are mechanically correct, not that they can predict
  anything real.
- No training happened. All three models are exactly as untrained as
  Step 2's `MoERegimeRouter` was — freshly initialized weights, no
  learning.
- SARIMA's `order`/`seasonal_order` defaults (a weekly-seasonality guess)
  are placeholders, explicitly flagged in `baselines.py`'s docstring as
  needing real tuning once Phase 2's actual baseline run happens — not
  validated as good choices here.
- The TFT here is intentionally simplified relative to the full published
  architecture (no multi-horizon quantile outputs, no static covariate
  encoder) — see `baselines.py`'s module docstring for the exact scope
  decision.
