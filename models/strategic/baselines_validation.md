# Strategic baseline models: validation results and explanation

Validates `models/strategic/baselines.py` two ways, going further than
`models/planning/validate_baselines.py` did for its own tier:

1. **End-to-end with real Strategic-tier data** — Step 2's
   `build_strategic_features()` → `make_sequence_windows()` →
   `Seq2SeqBaseline`/`TFTBaseline` → `losses/faoc_loss.py`'s `FAOCLoss`,
   and `SARIMABaseline` fit directly on ATL's real
   `recent_disruption_rate` series. Person C's own baseline validation
   used purely synthetic `torch.randn` tensors and a synthetic sine-wave
   series — reasonable at the time (no tier feature pipeline existed yet
   to test against), but this tier now has one (Steps 2-3), so this
   validation uses it.
2. **The same mechanical checks** `models/planning/validate_baselines.py`
   runs (output shape, range, order-sensitivity) — these architectures are
   freshly initialized and untrained, so this proves the mechanism is
   wired correctly for this tier's real feature width, not that either
   model has learned anything.

## What Step 4 is building

No new architectures — `models/strategic/baselines.py` imports
`SARIMABaseline`, `Seq2SeqBaseline`, and `TFTBaseline` directly from
`models/planning/baselines.py` (confirmed with the user as the preferred
approach: those classes are fully generic, so duplicating ~90 lines of
identical PyTorch code across tiers would just be two copies to keep in
sync). What's actually new: `make_sequence_windows()`, turning Step 2's
daily feature table into the `(batch, seq_len, num_features)` windows the
shared architectures expect, and `build_strategic_baselines()`, a small
factory returning one instance of each of the three, sized to this tier's
real feature width.

## How to reproduce this

```bash
cd nas_project
python -m models.strategic.validate_baselines
```

Runs on airport **ATL**, June 1–25, 2023 (25 days → 16 windows of
`seq_len=10`), using the real `build_strategic_features`/
`build_strategic_labels` output from Steps 2-3.

## Results

### 1. Real feature pipeline plugs into the shared architectures correctly

```
25 days, 24 feature columns: [...]
NaN columns before fill: {'recent_gust': 4}
windows shape: (16, 10, 24) (expect (16, 10, 24))
window_labels shape: (16,)
```

24 real feature columns (calendar + operational + disruption + weather),
not a guessed or hardcoded number — `build_strategic_baselines()`
deliberately takes `num_features` as a required argument rather than
defaulting to a hand-counted constant, precisely to avoid this kind of
column-count drifting out of sync with `features/strategic.py`. The only
NaNs are in `recent_gust` (4 of 25 days), consistent with
`strategic_features_validation.md`'s finding that gust is only reported
when one occurs — filled with 0 in `make_sequence_windows()`, which is
the domain-correct value (no gust event), not an arbitrary placeholder.

### 2. Seq2Seq / TFT — correct shape, range, and order-sensitivity on real windows

```
=== seq2seq ===
output shape: (16,) (expect (16,))
in (0, 1): True
shuffling the time axis changes the output (model is order-sensitive): True
composes with FAOCLoss on real labels, no error: loss=12,656.09

=== tft ===
output shape: (16,) (expect (16,))
in (0, 1): True
shuffling the time axis changes the output (model is order-sensitive): True
composes with FAOCLoss on real labels, no error: loss=11,314.37
```

Both produce one valid probability per window, both are sensitive to the
actual chronological order of the 10-day window (shuffling the time axis
changes the output — expected, since both use an LSTM/attention encoder
that should care about sequence order, not just pool features), and both
compose with `FAOCLoss` against real labels without error. The large raw
loss values (fresh, untrained weights against `FAOCLoss`'s real dollar-cost
calibration — see `losses/cost_figures.md`) are expected and not a bug:
same caveat every prior architecture in this project carries, this proves
the wiring works, not that either model has learned anything.

### 3. SARIMA — a real, worth-noting limitation, not swept under the rug

`SARIMABaseline`'s own default order `(1,1,1)`/`(1,1,1,7)` double-
differences and needs more history than this 25-point window to estimate
seasonal parameters — confirmed directly: it emits statsmodels'
`EstimationWarning: Too few observations to estimate starting parameters
for seasonal ARMA` and effectively zeroes those parameters out. Using a
lighter order instead (`(1,0,0)`/`(1,0,0,7)`, the same override
`models/planning/validate_baselines.py` applies to its own short synthetic
series) removes the warning and fits normally:

```
2023-06-26    1.950121
2023-06-27    1.332592
2023-06-28    0.683643
2023-06-29    0.175652
2023-06-30   -0.239381
2023-07-01    0.371384
2023-07-02    0.547880
```

Worth flagging honestly: `recent_disruption_rate` is a percentage-like
quantity that can't actually go negative, but SARIMA is a linear model
with no such constraint, and June 30's forecast comes out at **-0.24** —
a real limitation of applying an unconstrained linear baseline to a
bounded-below quantity from only 25 points of history, not a coding bug.
`SARIMABaseline`'s own docstring already says its defaults (and, by
extension, any order choice) are "a guess... Phase 2's actual baseline run
should tune these against real data, not trust these defaults blindly" —
this result is a concrete example of exactly that caveat playing out.

## Design decisions worth flagging

- **Reused, not duplicated, per the user's explicit choice**: SARIMA/
  Seq2Seq/TFT live only in `models/planning/baselines.py`. This creates a
  cross-tier import dependency, flagged in `models/strategic/baselines.py`'s
  docstring the same way `features/strategic_label.py` flags its
  dependency on Person C's `build_planning_labels`.
- **`make_sequence_windows()` doesn't validate contiguous dates or a
  single airport** — it trusts the caller, matching this project's
  existing convention of putting that responsibility on whoever builds the
  input table (e.g. `features/planning.py`'s disruption climatology also
  leaves "the caller decides the window").
- **`build_strategic_baselines(num_features)` has no default for
  `num_features`** — deliberately, to avoid a hardcoded feature count that
  could silently drift out of sync with `features/strategic.py`'s actual
  output width, the way the module docstring's earlier draft almost did
  before this validation caught the real count (24, not a guessed 19).

## What this validation does *not* cover

- Only ATL and a 25-day window were checked. No training happens anywhere
  in this project yet — every check here is mechanical (shapes, ranges,
  sensitivity), gated on the team-agreed temporal split
  (`TEAM_PLAN.md`) before any real training run.
- Doesn't address SARIMA's negative-forecast limitation — noted, not
  fixed (a real fix, e.g. forecasting in log-space or clipping at 0, is a
  modeling decision for whoever actually tunes this baseline against real
  data, not something to bake into the generic `SARIMABaseline` class
  silently).
