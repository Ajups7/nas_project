# Temporal blocked CV: validation results and explanation

Two different validation styles combined, matching the two things this
module does: pure date-arithmetic mechanics (fast, exhaustively checkable,
no data needed) for fold generation, and a **real-data** check for regime
stratification — deliberately reusing the exact ATL/ADK contrast Step 4
already established, to see if this step's stratification check correctly
flags a problem we already know is there.

## What Step 10 builds on

This doesn't reinvent embargo logic — it calls Step 5's
`assign_temporal_split()` directly, once per fold, rather than
re-deriving train/validation membership with separate code. The only new
mechanics are (1) generating a *sequence* of rolling fold boundaries
instead of one fixed split, and (2) checking whether each fold's
validation window has a reasonable mix of positive/negative labels.

**Honesty about "regime-stratified"**: real GDP/Ground-Stop/MIT regime
labels are still blocked on ASPM Advisories access (`README.md`). Until
then, "stratified" here means checking the binary EDCT-based label from
Step 4 — not the true 4-regime signal this project ultimately wants. This
is a real, useful check today, but a narrower one than the build doc's
phrasing implies until Advisories access comes through.

## How to reproduce this

```bash
cd nas_project
python -m validation.validate_temporal_blocked_cv
```

Takes a few minutes — the regime-balance section loads real EDCT data (8
months, 2 airports, ~13s/month — see `features/planning_features_validation.md`
for why).

## Results

### 1. Fold generation mechanics

```
fold 0: train [2016-01-01 -> 2018-12-31]  validation [2019-01-31 -> 2019-07-29]
fold 1: train [2016-01-01 -> 2019-07-29]  validation [2019-08-29 -> 2020-02-24]
fold 2: train [2016-01-01 -> 2020-02-24]  validation [2020-03-26 -> 2020-09-21]
```

3 requested, 3 generated. Each fold's training window expands to absorb
the previous fold's validation period — standard walk-forward
cross-validation.

### 2. Embargo gap — exactly 30 days at every fold boundary, not just one

```
fold 0: gap = 30 days (expect 30)
fold 1: gap = 30 days (expect 30)
fold 2: gap = 30 days (expect 30)
```

Confirms Step 5's leak-prevention mechanism (see
`common/temporal_split.py`'s leap-year-safe exact-day calculation) is being
applied identically at every fold boundary, not just approximated after
the first one.

### 3. Walk-forward chaining is exact

```
fold 0 validation_end (2019-07-29) == fold 1 train_end (2019-07-29): True
fold 1 validation_end (2020-02-24) == fold 2 train_end (2020-02-24): True
```

Confirms there's no accidental gap or overlap introduced between one
fold's validation set becoming available as the next fold's training data.

### 4. Boundary case — running out of data

```
requested 20 folds, data only supports 12 before running out
(last fold's validation_end: 2025-11-24, data_end: 2026-01-31)
```

Requesting more folds than the 2016–2026 dataset can support doesn't crash
or silently produce a malformed fold — it correctly stops at 12 and returns
what actually fits.

### 5. The real-data check: does stratification catch what Step 4 already found?

```
ATL, fold 0 validation window: n_dates=180  positive_rate=1.000  well_stratified=False
ADK, fold 0 validation window: n_dates=180  positive_rate=0.000  well_stratified=False
```

This is the result worth dwelling on. `features/planning_label_validation.md`
already established that ATL (busy hub) almost always has EDCT activity in
any 30-day window, while ADK (tiny airport) almost never does. Running
real EDCT data for this specific fold's 180-day validation window through
`check_fold_regime_balance()` reproduces that finding exactly: **ATL comes
back 100% positive, ADK comes back 0% positive** — both correctly flagged
`well_stratified=False`, for the two opposite reasons this check exists to
catch. A model "evaluated" on either of these airports alone in this fold
would get a meaningless score — F1, cost-weighted recall, and
regime-transition F1 (`validation/kpi_suite.py`) all become degenerate or
undefined when one class never appears. This check exists specifically so
that failure mode gets flagged before anyone trusts a fold's results,
rather than discovered after the fact.

## What this validation does *not* cover

- Only fold 0 was checked against real data (to limit EDCT loading cost —
  see `features/planning_features_validation.md`'s measured ~13s/month).
  The other folds' regime balance wasn't checked here, though the same
  function applies to them identically.
- This says nothing about *multi-airport* stratification — a real training
  run would presumably mix many airports per fold, which could dilute a
  single airport's extreme imbalance. This check operates one airport at a
  time, as built.
- The "regime" being stratified is still the binary EDCT proxy, not the
  true GDP/Ground-Stop/MIT regime split — see the honesty note above.
