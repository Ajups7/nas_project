# Planning label: validation results and explanation

Same philosophy as `planning_features_validation.md`: this runs against
**real** downloaded EDCT data, not synthetic inputs, because real data
exists to check it against.

## What the label actually is

For a given `(airport, date)`: **1 if EDCT activity occurred at that
airport on any day in `(date+1, date+30]`, else 0.** In plain terms — "will
this airport need a Ground Delay Program at some point in the next 1 to 30
days?"

**Why "any EDCT activity" instead of a percentage threshold**: an EDCT
(Expect Departure Clearance Time) only gets assigned when the FAA is
actively running a Ground Delay Program — see `schema.md`, "GDP proxy label
source." Its mere presence at an airport-hour already means a GDP was in
effect. Unlike FAOC-Loss's `avg_fn_hours`/`avg_fp_hours` (`losses/
cost_figures.md` §4), which had no natural threshold and had to be a
documented guess, this label needed no invented cutoff — presence/absence
of EDCT activity is the real signal, not an approximation of one.

**Known limitation, stated up front**: EDCT only proxies Ground Delay
Programs — it says nothing about Ground Stops or Miles-in-Trail
restrictions (`schema.md`). So this label is a **lower bound** on "any
disruption occurred," not the true GDP/GS/MIT label `PLAN.md` ultimately
wants. That requires ASPM Advisories access, still blocked on FAA account
approval (`README.md`).

## How to reproduce this

```bash
cd nas_project
python -m features.validate_planning_label
```

Uses ATL and 4 dates deliberately chosen to straddle a month boundary (late
June into early July 2019), so the 30-day forward window has to pull EDCT
from 3 different calendar months — this exercises the multi-month loading
path, not just the easy single-month case.

## Results

### 1. Labels for ATL, and an independent cross-check

```
airport       date  label
    ATL 2019-06-20      1
    ATL 2019-06-25      1
    ATL 2019-06-30      1
    ATL 2019-07-05      1
```

All four came back `1`. On its own that's not very informative — it could
mean the label works correctly (ATL genuinely almost always has a GDP
somewhere in a 30-day window — plausible for one of the busiest hubs in the
country) or it could mean the function is broken and always returns `1`
regardless of input. To tell those apart, every one of these four labels
was recomputed a **second, independent way** — different code, loading the
months directly and filtering with one boolean mask instead of the real
function's groupby/reindex approach:

```
2019-06-20: build_planning_labels=1  independent_check=1  [OK]
2019-06-25: build_planning_labels=1  independent_check=1  [OK]
2019-06-30: build_planning_labels=1  independent_check=1  [OK]
2019-07-05: build_planning_labels=1  independent_check=1  [OK]
```

Both methods agree on all four dates. That rules out a class of bug where
the real function's logic and a naive re-implementation would disagree —
but it still doesn't rule out both being wrong in the *same* way, which is
exactly what the next check is for.

### 2. Proof the label can actually say "no" — not just always "yes"

```
=== Edge case: horizon_days=1 (shortest real window) at ATL ===
airport       date  label
    ATL 2019-06-20      1

=== Does the label ever actually return 0? Same dates, quiet airport (ADK) ===
airport       date  label
    ADK 2019-06-20      0
    ADK 2019-06-25      0
    ADK 2019-06-30      0
    ADK 2019-07-05      0
```

Shrinking the window to just 1 day still gave ATL a `1` — a real Ground
Delay Program happened there the very next day, each time. Running the
exact same dates and window at **ADK** (Adak, a tiny airport in Alaska —
also the *lowest* delay-propagation-multiplier airport in `losses/
delay_propagation_multipliers.csv`, at 1.19) came back `0` every time.

This is the check that actually matters: **the label genuinely
discriminates based on real airport activity, rather than mechanically
returning the same value no matter what's fed in.** A busy hub and a
near-empty regional airport, same code, same dates — opposite results,
and the direction of that difference matches basic real-world intuition
about which of the two airports is more likely to need a Ground Delay
Program at all.

## Performance

Computing labels for 4 dates spanning 3 calendar months (2019-06, 2019-07,
2019-08) took **40.0 seconds** — consistent with `planning_features_validation.md`'s
measured ~13 seconds/month for `load_edct()`. This confirms
`build_planning_labels()`'s batching design works as intended: it identified
all 3 months needed *once*, up front, rather than reloading overlapping
months once per date (which, for 4 dates each needing up to 2 months, could
have meant up to 8 loads instead of 3).

## What this validation does *not* cover

- Only one pair of airports (ATL, ADK) was checked — both at the extreme
  ends of "busy" and "quiet." A mid-traffic airport wasn't tested, so
  there's no direct evidence yet of sensible behavior in between.
- This confirms the label is computed correctly from real EDCT data — it
  says nothing about whether "any EDCT activity in 30 days" turns out to be
  a *useful* target for the model to predict. That's a Phase 2/3 question,
  same caveat as every prior validation in this project.
- The known EDCT-only limitation (no Ground Stop/MIT signal) means this
  label likely *undercounts* true disruptions — it can only get more
  accurate once ASPM Advisories access comes through.
