# Day-ahead features: validation results and explanation

Validates `strategic.py` against **real downloaded data**
(`data/aspm_airport_analysis/`, `data/aspm_edct_report/`,
`data/weather_metar/`), the same approach as
`planning_features_validation.md` and `rotation_graph_validation.md`.

## What Step 2 is building

Four feature groups for the 2-24h forecast horizon, all answering the same
question: *what would you actually know about this airport the day before
the forecast day?*

1. **Calendar/seasonal signals** — reused directly from
   `features.planning.build_calendar_features`, not reimplemented (it's
   pure date arithmetic with nothing tier-specific in it).
2. **Recent operational conditions** — yesterday's actual ASPM Airport
   Analysis (delays, taxi times, on-time percentages).
3. **Recent disruption activity** — yesterday's actual EDCT rate (this
   project's GDP proxy).
4. **Recent weather actuals** — yesterday's actual METAR summary.

Why yesterday's *actuals* rather than climatology (Planning tier's
approach) or a live nowcast (Tactical tier's job): at 2-24h out there's no
real forecast to lean on, but there's a much stronger near-term signal
than "typical for this time of year" — day-to-day persistence. Using
today's own actuals to predict today would be leakage; yesterday's is the
most recent point that's both informative and genuinely knowable in
advance.

## How to reproduce this

```bash
cd nas_project
python -m features.validate_strategic_features
```

Runs on airport **ATL**, June 1–10, 2023 — deliberately spanning a month
boundary (June 1's "yesterday" is May 31, in a different `(year, month)`
file) to exercise the cross-month shift in `build_strategic_features`.

## Results

### 1. Cross-month shift works correctly — the main thing this validates

```
Missing values per column (excluding recent_gust): 0
```

Every row, including June 1st (whose features come from May 31st, loaded
from a *different* month's ASPM/EDCT files than June's), came back fully
populated. This is the one place a shift-by-one-day design could silently
break — an off-by-one or an unrequested month would show up as `NaN` on
exactly the boundary date, and it didn't.

**Direct cross-check** (not just "no NaNs" — an exact match):

```
raw June 1 recent_disruption_rate row: [0.50166667]
build_strategic_features June 2 row's recent_disruption_rate: [0.50166667]
```

`build_recent_disruption_features` computed June 1's own disruption rate
as `0.5017`; after the one-day forward shift inside
`build_strategic_features`, that exact value appears on June 2nd's row —
confirming the shift moves data forward by exactly one day, not zero or
two, and doesn't corrupt the value in transit.

### 2. Recent operational/disruption features — plausible day-to-day variation

```
      date  recent_pct_ontime_gate_dep  recent_disruption_rate
2023-06-01                   84.00                    0.502
2023-06-02                   69.59                    0.859
2023-06-03                   71.82                    0.654
```

On-time percentage and disruption rate move independently day to day (June
2 has both the lowest on-time % and the highest disruption rate of the
three) — the kind of correlated-but-not-identical movement real airport
operations data should show, not two columns secretly encoding the same
thing.

### 3. Recent weather — `recent_gust` has real, expected missingness

```
recent_gust    2
```

2 of 10 dates (June 5, June 6) came back with `NaN` for `recent_gust`.
This is expected, not a bug: `gust` is only reported when a gust actually
occurs (confirmed in `planning_features_validation.md` — ~89% `"M"`/missing
in ATL's full METAR history), so a calm day legitimately has no gust
observations to average. Every other numeric weather column had zero
missing values across all 10 dates.

## Design decisions worth flagging

- **`build_recent_operational_features`/`build_recent_disruption_features`
  return same-day aggregates, unshifted** — the one-day-forward shift
  happens only inside `build_strategic_features`, not in the individual
  helpers. This keeps each helper honestly named (it returns *that date's*
  actual conditions, nothing about "yesterday" baked into its own logic)
  and reusable by anything that wants same-day rather than day-ahead
  aggregates later.
- **EDCT's known missing-month handling is duplicated from
  `features/planning.py`/`features/planning_label.py`**, not extracted
  into a shared helper — matches those two modules, which already
  duplicate the identical file-existence check between themselves rather
  than centralizing it in `common/data_loader.py`. Kept consistent with
  that existing pattern rather than introducing a new shared utility
  unprompted (`common/data_loader.py` is Phase 0 shared infra —
  `TEAM_PLAN.md` asks for a heads-up to the team before changing it).
- **`recent_*` column prefix on every non-calendar feature** — makes it
  visually unambiguous, in any downstream table or model input list, which
  columns describe the forecast day itself (unprefixed: `month`,
  `day_of_week`, ...) versus the day before it (`recent_*`) — a mistake
  here (accidentally treating a `recent_*` column as same-day) would be a
  real leakage risk for a live deployment, even though it's harmless for a
  backtested table.

## What this validation does *not* cover

- Only ATL and a 10-day window were checked in detail. The cross-month
  shift logic is exercised once (May→June); it isn't re-tested against a
  year boundary (Dec 31→Jan 1) or a known-EDCT-gap month
  (2016-07/2025-09), though the file-existence check that handles gaps is
  the same code path already validated by
  `planning_features_validation.md`.
- No claim that these four feature groups are *sufficient* for the
  Strategic tier's baseline model (Step 4) — only that what's built here
  is computed correctly, with the right day-ahead semantics, from real
  data.
