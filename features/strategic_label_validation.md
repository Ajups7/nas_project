# Strategic label (1-day horizon): validation results and explanation

Validates `strategic_label.py` against **real downloaded EDCT data**
(`data/aspm_edct_report/`), the same approach as
`planning_features_validation.md`, `rotation_graph_validation.md`, and
`strategic_features_validation.md`.

## What Step 3 is building

The label for the 2-24h Strategic horizon: for a given `(airport, date)`,
`label = 1` if EDCT activity (this project's GDP proxy) occurred at that
airport on `date + 1` — the very next calendar day — else `0`.

This is **not a new implementation**. `features/planning_label.py`'s
`build_planning_labels(airport, dates, horizon_days)` already computes "did
EDCT activity occur in `(date+1, date+horizon_days]`" for an arbitrary
horizon; with `horizon_days=1` that window collapses to exactly
`(date+1, date+1]`, i.e. this tier's exact label. `strategic_label.py` is a
thin wrapper fixing that one parameter, not a reimplementation — see that
module's docstring for the reasoning and the cross-tier coupling this
creates (flagged the same way `models/planning/regime_transformer.py`
flags its dependency on this person's `rotation_graph.py`).

## How to reproduce this

```bash
cd nas_project
python -m features.validate_strategic_label
```

Runs on airport **ATL**, June 28 – July 3, 2023 — deliberately spanning a
month boundary (June 30's label depends on July 1st's EDCT file, a
different month than June 30 itself lives in) to exercise
`build_planning_labels`'s cross-month window handling under
`horizon_days=1` specifically, since it was previously only validated at
`horizon_days=30` in `planning_label`'s own history.

## Results

### 1. Labels look plausible and vary realistically

```
      date  label
2023-06-28      1
2023-06-29      1
2023-06-30      1
2023-07-01      1
2023-07-02      0
2023-07-03      1
```

Not a constant column — July 2nd genuinely comes back `0` while every
neighboring date is `1`, consistent with real GDP activity being
intermittent rather than a permanent daily fixture at a busy hub like ATL.

### 2. Direct cross-check against raw EDCT data — exact match, not just "no crash"

```
Checking label for 2023-06-30 - should reflect EDCT activity on 2023-07-01
  manual check on raw July EDCT file: had_edct = True
  build_strategic_labels' label for 2023-06-30: 1
  match: True
```

Rather than trust the wrapper's own internals, this loads July 2023's raw
EDCT file directly, filters to ATL and July 1st, and manually checks
`arrivals_with_edct > 0` or `departures_with_edct > 0` — independently of
any of `build_planning_labels`'s own code path. It matches. This
specifically exercises the case that matters most here: June 30th's label
depends on data from a **different calendar month's file** than June 30th
itself, and the right file was actually loaded and checked.

### 3. Joins cleanly with Step 2's features on `(airport, date)`

```
features rows: 6, labels rows: 6, joined rows: 6
      date  recent_disruption_rate  label
2023-06-28                 1.370000      1
2023-06-29                 0.333958      1
2023-06-30                 0.897292      1
2023-07-01                 1.821875      1
2023-07-02                 1.418125      0
2023-07-03                 1.301875      1
```

`build_strategic_features` (Step 2) and `build_strategic_labels` (Step 3)
share the same `(airport, date)` keys by construction — an inner join
loses zero rows on either side (6 in, 6 in, 6 out), confirming the two
modules are actually usable together as a training table, not just
individually correct in isolation. Worth noting July 2nd's `label = 0`
sits right after the *highest* `recent_disruption_rate` (1.82) of the six
days — a reminder that "yesterday was bad" and "tomorrow is bad" aren't
the same signal, which is exactly why this is a real forecasting problem
rather than a lookup.

## Design decisions worth flagging

- **Reuse, not reimplementation.** No EDCT-loading or windowing logic was
  written for this step — `build_planning_labels` already existed, was
  already validated (via `planning_label_validation.md`) at
  `horizon_days=30`, and this file only checks that its behavior still
  holds at `horizon_days=1`, particularly across a month boundary.
- **This creates a real cross-tier code dependency** on
  `features/planning_label.py`, owned by Person C, that isn't on
  `TEAM_PLAN.md`'s official shared-utility list (FAOC-Loss, rotation
  graph, temporal split). Flagged explicitly in `strategic_label.py`'s
  docstring: if that function's signature or window semantics change
  without a heads-up, this module breaks silently.

## What this validation does *not* cover

- Only one airport (ATL) and a 6-day window were checked in detail, and
  only one specific date's label was cross-checked directly against raw
  data (rather than every date in the window) — a reasonable spot-check,
  not exhaustive verification of every day's label.
- Doesn't re-validate `build_planning_labels`'s handling of the 2 known
  missing EDCT months (2016-07, 2025-09) under `horizon_days=1`
  specifically — that code path is shared with the already-validated
  `horizon_days=30` case and wasn't exercised by this window's dates.
