# Planning features: validation results and explanation

## Features (`features/planning.py`, 12 total)

One row per `(airport, date)`. The prediction is made at the end of `date`
about the window `(date+1, date+30]`.

| Group | Columns | Notes |
|---|---|---|
| Calendar | `day_of_year_sin`, `day_of_year_cos` | Cyclical, so Dec 31 ≈ Jan 1 |
| Weather climatology | `climo_{tmpf,dwpf,drct,sknt,gust,vsby,alti}` | METAR mean for that airport + calendar month |
| Disruption climatology | `climo_gdp_day_rate` | Fraction of GDP days for that airport + calendar month |
| Recent history | `recent_gdp_rate_7d`, `recent_gdp_rate_30d` | GDP-day rate over the past 7 / 30 days, through `date` |

### Design decisions (2026-10)

- **Climatology from train dates only** (≤ `TRAIN_END`, 2022-12-31). The
  earlier version averaged over every year given to it; with full history
  that puts validation/test GDPs into their own features.
- **Climatology describes the forecast window**, not the as-of month: each
  of the next 30 days gets its calendar month's value, then they're
  averaged. A June 25 prediction sees 5 days of June and 25 of July.
- **Disruption climatology uses the GDP-day definition** that the label
  uses (`common/disruption.py`). The earlier version averaged arrival and
  departure EDCT share — the inflated signal the label redesign removed.
- **Recent history added.** The original design excluded near-term EDCT
  signal at this horizon. A 30-day look-back is the airport's current
  state (construction, a stormy stretch), known at prediction time, and
  doesn't overlap the label window. Models are trained with and without it
  (`RECENT_COLUMNS`) to measure what it adds.
- **75 of 82 airports** — see `planning_label_validation.md`, "Excluded
  airports."

## How to reproduce

```bash
cd nas_project
python -m common.aspm_cache edct                  # one-time EDCT cache, ~15 min
python -m features.planning                  # builds data/cache/planning_dataset.parquet, ~45 s
python -m features.validate_planning_features
```

## Results

```
=== 1. Forward-window climatology: 2019-06-25 covers Jun 26-30 + Jul 1-25 ===
  climo_gdp_day_rate=0.23630  by hand=0.23630  [OK]
  climo_tmpf=62.470  by hand=62.470  [OK]

=== 2. Leakage: climatology must not change if post-train GDPs change ===
  2024-07-01 climo before=0.23118 after setting every post-train day to GDP=0.23118  [OK]

=== 3. Recent history: 2019-07-15, past 7 and 30 days through the date itself ===
  7d=0.5714 by hand=0.5714   30d=0.5667 by hand=0.5667  [OK]
```

### Dataset (75 airports, 267,162 rows)

| Split | Rows | Label prevalence |
|---|---|---|
| train | 188,292 | 32.1% |
| embargo | 4,500 | 28.2% (never used) |
| validation | 27,375 | 34.4% |
| test | 46,995 | 41.3% |

**Prevalence rises over time** (32% → 34% → 41%): GDPs became more frequent
after 2023. A model calibrated on train will tend to under-predict on
test; check calibration on validation and expect some drift.

Missing values (train): `recent_gdp_rate_7d` 0.5%, `recent_gdp_rate_30d`
0.06% — windows that fall entirely in the 2016-07 missing month. Imputed at
training time.

### Correlation with the label (train)

| Feature | r |
|---|---|
| `climo_gdp_day_rate` | 0.527 |
| `recent_gdp_rate_30d` | 0.484 |
| `recent_gdp_rate_7d` | 0.422 |
| `climo_sknt` | 0.150 |
| other weather climo, calendar | −0.04 – 0.08 |

Disruption climatology and recent history carry most of the linear signal;
weather climatology adds little on its own (it may still help non-linearly,
e.g. wind × season).

## What this validation does *not* cover

- Correlation is linear and pooled across airports; it says nothing about
  within-airport skill.
- Weather climatology is "typical weather," not forecast weather — no real
  forecast exists 30 days out.
