# Planning label: validation results and explanation

Runs against **real** EDCT data, not synthetic inputs.

## What the label is

For a given `(airport, date)`: **1 if a GDP day occurs at that airport on
any day in `(date+1, date+30]`, else 0.**

A **GDP day** (`common/disruption.py`, shared by all three tiers) is a day
on which **at least 10% of the airport's arrivals had an EDCT**.

Unknown days (missing EDCT month, or no arrivals reported) are handled
conservatively: a window containing a known GDP day is still `1`, but a
window with no GDP day *and* an unknown day can't be called `0`, so that row
is dropped. Windows running past the end of the data (2026-01-31) are
dropped too.

**Known limitation**: EDCT proxies Ground Delay Programs only — no Ground
Stop or Miles-in-Trail signal. Real GDP/GS/MIT labels need ASPM Advisories
access (`README.md`).

## Label redesign (2026-10)

### The original definition was nearly constant

Until 2026-10 the label was "any EDCT activity, arrival **or** departure, in
the next 30 days." Measured on the train period (2016-01 – 2022-12):

| | Original label |
|---|---|
| Overall prevalence | 90% |
| Median airport prevalence | 98% |
| Airports ≥ 95% positive | 65 of 82 |
| Airports ≤ 5% positive | 5 of 82 (no arrival data at all — see below) |

A model could score ~90% by memorizing which airports are busy, without
learning anything about timing.

### Why arrivals only

A GDP holds flights bound **for** a constrained airport at their origins.
So at airport X:

- **Arrivals with EDCT** are flights into X that were held — mostly a GDP
  **at X**.
- **Departures with EDCT** are flights leaving X held for a GDP at their
  **destination** — a disruption somewhere else.

Counting departures is why hubs read as "disrupted" almost daily: ATL
serves dozens of destinations, and one of them nearly always has a GDP.

Caveat: arrival EDCTs also come from Airspace Flow Programs (en-route
weather), not only the airport's own GDP. The threshold below filters most
of those isolated EDCTs out.

### Why 10%

There is **no published source** for this number. It is supported by two
checks on the train-period distribution of the daily share of arrivals with
an EDCT (193,546 airport-days):

**1. The distribution has a trough near 10% and a second hump at 20–50%.**

| Daily arrival-EDCT share | Airport-days | % |
|---|---|---|
| exactly 0 | 175,216 | 90.5 |
| 0.1–1% | 4,016 | 2.07 |
| 1–2% | 1,913 | 0.99 |
| 2–3% | 1,204 | 0.62 |
| 3–5% | 1,535 | 0.79 |
| 5–7.5% | 984 | 0.51 |
| **7.5–10%** | **570** | **0.29** |
| 10–15% | 867 | 0.45 |
| 15–20% | 1,056 | 0.55 |
| 20–30% | 1,945 | 1.00 |
| 30–50% | 2,811 | 1.45 |
| 50–100% | 1,429 | 0.74 |

Below 10%, counts fall off steadily — scattered, one-off EDCTs. Above it,
they rise into a second population: program days. The trough is shallow
(normalized for bin width it sits in roughly 10–15%), so this supports
"about 10%," not 10% exactly.

**2. Above 10%, days look like a sustained program.** Median hours per day
in which at least half of arrivals were held:

| Daily share | Median hours ≥ 50% held |
|---|---|
| < 10% | 0 |
| 10–20% | 2 |
| ≥ 20% | 8 |

### Effect of the threshold

| Threshold | Median airport GDP-day rate | Median airport 30-day label prevalence | Airports ≥ 95% positive |
|---|---|---|---|
| 5% | 2.5% | 28% | 1 of 82 |
| **10%** | **1.1%** | **14.5%** | **0 of 82** |
| 20% | 0.5% | 8% | 0 |
| 30% | 0.3% | 3% | 0 |

Caution on the "median airport" column: it is over all 82 airports,
counting the 5 no-data airports as 0, and per-airport prevalence is bimodal
(many near 0, many well above), so the median jumps with small changes in
which airports are included. On the final 75-airport training set
(`features/planning.py`'s `eligible_airports()`), 10% gives: pooled train
prevalence **32%**, per-airport quartiles **4% / 26% / 60%**, 24 airports
≤ 5%, none ≥ 95%. Prefer those numbers.

Highest GDP-day rates at 10%: EWR 38%, SFO 29%, LGA 25%, JFK 17%, BOS 16% —
the airports known for chronic GDPs, which is a sanity check that the
definition captures real programs.

**5% and 20% are evaluated as a sensitivity check** (the `threshold`
argument of `build_planning_labels`), so no conclusion rests on 10% alone.
If ASPM Advisories access arrives, calibrate the threshold against real GDP
issuances.

### Excluded airports (75 of 82 kept)

Airports need a known GDP-day value on ≥ 90% of train days
(`eligible_airports()` in `features/planning.py`):

- **APA, ASE, BJC, BOI, CMH** appear in the ASPM EDCT report only from
  mid-2023 — no train-period data at all.
- **OXR** (78% of train days known) and **GYY** (85%): dropping
  unconfirmable 0s biases their labels — OXR keeps only 4% of its train
  rows, 95% of them positive.

Every other airport keeps 96–98% of its train rows (the loss is the
2016-07 missing month).

## How to reproduce

```bash
cd nas_project
python -m common.aspm_cache edct          # one-time, ~15 min: Parquet cache of all EDCT months
python -m features.validate_planning_label
```

## Results

The independent check recomputes each label from the raw `.xls` files with
a plain per-day loop — a different path from both the Parquet cache and the
rolling-window implementation.

```
=== 1. Cross-check against raw .xls (2019-06 .. 2019-08) ===
  EWR 2019-06-20..07-05: label=1 independent=1 [OK] x4
  ATL 2019-06-20..07-05: label=1 independent=1 [OK] x4
  ABQ 2019-06-20..07-05: label=0 independent=0 [OK] x4
mismatches: 0

=== 2. Label takes both values (all 2019 dates) ===
  EWR: prevalence 1.000 over 365 dates
  ATL: prevalence 0.548 over 365 dates
  ABQ: prevalence 0.000 over 365 dates

=== 3. Missing month 2016-07: a 0 can't be confirmed, a 1 still can ===
  EWR: kept 2/2 -> both label 1
  ABQ: kept 0/2 -> []

=== 4. Window past end of data (2026-01-31) is dropped ===
  EWR horizon 30: kept 1/2 (expect 1)

=== 5. Strategic wrapper (horizon 1) still works ===
  EWR 2019 next-day GDP prevalence: 0.581 over 365 dates

=== 6. ASE (in ASPM only from mid-2023): 2019 dates are all unknown, so dropped ===
  ASE 2019 rows kept: 0 (expect 0)
  ASE 2024 rows kept: 31 (expect >0)

=== 7. Airport absent from EDCT entirely raises ===
  ValueError: no arrival EDCT data for airport ZZZ
```

Full output: `logs/validate_planning_label.log`.

## What this validation does *not* cover

- Individual airports remain extreme (EWR 100%, ABQ 0% in 2019); the
  learnable signal is in mid-range airports and seasonal swings. Evaluate
  per airport, not just pooled, so a good pooled score can't come from
  airport identity alone.
- Whether the label is *useful* to predict is a training-time question.
- No ground-truth GDP records were used; see "Why 10%."
