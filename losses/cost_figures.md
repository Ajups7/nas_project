# FAOC-Loss: sourced cost figures

Step 1 of the Planning tier build sequence. These are the FAA/A4A dollar
figures FAOC-Loss (Asymmetric, cost-calibrated loss) calibrates against.
Fetched and compiled 2026-09-03. Nothing here was in the repo or in the four
downloaded datasets (BTS, ASPM Airport Analysis, ASPM EDCT, METAR) — none of
those carry cost data, only operational/weather records.

## 1. Base cost-per-flight-hour: A4A direct operating cost

Source: [Airlines for America — U.S. Passenger Carrier Delay Costs](https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/),
"Annual and Per-Minute Cost of Delays to U.S. Airlines" table.

**Calendar year 2025, cost per block minute (taxi + airborne time), U.S. passenger carriers:**

| Category | $/minute | YoY change |
|---|---|---|
| Crew (pilots/flight attendants) | $37.01 | +5.1% |
| Fuel | $29.34 | -11.3% |
| Maintenance | $18.35 | +0.3% |
| Aircraft ownership | $9.76 | -4.2% |
| Other | $3.95 | -0.8% |
| **Total direct operating cost** | **$98.41** | **-2.3%** |

→ **≈ $5,904.60 per aircraft block-hour.** This is the base unit for
converting a delay/disruption's duration into a dollar figure — the direct
cost an airline actually incurs per hour of block time. Methodology: DOT
Form 41 data for U.S. scheduled passenger carriers.

## 2. Cross-check: FAA's own operating-cost figures

Source: [FAA Office of Aviation Policy and Plans — Economic Values for FAA Investment and Regulatory Decisions, A Guide: 2021 Update](https://www.faa.gov/sites/faa.gov/files/regulations_policies/policy_guidance/benefit_cost/econ-value-section-4-op-costs.pdf),
Section 4, based on 2018 Form 41 data.

| Figure | Value (2018 $) |
|---|---|
| Part 121 passenger carriers, all aircraft, avg. **total** cost/block hour (Schedule P-5.2) | $3,985 |
| Part 121 passenger + all-cargo combined, avg. total cost/block hour, **all indirect costs included** (Table 4-2, "Total Carriers") | $9,359 |
| Narrow-body (160 seats and below), passenger, total cost/block hour | $4,045 |

**Why the gap between A4A's $98.41/min (~$5,905/hr) and FAA's $9,359/hr:**
A4A reports *direct* operating cost only (crew, fuel, maintenance, ownership).
The FAA's $9,359 figure folds in indirect costs (traffic servicing, G&A,
reservations, transport-related expenses) — i.e. full airline cost
allocation, not the marginal cost of one delayed aircraft. **For FAOC-Loss,
the A4A figure is the more defensible base rate** — it's what the
delay-cost literature and FAA's own benefit-cost guidance conventionally
cite as "cost of delay per aircraft," and it isolates the cost actually
attributable to the extra block time, not general overhead.

## 3. Delay propagation multiplier (per-airport, network-effect weight)

Source: [FAA Economic Values Guide, Section 10.1 — Delay Propagation Multipliers](https://www.faa.gov/regulations_policies/policy_guidance/benefit_cost/econ-value-section-10-other-values.pdf)
(developed with MITRE from ASQP data; methodology: *Calculating Delay
Propagation Multiplier for Cost-Benefit Analysis*, MITRE, Feb. 2010).

Full table: [`delay_propagation_multipliers.csv`](delay_propagation_multipliers.csv)
— 304 major U.S. commercial airports, 3-year average 2016–2018, keyed by
`airport` (matches `schema.md`'s join key directly).

`DM(i) = (D(i) + Dp(i)) / D(i)` — a multiplier of 1.65 at airport *i* means
a one-minute reduction in original delay there yields a further 0.65 minutes
of downstream delay reduction, system-wide. **National composite: 1.51.**

**Coverage against this project's data:** of the 82 airports in
`data/weather_metar/`, 75 have an airport-specific multiplier in this table;
7 do not (`APA`, `BJC`, `GYY`, `OXR`, `RFD`, `TEB`, `VNY` — all
GA/reliever fields, not in FAA's "major commercial airport" set). Fall back
to the 1.51 national composite for those seven.

**Why this matters for FAOC-Loss specifically:** a missed disruption at a
high-multiplier airport (e.g. ORD at 1.48, ATL at 1.52, or a >1.7 outlier
like BUR/DAL/HOU) costs more in system-wide terms than the same miss at a
low-multiplier field (e.g. ADK at 1.19, LWS at 1.19) — this is the
principled way to make the loss's *cost* scale with real cascading impact,
not just flag count, and it plugs straight into what Person B's
rotation-chain graph is already modeling structurally.

## 4. What's still a modeling decision, not a sourced fact

The dollar figures above give the **scale** of cost per flight-hour. They do
**not** by themselves give the false-negative vs. false-positive cost
*ratio* that makes the loss asymmetric — no public FAA/A4A source publishes
"cost of a missed GDP prediction" vs. "cost of a false GDP alarm" as
distinct figures, because that ratio depends on the operational response
being modeled (e.g. a false alarm might mean an unnecessary partial ground
hold — a fraction of full block-hour cost — while a miss means the full
uncontrolled cascading delay). **This ratio has to be a documented modeling
assumption inside `faoc_loss.py` itself** (with the reasoning stated in a
comment/docstring), not something to keep hunting for externally.

## 5. Why no 2016–2026 time series

This project's operational data (BTS/ASPM/EDCT/METAR) spans 2016–2026, but
the cost figures above are deliberately **not** matched year-by-year against
it:

- The FAA's own guide is only republished periodically (2007, 2015, 2021
  editions) — there's no clean annual series to draw from even if we wanted
  one.
- FAOC-Loss's job is to weight the *relative* cost of a false negative vs.
  false positive consistently across training examples. Inflating/deflating
  that weight year-by-year would inject nominal-dollar drift (fuel price
  cycles, wage growth) into the loss surface — noise that has nothing to do
  with the disruption-prediction task the model is actually learning.
- The delay propagation multipliers are already a fixed 3-year average
  (2016–2018) and are airport-specific rather than year-specific — they
  don't need extending either.

**Recommendation:** calibrate FAOC-Loss with the single current A4A
figure (~$5,905/hr) and the fixed per-airport multiplier table, both
committed here with citations. Revisit only if the model is later
recalibrated for deployment in a specific future year.
