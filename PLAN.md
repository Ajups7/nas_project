# NAS-DisruptNet: Project Plan

Hierarchical deep learning framework for probabilistic, multi-horizon forecasting of
disruptive regime transitions (GDP, Ground Stop, Miles-in-Trail) in the U.S. National
Airspace System, using loss functions, architectures, and evaluation criteria aligned
with FAA/ATCSCC operational decision-making rather than generic statistical accuracy.

Scope note: building all four phases below simultaneously is the most common way
projects like this stall. Work through them in order — get Phase 1 data and Phase 2
baselines solid before layering in the novel components in Phase 3.

## Phase 1 — Data foundation

1. **BTS On-Time Performance** (transtats.bts.gov) — flight-level delays,
   cancellations, tail numbers for rotation-chain construction.
   Status: DONE — 121/121 months (2016-01 to 2026-01).
   `data/bts_ontime/csv/`

2. **ASPM Advisories** (aspm.faa.gov, login-gated) — the actual GDP/Ground
   Stop/MIT event records; this is the real label source for the target variable.
   Status: NOT STARTED — needs an FAA account request submitted via the
   FAA Operations & Performance Data portal. Longest lead-time item; start early.

3. **Weather** (NOAA HRRR archives, METAR/TAF, NEXRAD) — feature source.
   Status: NOT STARTED.

   Added along the way, not in the original three:

   - **ASPM Airport Analysis** (aspm.faa.gov) — airport-level hourly
     capacity/demand context (AAR/ADR-style features).
     Status: DONE — 121/121 months (2016-01 to 2026-01).
     `data/aspm_airport_analysis/`

   - **ASPM EDCT Report** (aspm.faa.gov) — EDCT (Expect Departure Clearance
     Time) counts per airport/hour, a GDP proxy label usable until real
     Advisories access comes through.
     Status: DONE — 119/121 months. Known gaps: 2016-07, 2025-09 (both
     failed twice on retry; accepted as gaps rather than blocking the pipeline).
     `data/aspm_edct_report/`

## Phase 2 — Baseline models (no novel components yet)

4. Build one working tactical (0-2h) model and one strategic (2-24h) model
   using a standard loss (e.g. weighted cross-entropy), benchmarked against
   the seq2seq / Temporal Fusion Transformer / SARIMA baselines named in the
   paper's Section 5. Goal: a working, evaluable end-to-end pipeline before
   adding anything novel.
   Status: NOT STARTED.

## Phase 3 — The paper's actual novelty, layered in one at a time

5. **FAOC-Loss** — asymmetric, cost-sensitive loss calibrated to FAA GDP
   cost-per-flight-hour economics. Testable in isolation once baselines run,
   since it only changes the loss function, not the architecture.
   Status: NOT STARTED.

6. **Sector-graph-attention Regime Transformer** / rotation-chain GNN —
   architecture upgrade representing disruption propagation over scheduled
   aircraft rotations rather than geographic proximity alone.
   Status: NOT STARTED.

7. **Mixture-of-Experts regime routing**.
   Status: NOT STARTED.

## Phase 4 — Validation framework

8. Six-metric operational KPI suite, temporal blocked cross-validation with
   regime stratification, adversarial stress testing, shadow-mode deployment.
   Applied throughout Phases 2-3, formalized once the model set is stable.
   Status: NOT STARTED.

## Current position

Still inside Phase 1. Real gaps against the original plan: the ASPM Advisories
login request (item 2) and NOAA weather data (item 3) haven't been started.
Once those are in hand, Phase 2 (baseline models) is the next milestone.
