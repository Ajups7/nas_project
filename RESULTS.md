# NAS-DisruptNet: results across the three tiers

All numbers are **test-set**: forecasts made from 2024-03-01 to the last
date whose outcome is known (2026-01-01 for Planning, 2026-01-30 for
Strategic, end of January 2026 for Tactical; data ends 2026-01-31). Each
tier was evaluated **once**, after its choices were frozen. Neural models:
mean of 3 training runs with different random starts (± = their spread). Details, validation results and every decision behind them:

- Planning (1–30 days): [`models/planning/training_results.md`](models/planning/training_results.md)
- Strategic (2–24 hours): [`models/strategic/training_results.md`](models/strategic/training_results.md)
- Tactical (0–2 hours): [`models/tactical/training_results.md`](models/tactical/training_results.md)

## Common setup

- **Label: GDP only, from an EDCT proxy.** ASPM Advisories (real
  GDP / Ground Stop / Miles-in-Trail records) were never available. A GDP
  day = ≥ 10% of the airport's arrivals held with an EDCT; a GDP hour =
  ≥ 50% held (and ≥ 2). Both thresholds sit at troughs of the observed
  distributions; no published source. Departure EDCTs are excluded from
  the definition — they reflect GDPs at destinations.
- 75 airports; train 2016–2022 (COVID period Mar 2020–Jun 2021 excluded),
  validation 2023, test 2024–2025, 30-day embargoes between splits.
- Decision cost uses FAOC-Loss's weights (miss = $11,809 × airport
  delay-propagation multiplier, false alarm = $1,476 — placeholder
  durations of 2 h and 0.25 h, `losses/cost_figures.md` §4), at the
  cost-ratio threshold (≈ 6–9%) fixed in advance.
- Weather is observed METAR (as climatology at Planning, actuals at
  Strategic/Tactical); no numerical weather forecasts (HRRR) were used.

| Tier | Event rate (test) | Best simple policy | Main model | Cost reduction vs best simple |
|---|---|---|---|---|
| Planning (GDP day in next 1–30 d) | 41.3% | always flag $867 | DisruptNet $620 | −28% |
| Strategic (GDP day tomorrow) | 5.6% | never flag $995 | STGNN top-10 $464 | −53% |
| Tactical (GDP hour in next 2 h) | 2.3% | never flag $407 | STGNN top-10 $36 | −91% |

Costs are per airport-day (Planning, Strategic) or per airport-hour
(Tactical).

## The paper's claims vs. the results

Claims as stated in the paper's Introduction (Goal, Objectives, Practical
contributions, Section 5). Theoretical contributions are left for after the
analysis. Verdicts: **Supported**, **Partly**, **Not supported**,
**Not tested**.

| Paper says | Results | Verdict |
|---|---|---|
| Disruptive events are the high-impact 5–10% of NAS days | GDP days 4–6% of airport-days (Strategic), GDP hours 1.5–2.3% of open airport-hours (Tactical); at 1–30 days, "a GDP somewhere in the next month" is 32–44% | **Supported** for the day/hour framing |
| Symmetric losses + IID splits look accurate but fail operationally | All splits are temporal with embargoes (no IID). A standard 0.5 threshold costs *more than always flagging* at Planning; pooled AUC (~0.91) is mostly airport identity; one validation year misled twice (recent history, TFT) — caught by blocked CV and the test | **Supported** |
| Goal: **probabilistic** forecasts | Calibration error 0.002–0.05 for standard-loss models at every tier; under-prediction of the 2025 rise at Planning (37% vs 44%) | **Supported**, with drift at long horizons |
| Goal: **multi-horizon**, hierarchical (0–2 h, 2–24 h, 1–30 d) | Three tiers built, trained and tested on a shared split; each tier is a separate model — no coupling between tiers was built | **Partly** |
| Goal: forecasts of **regime transitions** | Planning: transition F1 ≤ 0.044 (no model times transitions). Tactical: 80% of onsets are visible from held arrivals; on clean onsets (no held arrivals yet, 0.14% of hours) best PR-AUC 0.22, of which half comes from network visibility of already-issued programs | **Partly** — onset warning exists at 0–2 h but is modest |
| Goal: **GDP, Ground Stop and Miles-in-Trail** | GDP only (EDCT proxy); ASPM Advisories never available | **Not tested** for GS/MIT |
| Obj. 1: integrate flight ops, ATC initiative logs, high-res weather, rotation topology | Flight ops, rotation topology, observed weather: yes. Initiative logs, numerical weather forecasts: no | **Partly** |
| Obj. 2: rotation-chain graph architectures beat treating airports as independent | No effect at 1–30 d or 2–24 h. At 0–2 h the graph helps (−14% cost) but entirely through neighbors' departure EDCTs — already-issued programs, not propagation | **Not supported** as propagation forecasting |
| Obj. 3: FAOC-Loss aligns training with GDP cost economics | Same operational cost as standard training + cost-derived threshold (worse at Planning, ~1–2% better at Strategic, tied at Tactical); destroys calibration | **Not supported** as an improvement |
| Obj. 3: MoE regime routing | Ties logistic regression at Planning; beaten by sequence/graph models at short horizons; regime experts untrainable without regime labels | **Not supported** |
| Obj. 3: robust to structural breaks (pandemic) | COVID excluded, not modeled; including it changed nothing material; 2025 rise under-predicted at Planning | **Partly** |
| Obj. 4: six-metric KPI suite, temporal blocked CV with regime stratification, stress tests, shadow mode | All four used **at Planning only**; stratification is a positive/negative balance check, not regime-based. Strategic/Tactical: cost, calibration, onset metrics and sensitivity checks, no stress tests or shadow mode | **Partly** |
| Practical: calibrated probabilities of GDP/GS/MIT **onset** with lead time for the **90-minute CDM window** | At 0–2 h: calibrated GDP probabilities; clean-onset PR-AUC 0.22 at a 0.14% base rate (~150× chance) — useful ranking, but most alerts false. GS/MIT not available | **Partly** (GDP only, modest) |
| Section 5: results vs. Seq2Seq, TFT, SARIMA | Main models beat SARIMA everywhere; tie Seq2Seq/TFT at Planning, edge them at Strategic, beat them at Tactical. Logistic regression ties the main model at Planning — must be in the table | **Partly** |

## By objective

### Objective 1 — integrated multi-resolution data

Built: BTS (rotation graph), ASPM Airport Analysis, ASPM EDCT, METAR,
joined at daily (Planning, Strategic) and hourly UTC (Tactical)
resolution, with one-time Parquet caches (`common/aspm_cache.py`).
Not available: ATC initiative logs (ASPM Advisories), high-resolution
numerical weather. The label is a GDP-only proxy throughout.

### Objective 2 — spatially aware architectures (rotation-chain graph)

Same ablation at every tier: graph of each airport's 10 strongest 2019
rotation partners vs. any flights vs. no graph.

| Tier | Top-10 graph | No graph | Graph effect |
|---|---|---|---|
| Planning (Regime Transformer) | $734 ± 93 | $637 ± 33 | none / harmful |
| Strategic (STGNN) | $464 ± 1, within-AUC 0.717 | $464 ± 13, 0.716 | none |
| Tactical (STGNN) | $36.3, clean-onset PR-AUC 0.216 | $42.0, 0.103 | large |
| Tactical, departure EDCTs removed | $42.0, 0.099 | $41.9, 0.103 | none |

- At 1–30 days and 2–24 hours the rotation graph adds nothing.
- At 0–2 hours it cuts cost 14% and doubles clean-onset precision — but
  **all of that comes from neighbors' departure EDCTs**: flights to this
  airport being held elsewhere, i.e. a GDP already issued, visible through
  the network before held flights land. Without that signal the graph is
  worth nothing on test. Propagation of delays/weather along rotations is
  not detectable with these inputs.
- The STGNN architecture itself (interleaved recurrent + graph update)
  ranks within airports best at the Strategic horizon (0.716 vs 0.69),
  with or without the graph.

### Objective 3 — FAOC-Loss and Mixture-of-Experts routing

| Tier | Standard loss + cost-ratio threshold | FAOC-Loss at 0.5 | FAOC mean prediction vs actual |
|---|---|---|---|
| Planning | $620 | $648 | 68% vs 41% |
| Strategic | $466 | $459 | 23% vs 5.6% |
| Tactical | $45.9 | $45.3 | 4.8% vs 2.5% (2025) |

(Test set. Validation sensitivity settings: Planning FAOC more expensive
in all four; Strategic FAOC cheaper by $4–7 in all four; Tactical within
$0.5 in all four.)

- **FAOC-Loss roughly matches standard training with a cost-derived
  decision threshold, with a tier-dependent sign**: worse at Planning
  (test +$28, ~5%, and in all four validation settings), slightly but consistently cheaper at Strategic (−1–2%
  in all four validation settings and on test), tied at Tactical (within
  $0.5 in all four settings). Its probabilities are inflated 1.6–5× at
  every tier. Training on the cost ratio moves the decision threshold into
  the model; calibrated training + the threshold reaches about the same
  cost and keeps usable probabilities.
- **Mixture-of-Experts DisruptNet** ties logistic regression at Planning
  ($620 vs $617), beats it modestly at Strategic ($466 vs $478) and
  Tactical ($46 vs $54), and is beaten by sequence/graph models at the
  short horizons. Its four regime experts (normal/GDP/GS/MIT) can't
  specialize without regime labels.
- **Structural breaks**: COVID was excluded rather than modeled
  (including it made no consistent difference, Planning sensitivity
  check). The 2025 rise in GDP frequency was under-predicted at Planning
  (37% vs 44%), mildly at Strategic (5.1–5.6% vs 6.2%), not at Tactical.

### Objective 4 — operational validation

Scope: the full protocol (six-metric KPI suite, temporal blocked CV,
stress tests, shadow mode) was applied at **Planning only**. Strategic and
Tactical were evaluated on cost, calibration, within-airport and (Tactical)
onset metrics, plus validation sensitivity checks. Blocked CV's "regime
stratification" is a positive/negative balance check — real regime labels
don't exist.

- **Cost-based evaluation changes conclusions**: at Planning, the 0.5
  threshold costs more than always flagging; pooled AUC (~0.91) is mostly
  airport identity (a lookup table scores 0.90) — within-airport AUC
  (~0.65) is the real skill measure.
- **Temporal blocked CV** overturned a single-year conclusion (recent
  history at Planning) and was used to fix the probability floor.
- **Validation-tuned thresholds don't transfer**: on Planning test they
  raised cost for most models; the fixed cost-ratio rule held.
- **KPI suite on trained models (Planning)**: false-alarm cost and
  cost-weighted recall (in the decision cost), calibration error (0.04 for
  the main model on test), forecast stability (daily |Δp| < 0.005;
  decisions flip on 0.12% of airport-days), lead-time credit (~0.32 for
  every model — set by where GDPs fall in the window, not by the model),
  regime-transition F1 (below).
- **Regime transitions**: no Planning model forecasts transitions
  (regime-transition F1 ≤ 0.044). At Tactical, 70% of positive hours are
  continuations and 80% of onsets are visible from held arrivals; on
  **clean onsets** (no held arrivals yet, 0.14% of hours) the best model
  reaches PR-AUC 0.22 (0.10 without network EDCT visibility).
- **Stress tests** (Planning main model): fails visibly on NaN, stays
  valid under extreme inputs; worst-case perturbations of 0.1 SD raise
  cost 6–15%; a recent-history feed outage costs < 2%.
- **Shadow mode**: chronological replay of the Planning test window
  reproduces the direct evaluation exactly (live backlog 2,250
  predictions). Replay only — no live feed exists.

### Section 5 baselines (Seq2Seq, TFT, SARIMA) and simple references

| Tier | Logistic | SARIMA | Seq2Seq | TFT | Main model |
|---|---|---|---|---|---|
| Planning | **$617** | $726 | $641 | $637 | $620 |
| Strategic | $478 | $496 | $469 | $497 | **$464** |
| Tactical | $54 | — | $40.4 | $43.6 | **$36.3** |

Logistic regression with airport identity is competitive at Planning and
within ~3% of the best at Strategic; it should be reported as a baseline.
SARIMA is the weakest learned model where run; it was not run hourly.

## Limitations to state

- GDP-only EDCT proxy; no Ground Stop or Miles-in-Trail labels.
- Thresholds (10% daily, 50% hourly) and FAOC durations are modeling
  choices. Sensitivity checks on validation — daily 5%/20% (Planning,
  Strategic), hourly 30%/70% (Tactical), and COVID kept in train (all
  three) — changed no conclusion: the graph effect (none at Planning and
  Strategic; large at Tactical in every setting, −6% to −21% cost), the
  FAOC comparison and the model ranking all held. Tables in each tier's
  write-up; Strategic/Tactical rows in `models/results/short_horizon_sensitivity.csv`.
- Observed weather only; no forecasts.
- One test window (2024–2025), and it is now spent.
- Shadow mode is a historical replay, and like the stress tests and full
  KPI suite it was run at Planning only.
- The three tiers are separate models; the paper's "hierarchical"
  framework has no coupling between tiers yet.
- The Tactical clean-onset subset defines "before a program exists" as
  "no arrival held yet"; without ASPM Advisories the true issuance time is
  unknown, so some clean onsets may follow an issuance whose first held
  flights hadn't landed.
