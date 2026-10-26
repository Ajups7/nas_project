# Planning tier: training results and decisions

Planning tier (1–30 day horizon), trained 2026-10-21/22. Validation window:
2023-01-31 → 2024-01-30 (27,375 airport-days, 34.4% positive). Test window:
2024-03-01 → 2026-01-01 as-of dates (46,995 airport-days, 41.3% positive),
**evaluated once** on 2026-10-22 after the choices below were frozen.

## Final results: test set

Frozen configuration (see "Frozen before the test evaluation"): 10% GDP-day
label, 75 airports, COVID excluded from train, 12 features + one-hot airport
identity. Neural models: mean over 3 seeds (± = seed std). Cost uses the
rule fixed in advance (cost-ratio threshold; 0.5 for FAOC).

| Model | Cost ($/airport-day) | Log loss | Within-airport AUC | Pooled AUC | ECE |
|---|---|---|---|---|---|
| *Never flag* | 7,426 | | | | |
| *Always flag* | 867 | | | | |
| Climatology lookup | 764 | 0.549 | 0.647 | 0.908 | 0.049 |
| **Logistic regression** | **617** | **0.343** | 0.659 | 0.928 | 0.038 |
| **PlanningDisruptNet, standard (main)** | **620 ± 4** | 0.351 | 0.652 ± 0.026 | 0.924 | 0.041 |
| PlanningDisruptNet, FAOC | 648 ± 3 | 0.622 | 0.620 | 0.922 | 0.263 |
| Seq2Seq | 641 ± 15 | 0.354 | 0.667 | 0.924 | 0.040 |
| TFT | 637 ± 21 | 0.358 | 0.639 | 0.925 | 0.051 |
| Regime Transformer, top-10 rotation graph | 734 ± 93 | 0.366 | 0.658 | 0.918 | 0.037 |
| Regime Transformer, full graph | 630 ± 42 | 0.352 | 0.667 | 0.925 | 0.047 |
| Regime Transformer, no graph | 637 ± 33 | 0.348 | **0.687** | 0.928 | 0.045 |
| SARIMA | 726 | 0.484 | 0.654 | 0.913 | 0.059 |

What the test confirms:

1. **The main model cuts decision cost 28% vs always-flagging** ($620 vs
   $867) and 92% vs never-flagging, with calibrated probabilities.
2. **Logistic regression with airport identity ties it** ($617, best log
   loss). Every neural model lands within ~$20 of it, inside seed noise.
   At this horizon the inputs (airport identity, climatology, recent
   level) carry the result; architecture barely matters.
3. **FAOC-Loss is worse than standard loss + cost-ratio threshold** ($648
   vs $620; seed stds 3–4) — consistent with validation and with every
   sensitivity variant.
4. **The rotation graph doesn't help** at 1–30 days: top-10 graph is the
   worst neural model on cost and the least stable ($734 ± 93); no graph
   has the best within-airport AUC (0.687).
5. **TFT's validation lead didn't hold** ($637 vs $620 on test).
6. **Drift, as predicted**: every model under-predicts on test (mean
   prediction ~0.37 vs 0.413 actual; 2025: ~0.38 vs 0.443).
7. **Validation-tuned thresholds don't transfer**: they made test cost
   *worse* for most models (DisruptNet $620 → $637, TFT $637 → $674). The
   fixed cost-ratio rule generalizes better than a threshold fitted to one
   validation year.

Full per-seed numbers, 2024/2025 breakdowns and row-level predictions:
`results/test_results_per_seed.csv`, `results/test_results_summary.csv`,
`results/test_predictions.parquet`.

## Operational validation on test (Objective 4, `operational_validation.py`)

Frozen models only; nothing here feeds back into any model choice.

### Remaining KPIs (seed-averaged predictions, fixed decision rule)

| Model | Regime-transition F1 | Forecast stability (mean daily \|Δp\|) | Decision flip rate | Lead-time credit |
|---|---|---|---|---|
| Climatology lookup | 0.006 | 0.0036 | 0.51% | 0.316 |
| Logistic regression | 0.012 | 0.0049 | 0.20% | 0.319 |
| **PlanningDisruptNet, standard** | 0.038 | 0.0045 | 0.12% | 0.318 |
| PlanningDisruptNet, FAOC | 0.029 | 0.0013 | 0.10% | 0.318 |
| Seq2Seq | 0.029 | 0.0042 | 0.09% | 0.317 |
| TFT | 0.004 | 0.0022 | 0.14% | 0.317 |
| Regime Transformer, top-10 | 0.031 | 0.0064 | 0.66% | 0.318 |
| Regime Transformer, full | 0.044 | 0.0060 | 0.28% | 0.317 |
| Regime Transformer, no graph | 0.030 | 0.0059 | 0.31% | 0.318 |
| SARIMA | 0.032 | 0.0167 | 0.67% | 0.317 |

- **No model forecasts regime transitions at this horizon** (F1 ≤ 0.044):
  they track each airport's level, not when it flips into or out of a
  GDP-likely state. Consistent with the ~0.65 within-airport ceiling.
  Onset timing is a job for the Tactical/Strategic tiers.
- **Guidance is very stable**: daily probability changes < 0.5 points
  for the main model, decisions flip on 0.12% of airport-days. SARIMA is
  the least stable.
- **Lead-time credit doesn't separate models** (~0.32; caught windows'
  first GDP day is ~9.5 days out): it's set by where GDPs fall in the
  window, not by the model.
- Lead-time credit here = days to the first GDP day in the window / 30.
  Forecast stability is adapted: the KPI is defined for repeated forecasts
  of one target date; the Planning label is a rolling window, so
  consecutive as-of dates are compared instead.

### Stress tests (main model, 3 seeds; test-window inputs)

| Scenario | Effect |
|---|---|
| Extreme magnitudes (×1e6, ×1e-6), all-zero input, single example | Outputs finite and in [0, 1] for every seed |
| NaN in one input | NaN appears in that example's output only — fails visibly, not silently |
| FGSM, ε = 0.05 SD on the 12 real features | ~1% of decisions flip; cost $620 → $635–666 |
| FGSM, ε = 0.1 SD | 2–4% flip; cost $656–711 |
| FGSM, ε = 0.25 SD | 7–9% flip; cost $772–840 |
| FGSM, ε = 0.5 SD | 15–16% flip; cost $968–1,020 (worse than always-flag, $867) |
| Recent-history feed outage (rates → climatology) | 0.4–0.5% flip; cost +$1–7 — degrades gracefully |
| Persistent-GDP spike (recent rates = 1 everywhere) | Everything flagged; cost = always-flag ($867) |

FGSM perturbs only the 12 real features (airport identity masked), in
standardized units. Worst-case perturbations of a tenth of a standard
deviation already cost 6–15%; a feed outage barely matters because the
model leans on climatology and airport identity.

### Shadow mode (main model, seed 0)

Chronological replay of the 670 test days: each day's predictions are
recorded, and outcomes are revealed only once their 30-day window has
elapsed. All 46,995 predictions resolve; the live backlog holds steady at
2,250 (75 airports × 30 days). The replayed log reproduces the direct test
evaluation exactly (log loss 0.346, within-airport AUC 0.672 for seed 0).
This is a replay of history, not a live deployment — no real-time feed
exists for this project.

## Validation, final configuration (with airport identity)

| Model | Cost | Log loss | Within-airport AUC | ECE |
|---|---|---|---|---|
| PlanningDisruptNet, standard | 655 ± 16 | **0.353** | 0.648 | **0.022** |
| PlanningDisruptNet, FAOC (0.5) | 677 ± 14 | 0.756 | 0.620 | 0.322 |
| Seq2Seq | 668 ± 20 | 0.359 | 0.650 | 0.029 |
| TFT | **641 ± 5** | 0.365 | 0.646 | 0.031 |
| Regime Transformer, top-10 graph | 674 ± 19 | 0.365 | 0.624 | 0.029 |
| Regime Transformer, full graph | 667 ± 8 | 0.357 | 0.597 | 0.020 |
| Regime Transformer, no graph | 663 ± 12 | 0.360 | **0.660** | 0.025 |

## First pass (before airport identity)

Everything from here down is the decision history, in the order it
happened. Numbers in this section are validation, without airport identity.

| Model | Pooled AUC | Within-airport AUC | Log loss ↓ | Brier ↓ | ECE ↓ | Decision cost ↓ ($/airport-day) |
|---|---|---|---|---|---|---|
| *Never flag* | | | | | | 6,189 |
| *Always flag* | | | | | | 968 |
| Climatology lookup | 0.899 | 0.651 | 0.534 | 0.120 | 0.060 | – |
| Logistic regression | 0.913 | 0.631 | 0.395 | 0.112 | 0.052 | – |
| **PlanningDisruptNet, standard loss** | 0.911 | 0.650 | 0.357 | 0.110 | 0.031 | 714 |
| PlanningDisruptNet, FAOC-Loss | 0.907 | 0.641 | 0.735 | 0.218 | 0.297 | 716 (at 0.5) |
| Seq2Seq | 0.908 | 0.632 | 0.361 | 0.110 | 0.028 | 714 |
| **TFT (simplified)** | 0.912 | 0.644 | **0.357** | **0.110** | **0.023** | **656** |
| SARIMA (per airport) | 0.909 | **0.658** | 0.566 | 0.117 | 0.081 | 702 |

Neural models: mean of 3 seeds. Decision cost uses the cost-ratio threshold
(flag if p ≥ fp / (fp + fn × multiplier), ≈ 6–9%) for every model except
FAOC, which is evaluated at 0.5 (see "Standard vs FAOC-Loss"). Logistic and
climatology lookup were scored before the cost metrics existed.

What the table says:

1. **Pooled AUC (~0.91) is mostly airport identity.** A lookup table of
   "this airport's label rate in this month" already scores 0.899.
2. **Within-airport skill tops out at ~0.63–0.67** for every model, in every
   year tested (see "Feature set"). At a 1–30 day horizon, these features
   predict *which airport* and *what season*, and only weakly *which
   stretch* of the year.
3. **The neural models' gain is probability quality**, not ranking: log
   loss 0.357 vs 0.395 for logistic regression, ECE 0.02–0.03 vs 0.05.
4. **FAOC-Loss matches, but does not beat, the standard loss + a
   cost-ratio threshold** — and destroys calibration doing it.
5. **TFT, a baseline, has the lowest decision cost.** Cause:
   PlanningDisruptNet can't predict below ~5% at quiet airports (see
   "TFT vs DisruptNet").

## How to reproduce

```bash
cd nas_project && source .venv/bin/activate
python -m common.aspm_cache edct                          # once, ~15 min: Parquet cache of all EDCT months
python -m features.planning                          # ~45 s: data/cache/planning_dataset.parquet
python -m models.planning.train_reference            # climatology lookup, logistic regression
python -m models.planning.train_disruptnet standard  # pos_weight and recent-history sweep
python -m models.planning.train_disruptnet anomaly   # recent-anomaly features (negative result)
python -m models.planning.feature_selection_cv       # blocked CV over feature sets, ~5 min
python -m models.planning.train_disruptnet faoc      # standard vs FAOC, same architecture
python -m models.planning.compare_losses             # decision costs for standard vs FAOC
python -m models.planning.train_sequence_baselines   # Seq2Seq, TFT, ~8 min
python -m models.planning.train_sarima               # SARIMA, ~15 min
python -m models.planning.investigate_tft_cost       # TFT vs DisruptNet breakdown
python -m models.planning.fix_probability_floor      # probability-floor fixes, 4 years, ~25 min
python -m models.planning.train_disruptnet final     # frozen config: + airport identity
python -m models.planning.train_sequence_baselines --airport-onehot
python -m models.planning.rotation_adjacency         # once, ~3.5 min: 2019 rotation graph
python -m models.planning.train_regime_transformer top10 full none --airport-onehot   # ~25 min
python -m models.planning.train_sarima --with-test   # also saves test forecasts
python -m models.planning.sensitivity none           # ~1.5 h
python -m models.planning.evaluate_test              # ONCE; refuses to re-run
python -m models.planning.operational_validation     # KPIs, stress tests, shadow mode on frozen models
```

Results land in `models/planning/results/`, checkpoints in
`models/planning/checkpoints/`, logs in `logs/`.

Training is fast — seconds per model for PlanningDisruptNet (1,848
parameters), 1–3 minutes for Seq2Seq/TFT (~10.5k parameters) — on ~150k
training rows × 12 features, on the VM's CPU. No GPU needed for this tier.

## Decisions, in order

### 1. Label: GDP day = ≥ 10% of arrivals had an EDCT

The original label ("any EDCT, arrival or departure, in the next 30 days")
was 90% positive; 65 of 82 airports were ≥ 95% positive. Departure EDCTs
reflect GDPs at *destinations*. Redefined in `common/disruption.py`, shared
with the Strategic label. 10% sits at the trough of the daily
arrival-EDCT-share distribution and is where days start showing multi-hour
held arrivals — there is no published source for it. Full evidence:
`features/planning_label_validation.md`. 5% and 20% are pending sensitivity
checks.

### 2. Airports: 75 of 82

Excluded APA, ASE, BJC, BOI, CMH (in ASPM only from mid-2023) and OXR, GYY
(patchy arrival data biased their labels). Rule: known GDP-day value on
≥ 90% of train days (`eligible_airports()`).

### 3. Features: 12, climatology from train years only

Calendar (2), weather climatology (7), GDP-day climatology (1) — all
describing the 30-day forecast window — plus recent GDP-day rate over the
past 7 and 30 days (2). Climatology uses train dates only (a leakage test
confirms). Details: `features/planning_features_validation.md`.

### 4. COVID excluded from training (2020-03-01 → 2021-06-30)

2020 label prevalence was 14% against 36–38% in 2016–2019. On validation
the two choices differed by ~0.01 on every metric (mixed direction), so the
decision rests on the domain argument: validation and test are normal
operations. Removes 19% of train rows. "COVID included" is a pending
sensitivity check.

| Logistic regression, all features | Pooled AUC | Within-airport AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| All train | 0.906 | 0.616 | 0.387 | 0.117 | 0.069 |
| COVID excluded | 0.913 | 0.631 | 0.395 | 0.112 | 0.052 |

### 5. Loss weighting: plain BCE (`pos_weight = 1`)

Balancing weight (1.78) gave no ranking gain and doubled calibration error
(mean prediction 41% vs actual 34%).

| PlanningDisruptNet | Within-airport AUC | Log loss | ECE | Mean prediction |
|---|---|---|---|---|
| `pos_weight = 1` | 0.650 ± 0.013 | 0.357 | 0.031 | 35.1% |
| `pos_weight = 1.78` | 0.645 ± 0.019 | 0.381 | 0.067 | 41.1% |

### 6. Feature set: keep raw recent history

On 2023 alone, recent history improved pooled metrics but *lowered*
within-airport AUC, and a "recent anomaly" variant (recent rate minus the
airport's normal rate) did worse still. Within-airport, recent history
predicted the label in the train years (AUC 0.61) but reversed in 2023
(anomaly AUC 0.43).

Walk-forward blocked CV (`feature_selection_cv.py`; climatology rebuilt per
fold; COVID removed) over 2018, 2019 and 2021–22, plus 2023:

| PlanningDisruptNet log loss | 2018 | 2019 | 2021–22 | 2023 |
|---|---|---|---|---|
| no recent history | 0.441 | 0.380 | 0.498 | 0.375 |
| **raw recent history** | **0.414** | **0.364** | **0.434** | **0.357** |
| recent anomaly | 0.448 | 0.373 | 0.439 | 0.364 |

| Within-airport AUC | 2018 | 2019 | 2021–22 | 2023 |
|---|---|---|---|---|
| Logistic, no recent | 0.631 | 0.670 | 0.631 | 0.660 |
| Logistic, raw recent | 0.630 | 0.653 | 0.623 | 0.631 |
| DisruptNet, no recent | 0.560 | 0.647 | 0.632 | 0.673 |
| DisruptNet, raw recent | 0.548 | 0.649 | 0.588 | 0.650 |

The trade-off holds every year: raw recent history gives better
probabilities (log loss −0.02 to −0.06); leaving it out gives slightly
better within-airport ranking (0.00 to +0.04). Interpretation: recent
history tracks each airport's **current level**, which drifts year to year
(ATL: 59% → 80% positive from 2024 to 2025); it doesn't help with timing
within a year. Kept for probability quality. The anomaly variant is a
documented negative result.

The CV also showed PlanningDisruptNet ranking *worse* within airports than
logistic regression (~0.61 vs ~0.64), while having much better log loss.

### 7. Early stopping on each model's own objective

Validation BCE for the standard loss; validation FAOC-Loss for FAOC (FAOC
is deliberately miscalibrated, so judging it by BCE would penalize the
point of it).

## Standard vs FAOC-Loss

Same architecture, features, and seeds; only the loss differs. FAOC weights
a miss at $11,809 × the airport's delay-propagation multiplier and a false
alarm at $1,476 (placeholder durations: 2 h and 0.25 h,
`losses/cost_figures.md` §4), so the cost-optimal threshold for a
*calibrated* model is 5.9–8.7%.

| Model, decision rule | Cost ($/airport-day) | Flag rate | Cost-weighted recall |
|---|---|---|---|
| Standard, 0.5 | 1,524 | 33% | 0.77 |
| **Standard, cost-ratio threshold** | **714** | 67% | 0.97 |
| Standard, best tuned threshold | 701 ± 16 | 61% | 0.95 |
| **FAOC, 0.5** | **716** | 60% | 0.95 |
| FAOC, best tuned threshold | 696 ± 9 | 64% | 0.96 |
| FAOC, cost-ratio threshold | 940 | 98% | 1.00 |

- **FAOC-Loss ≈ standard loss + cost-ratio threshold** ($716 vs $714; with
  tuned thresholds, $696 vs $701, within seed variation). Training on the
  cost ratio moves the decision threshold *into* the model — what decision
  theory predicts for a reweighted loss.
- **The price is calibration**: FAOC's mean prediction is 64% against an
  actual 34% (ECE 0.30 vs 0.03), so its probabilities can't be read as
  probabilities.
- **The naive 0.5 threshold is the real mistake** ($1,524, worse than
  always flagging). Both cost-aware approaches beat always-flag by ~27%.
- Don't apply the cost-ratio threshold to FAOC — that counts the cost twice.

Suggested framing for the paper: *cost-sensitive training and calibrated
training with a cost-derived threshold reach the same operational cost; the
latter keeps calibrated probabilities.* The dollar figures depend on the
placeholder durations; the equivalence doesn't.

**Choice: standard loss + cost-ratio threshold** as the main Planning
model; FAOC reported as an equivalent-cost alternative.

## The paper's baselines

### Seq2Seq and TFT

Inputs: the past 30 days ending on the as-of date, each day = the 12
features + that day's GDP-day indicator (`models/planning/sequences.py`).
Same label, split, COVID exclusion and metrics. The first 29 days of 2016
have no full window, so train has 149,854 rows (vs 151,963); validation rows
are identical. Seq2Seq's best epoch was always the first; TFT's 1–6.

| | Within-airport AUC | Log loss | ECE | Cost |
|---|---|---|---|---|
| Seq2Seq | 0.632 ± 0.008 | 0.361 | 0.028 | 714 ± 20 |
| TFT | 0.644 ± 0.013 | 0.357 | 0.023 | 656 ± 11 |

### SARIMA

One model per airport on the daily GDP-day series, with monthly GDP-day
climatology as an exogenous input (for the annual cycle); order (1,0,1),
seasonal (1,0,1,7); COVID days and missing months as gaps. 30-day
forecasts from every validation origin come from one filter pass (checked
against statsmodels' `get_prediction`: max difference 0.0). P(any GDP day
in 30) = 1 − ∏(1 − clip(forecast)), which assumes independent days — hence
the poor calibration (mean prediction 41%).

- Best within-airport AUC of any model (0.658), worst calibration of the
  well-behaved models (ECE 0.081).
- MLE did not converge for 12 airports (ANC, CLE, CVG, FLL, HNL, MSY, OAK,
  OGG, PIT, SJC, SMF, STL) — mostly airports with few or no GDP days, where
  the series is near-constant. Not yet investigated per airport.

## TFT vs DisruptNet: why the baseline wins on cost

`investigate_tft_cost.py`, seed-averaged predictions:

- Gap $61/airport-day ($713 vs $652): **$48 from false alarms, $13 from
  misses.** TFT flags fewer days (63% vs 66%) and still misses less.
- 1,777 rows are flagged by DisruptNet only — **3.7% of them were real
  GDPs**, far below the ~7% threshold where flagging pays. 907 rows are
  flagged by TFT only, 9.4% real.
- **The entire gap comes from airports below 30% prevalence**; busy
  airports contribute $0. CVG had *no* GDP-labeled days in validation:
  DisruptNet flagged 76% of its days, TFT none.
- **Cause: PlanningDisruptNet has a probability floor of ~5%.** At CVG (0
  positives in train *and* validation) it never predicts below 5.0% and
  averages 8.4%, in train as well as validation — so it's a fit problem,
  not 2023 drift. That floor sits right at the decision threshold, so quiet
  airports get flagged. TFT puts 3,656 validation rows below 2%;
  DisruptNet 735.
- **It is not extra information**: the rows DisruptNet flags and TFT
  doesn't have a recent 30-day GDP rate of essentially 0 (mean 0.005; 86%
  exactly 0), which DisruptNet also sees.
- Why log loss doesn't catch it: predicting 5% instead of 1% on a negative
  row costs only 0.04 in log loss, but flips a decision at a ~7% threshold.
  Log loss is insensitive exactly where the cost metric is most sensitive.

### Fix: airport identity (`fix_probability_floor.py`)

Four fixes, 3 seeds each, scored on 2023 and the three CV years:

| Variant | Cost, mean of 4 years | Log loss | ECE | Mean p, quiet airports |
|---|---|---|---|---|
| none (base) | 806 | 0.393 | 0.045 | 9.1% |
| longer training (lr 3e-4, patience 20) | 809 | 0.393 | 0.044 | 8.7% |
| isotonic recalibration on train | 825 | 0.399 | 0.045 | 5.5% |
| log-scaled GDP rates | 768 | 0.379 | 0.036 | 7.3% |
| **one-hot airport identity** | **685** | **0.368** | **0.035** | **2.5%** |

Airport identity wins in all four years (2018: $881 → $663; 2023: $714 →
$655); within-airport AUC unchanged on average. Longer training does
nothing, which confirms the cause: quiet airports were indistinguishable
in the features, not under-trained. Airport identity was then given to
**every** neural model and to logistic regression, so comparisons stay
fair. With it, PlanningDisruptNet's best validation epoch is the first
one: it learns airport base rates at once and then overfits train-period
levels that drift by 2023.

## Regime Transformer (`train_regime_transformer.py`)

One sample = one date, all 75 airports: each airport's past-30-day window
(same inputs as TFT, + airport identity) → LSTM → attention over
rotation-chain neighbors → attention over 4 learned regime embeddings →
one probability per airport. Unlabeled airport-dates are masked, so it is
scored on exactly the same rows as every other model.

Graph: Person B's `features/rotation_graph.py` over the 75 airports, all of
2019 BTS (`rotation_adjacency.py`). 48% of airport pairs have direct
flights (median 37–40 partners), so three variants were trained:
**top-10** (each airport's 10 strongest rotation partners, symmetrized:
588 edges, median 11 neighbors, hubs up to 54), **full** (any flights,
1,418 edges), **none** (self only). TEB and VNY have no airline flights.

Result (validation and test): **no graph is as good as or better than
either rotation graph**; top-10 is the worst and least stable on test.
Plausible reason: GDPs propagate along rotations within hours, so
rotation structure should matter for the Tactical/Strategic horizons, not
1–30 days. The graph was selected for the sensitivity checks by a rule
fixed before seeing results (lowest mean validation cost → "none";
`logs/sensitivity_graph_choice.txt`).

## Sensitivity (`sensitivity.py`, validation, 3 seeds)

Cost ($/airport-day). Costs are not comparable *across* label thresholds
(prevalence differs).

| Variant (validation prevalence) | Logistic | DisruptNet std | DisruptNet FAOC | TFT | Regime Tr. (no graph) |
|---|---|---|---|---|---|
| 5% threshold (40.5%) | **772** | 803 | 807 | 778 | 794 |
| 10%, main (34.4%) | – | 655 | 677 | **641** | 663 |
| 20% threshold (28.8%) | 624 | **609** | 627 | 633 | 671 |
| 10%, COVID included (34.4%) | 647 | 645 | 650 | 659 | **631** |

- FAOC never beats the standard loss (ties at 5% and COVID-included,
  worse at 10% and 20%).
- No architecture wins consistently; gaps are ~$10–60, comparable to seed
  noise, and logistic regression is competitive everywhere.
- COVID exclusion has no consistent effect (some models slightly better
  with COVID kept, some worse); it changes no conclusion.
- Within-airport AUC is highest at the 5% threshold (0.65–0.70) — more,
  milder GDP days are easier to time than rare severe ones.

## Frozen before the test evaluation (written 2026-10-22, before `evaluate_test.py` ran)

- Label: GDP day = ≥ 10% of arrivals with an EDCT; label = any GDP day in
  the next 1–30 days. 75 airports. COVID excluded from train.
- Inputs: the 12 Planning features + one-hot airport identity (all neural
  models and logistic regression).
- **Main model: PlanningDisruptNet, standard loss (`pos_weight = 1`),
  cost-ratio decision threshold**, checkpoints `final_standard_seed{0,1,2}`.
- Comparators: climatology lookup; logistic regression; PlanningDisruptNet
  FAOC (threshold 0.5); Seq2Seq, TFT (`*_onehot`); Regime Transformer with
  top-10 / full / no graph (`*_onehot`); SARIMA.
- Decision rules: fixed in advance (cost-ratio threshold; 0.5 for FAOC),
  plus a secondary threshold tuned on **validation** and applied to test.
- Nothing is retrained or tuned on test; `evaluate_test.py` refuses a second run.

## Caveats

- "Tuned threshold" costs on validation are tuned and scored on the same
  year — optimistic; on test, validation-tuned thresholds did worse than
  the fixed rule.
- Dollar costs inherit FAOC-Loss's placeholder durations.
- The label is a GDP proxy (no Ground Stops or Miles-in-Trail).
- The test split is now spent: further model changes need a new
  evaluation window (or must be reported as post-hoc).

## Not done yet

- SARIMA non-convergence at 12 low-GDP airports (ANC, CLE, CVG, FLL, HNL,
  MSY, OAK, OGG, PIT, SJC, SMF, STL) not investigated per airport.
- Model comparison across CV years was done for feature sets and the
  probability-floor fix, not for every architecture.
- Drift: all models under-predict on 2025. A rolling retrain (e.g.
  including 2023 in training) is the natural next experiment — on a
  fresh evaluation window, since this test set is now spent.
- `PLAN.md` still lists Phase 2 as not started and ASPM Advisories/weather
  as pending; it predates the EDCT-proxy pivot and this training.
- Nothing from this work is committed to git yet.
