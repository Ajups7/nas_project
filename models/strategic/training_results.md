# Strategic tier: training results and decisions

Strategic tier (2–24 h): at the end of day D, predict whether D+1 is a GDP
day (≥ 10% of arrivals with an EDCT — `common/disruption.py`). Trained
2026-10-22. Same protocol as the Planning tier
(`models/planning/training_results.md`): 75 airports, COVID excluded from
train, one-hot airport identity, 3 seeds, early stopping on validation,
cost-ratio decision threshold (FAOC: 0.5).

## Final results: test set (evaluated once, 2026-10-22)

Test: 50,314 airport-days, 5.6% positive (2024: 5.1%, 2025: 6.2%).
Neural models: mean of 3 seeds (± seed std). Cost at the rule fixed in
advance (cost-ratio threshold; FAOC 0.5).

| Model | Cost ($/airport-day) | Within-airport AUC | Pooled AUC | Log loss | ECE |
|---|---|---|---|---|---|
| *Never flag* | 995 | | | | |
| *Always flag* | 1,394 | | | | |
| Climatology lookup | 593 | 0.586 | 0.844 | 0.181 | 0.012 |
| Persistence | 547 | 0.565 | 0.866 | 0.162 | 0.010 |
| SARIMA | 496 | 0.676 | 0.874 | 0.211 | 0.011 |
| TFT | 497 ± 7 | 0.661 | 0.885 | 0.153 | 0.007 |
| Logistic regression | 478 | 0.690 | 0.887 | 0.151 | 0.004 |
| Seq2Seq | 469 ± 7 | 0.691 | 0.892 | 0.149 | 0.004 |
| DisruptNet (MoE), standard | 466 ± 7 | 0.691 | 0.893 | 0.149 | 0.005 |
| DisruptNet (MoE), FAOC, at 0.5 | 459 ± 2 | 0.695 | 0.893 | 0.330 | 0.173 |
| STGNN, full graph | 467 ± 8 | 0.710 | 0.893 | 0.149 | 0.006 |
| **STGNN, top-10 rotation graph (main)** | **464 ± 1** | **0.717** | 0.893 | 0.150 | 0.008 |
| STGNN, no graph | 464 ± 13 | 0.716 | **0.896** | **0.148** | 0.007 |

1. **Learned models cut decision cost ~53% vs the best trivial policy**
   (never flag, $995) and ~15% vs persistence ($547). At a 4–6% event rate,
   always-flagging is the *worst* policy ($1,394).
2. **The STGNN architecture ranks within airports best** (0.71–0.72 vs
   0.69 for DisruptNet, logistic, Seq2Seq), confirming validation.
3. **The rotation graph adds nothing on test**: top-10 graph and no graph
   are identical ($464, 0.717 vs 0.716); the full graph is marginally worse.
4. **FAOC ≈ standard loss** ($459 vs $466, ~1 seed std apart), with
   probabilities inflated 4× (23% predicted vs 5.6% actual).
5. **Logistic regression is within ~$15 of the best neural models.**
6. **Drift is milder than at the Planning horizon**: 2025 predictions run
   5.1–5.6% against 6.2% actual. Validation-tuned thresholds changed test
   cost by only a few dollars.

Per-seed results and row-level predictions: `results/test_results_per_seed.csv`,
`results/test_results_summary.csv`, `results/test_predictions.parquet`.

## Decisions

### Features fixed for timing (`features/strategic.py`)

Person B's original features put day D−1's actuals in row D, while the
label (via `build_planning_labels(horizon_days=1)`) is D+1 — a one-day gap
that ignored day D's actuals, complete by the end of D. Revised so row D =
what is known at the end of D. `validate_strategic_features.py` checks it:
row D's operations equal day D's (not D−1's), tomorrow's schedule equals
the next row's, the label equals D+1's GDP flag.

Also changed: disruption features use the arrival-based GDP-day definition;
added D+1's scheduled arrivals/departures (published in advance — demand
vs. capacity is what triggers a GDP), target-day day-of-week, daily weather
extremes (min visibility, max wind/gust, thunderstorm share), and D+1's
train-period GDP climatology. Data comes from the Parquet caches
(`common/aspm_cache.py`). Missing gusts = no gust reported → set to max
wind; other gaps → train-period medians.

### Label and data

| Split | Rows | Prevalence |
|---|---|---|
| train (COVID excluded) | 152,875 | 5.0% |
| validation | 27,374 | 4.2% |
| test | 50,314 | 5.6% |

GDP days are rare (4–6% of airport-days), matching the paper's premise. The
label equals today's GDP flag on 94% of rows (mostly both 0).

Strongest single features (correlation with the label, train): GDP rate
over the past 7 days 0.42, target-day GDP climatology 0.41, today's arrival
EDCT share 0.35, today's taxi-out time/delay 0.23–0.25, tomorrow's
scheduled volume 0.20.

## Validation results

| Model | Within-airport AUC | Pooled AUC | Log loss | ECE | Cost ($/airport-day) |
|---|---|---|---|---|---|
| Persistence (P(label \| airport, GDP today)) | 0.545 | 0.863 | 0.136 | 0.010 | 513 |
| Climatology lookup (airport × target month) | 0.598 | 0.851 | 0.146 | 0.015 | 485 |
| SARIMA (1-day-ahead) | 0.610 | 0.879 | 0.203 | 0.010 | 433 |
| TFT | 0.639 ± 0.015 | 0.888 | 0.127 | 0.006 | 426 ± 5 |
| Seq2Seq | 0.670 ± 0.006 | 0.896 | 0.123 | 0.004 | 399 ± 10 |
| Logistic regression | 0.681 | 0.894 | 0.125 | 0.005 | 398 |
| DisruptNet (MoE), standard | 0.679 ± 0.020 | 0.899 | 0.123 | 0.005 | 389 ± 5 |
| DisruptNet (MoE), FAOC, at 0.5 | 0.688 ± 0.014 | 0.901 | 0.301 | 0.167 | 384 ± 2 |
| STGNN, full graph | 0.681 ± 0.010 | 0.895 | 0.123 | 0.005 | 394 ± 2 |
| STGNN, top-10 rotation graph | 0.699 ± 0.026 | 0.898 | 0.122 | 0.003 | 390 ± 2 |
| **STGNN, no graph** | **0.717 ± 0.024** | **0.901** | **0.122** | 0.006 | **386 ± 3** |

- **The rotation graph does not help at 2–24 h either**: no graph >
  top-10 > full graph on within-airport AUC and cost. The STGNN's
  interleaved day-by-day recurrent update is the best architecture for
  ranking within airports — without the graph.
- **FAOC ≈ standard loss on cost** ($384 vs $389), with probabilities
  inflated ~5× (21% predicted vs 4.2% actual).
- **Learned models beat persistence and climatology by ~20% on cost**;
  among learned models, cost differences are ≤ $40 and mostly within seed
  noise. Logistic regression is again competitive ($398).
- Within-airport skill is higher than at the Planning horizon (best 0.72
  vs ~0.67): next-day timing is partly predictable.

## Sensitivity (validation only, `models/short_horizon_sensitivity.py`)

Run after the test evaluation; validation only, nothing selected from it.
Cost per airport-day; not comparable across thresholds (the label changes).

| Setting (validation positive rate) | Logistic | DisruptNet std | DisruptNet FAOC | STGNN top-10 | STGNN no graph |
|---|---|---|---|---|---|
| GDP day ≥ 5% (5.0%) | 459 | 450 ± 4 | 445 ± 2 | 453 ± 4 | **445 ± 2** |
| **≥ 10%, main** (4.2%) | 398 | 389 ± 5 | 384 ± 2 | 390 ± 2 | 386 ± 3 |
| ≥ 20% (3.2%) | 323 | 322 ± 3 | **315 ± 2** | 319 ± 4 | 320 ± 6 |
| ≥ 10%, COVID kept in train (4.2%) | 394 | 388 ± 9 | 384 ± 8 | 386 ± 9 | 392 ± 12 |

- The rotation graph has no consistent effect: no graph wins at 5% and
  at the main setting, ties at 20%, loses with COVID kept — all within
  seed noise.
- FAOC is cheaper than the standard loss in every setting, by $4–7
  (1–2%), and on test ($459 vs $466) — a small, consistent edge at this
  tier, with probabilities still inflated ~5×.
- Logistic regression stays within $5–15 of the neural models.
- Keeping COVID in train changes nothing material.

## Frozen before the test evaluation (written 2026-10-22, before `evaluate_test.py` ran)

- Label, airports, COVID exclusion, features and airport identity as above.
- Every model in the table is evaluated on test from the predictions it
  wrote at training time (`results/predictions/test_*.parquet`, sealed —
  not scored until now). Nothing is retrained or re-selected.
- Main model for the narrative: **STGNN with the top-10 rotation graph**
  (the paper's Strategic architecture as designed); the no-graph and
  full-graph variants are its ablation.
- Decision rules: cost-ratio threshold (FAOC: 0.5), plus a threshold
  tuned on validation and applied to test.

## How to reproduce

```bash
cd nas_project && source .venv/bin/activate
python -m common.aspm_cache airport_analysis         # once, ~20 min
python -m models.planning.rotation_adjacency          # once, if not built for Planning
python -m features.strategic                          # ~1 min: daily features + labels
python -m features.validate_strategic_features
python -m models.strategic.train reference disruptnet sequence graph sarima   # ~1 h
python -m models.strategic.evaluate_test              # ONCE; refuses to re-run
for v in thr05 thr20 covid_included; do python -m models.short_horizon_sensitivity strategic $v; done   # ~45 min, validation only
```
