# Tactical tier: training results and decisions

Tactical tier (0–2 h): at the end of UTC hour t, predict whether a GDP hour
occurs at t+1 or t+2. Trained 2026-10-22/23. Same protocol as the other
tiers (`models/planning/training_results.md`): 75 airports, COVID excluded
from train, one-hot airport identity, 3 seeds, early stopping on
validation, cost-ratio decision threshold (FAOC: 0.5). Cost is $ per
airport-**hour** decision, using FAOC-Loss's weights.

## Final results: test set (evaluated once, 2026-10-23)

Test: 974,822 airport-hours, 2.28% positive; onset hours 956,305 (0.68%);
clean-onset hours 935,843 (0.14%). Mean of 3 seeds (± seed std); cost in
$ per airport-hour at the rule fixed in advance.

| Model | Cost, all hours | Cost, onset hours | Onset PR-AUC | Clean-onset AUC | **Clean-onset PR-AUC** | Log loss |
|---|---|---|---|---|---|---|
| *Never flag* | 406.8 | | | | | |
| *Always flag* | 1,442.5 | | | | | |
| Climatology lookup | 281.7 | 162.0 | 0.047 | 0.779 | 0.006 | 0.0986 |
| Persistence | 123.2 | 121.2 | 0.028 | 0.798 | 0.006 | 0.0430 |
| Logistic regression | 53.5 | 49.5 | 0.455 | 0.885 | 0.035 | 0.0263 |
| DisruptNet (MoE), standard | 45.9 ± 0.2 | 41.9 | 0.364 | 0.933 | 0.041 | 0.0484 |
| DisruptNet (MoE), FAOC, at 0.5 | 45.3 ± 0.4 | 41.7 | 0.361 | 0.933 | 0.042 | 0.0693 |
| TFT (= TacticalDisruptNet) | 43.6 ± 0.1 | 39.9 | 0.609 | 0.940 | 0.098 ± 0.010 | 0.0205 |
| Seq2Seq | 40.4 ± 0.3 | 37.0 | 0.649 | 0.947 | 0.108 ± 0.003 | 0.0192 |
| STGNN, no graph | 42.0 ± 0.2 | 38.6 | 0.619 | 0.940 | 0.103 ± 0.007 | 0.0201 |
| STGNN, full graph | 37.9 ± 0.5 | 34.3 | 0.665 | 0.958 | 0.187 ± 0.024 | 0.0186 |
| **STGNN, top-10 rotation graph (main)** | **36.3 ± 1.4** | **32.7** | **0.686** | **0.959** | **0.216 ± 0.030** | **0.0179** |
| STGNN top-10, departure EDCTs removed | 42.0 ± 1.0 | 38.4 | 0.626 | 0.943 | 0.099 ± 0.002 | 0.0197 |
| STGNN no graph, departure EDCTs removed | 41.9 ± 0.0 | 38.5 | 0.620 | 0.940 | 0.103 ± 0.004 | 0.0201 |

1. **The main model cuts decision cost 91% vs never-flagging** ($36 vs
   $407) and 71% vs persistence ($123).
2. **Clean-onset early warning is real but modest**: PR-AUC 0.22 at a
   0.14% base rate (~150× chance) for the top-10 STGNN; ~0.10 for the
   sequence models without graph information.
3. **The ablation holds on test**: without departure EDCTs, the top-10
   graph equals no graph exactly ($42.0 vs $41.9; clean-onset 0.099 vs
   0.103). The graph's whole advantage is network visibility of
   already-issued programs.
4. **Sequence context beats tabular**: Seq2Seq/TFT/STGNN ≫ DisruptNet and
   logistic regression on onsets.
5. **FAOC ≈ standard loss** ($45.3 vs $45.9) — the same at all three tiers.
6. **Calibration holds into 2025** for the sequence/graph models (mean
   prediction 2.4–2.5% vs 2.5% actual). The tabular DisruptNet's log loss
   doubled from validation to test (0.021 → 0.048) — overconfident on
   some test hours; its cost and ranking held.
7. Validation-tuned thresholds change test cost by under $1.5.

Per-seed results and row-level predictions: `results/test_results_per_seed.csv`,
`results/test_results_summary.csv`, `results/test_predictions.parquet`.

## Decisions

### GDP hour (`common/disruption.py`)

GDP hour = ≥ 50% of the hour's arrivals had an EDCT, and ≥ 2 were held.
Among train-period hours with ≥ 5 arrivals, the share of arrivals with an
EDCT falls from 10% to a trough at 30–50%, then rises to a second mode at
70–80% (hours inside a running program):

| Hourly share | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|
| Hours | 12,574 | 8,640 | 6,023 | 5,858 | 8,555 | 11,751 | 13,655 | 12,785 | 5,507 |

At ≥ 50%, 97.5% of GDP hours fall on GDP days by the daily definition
(consistent with the other tiers), and they cover 31% of hours on GDP
days. The ≥ 2 held minimum keeps a single held flight at a small airport
from counting as a program.

### Features (`features/tactical.py`, 43 + airport identity)

Rewritten from Person A's per-airport, per-month pandas version as dense
(airport × hour) arrays on a **UTC axis**, so index t is the same moment at
every airport (required to test simultaneous propagation). Kept Person A's
3-hour rolling mean/delta design. Person A's "lightweight rotation signal"
(mean arrival delay of flights landing each hour, from 28 GB of BTS) is
taken from ASPM Airport Analysis instead, which reports the same quantity
per airport-hour. `validate_tactical_features.py` checks UTC alignment,
GDP-hour flags, all 5.17M labels, weather timing, and that nothing looks
ahead (scheduled counts for t+1, t+2 are published in advance). Running
sums are in float64: in float32 the drift over 88k hours was visible.

Only hours with scheduled arrivals in t+1..t+2 are scored.

### Three evaluation subsets

| Validation subset | Rows | Positive rate |
|---|---|---|
| All scored hours | 525,850 | 1.54% |
| **Onset**: not in a GDP hour now | 519,297 | 0.52% |
| **Clean onset**: no arrival held with an EDCT yet | 509,314 | 0.11% |

70% of positive hours are continuations of a GDP already running
(current arrival-EDCT share alone correlates 0.80 with the label).
Of the onsets, **80% are already visible**: inbound flights get EDCTs
before the hour the program crosses 50%. Only clean onsets — no held
arrivals yet — test early warning before a program exists, i.e. the
paper's 90-minute CDM claim. PR-AUC is the meaningful ranking metric at
these base rates.

## Validation results

Mean of 3 seeds (± seed std).

| Model | Cost, all hours | Onset PR-AUC | Clean-onset AUC | **Clean-onset PR-AUC** | Log loss |
|---|---|---|---|---|---|
| Persistence (airport × in-GDP-now) | 97.2 | 0.018 | 0.798 | 0.004 | 0.0353 |
| Climatology (airport × month × local hour) | 237.6 | 0.031 | 0.823 | 0.005 | 0.0685 |
| Logistic regression | 43.2 | 0.421 | 0.898 | 0.028 | 0.0209 |
| DisruptNet (MoE), standard | 35.6 ± 0.3 | 0.483 | 0.947 | 0.057 ± 0.002 | 0.0211 |
| DisruptNet (MoE), FAOC, at 0.5 | 35.6 ± 0.6 | 0.471 | 0.948 | 0.056 ± 0.003 | 0.0421 |
| Seq2Seq (12-h windows) | 33.8 ± 0.5 | 0.618 | 0.961 | 0.121 ± 0.002 | 0.0154 |
| TFT (= Person A's TacticalDisruptNet, plain BCE) | 35.6 ± 0.5 | 0.590 | 0.950 | 0.108 ± 0.021 | 0.0161 |
| STGNN, no graph | 35.5 ± 0.6 | 0.573 | 0.951 | 0.100 ± 0.008 | 0.0162 |
| STGNN, full graph | 29.7 ± 0.8 | 0.642 | 0.970 | 0.241 ± 0.038 | 0.0144 |
| **STGNN, top-10 rotation graph** | **28.4 ± 1.6** | **0.672** | **0.968** | **0.316 ± 0.041** | **0.0137** |

- **The rotation graph helps at this horizon** — the first tier where it
  does: top-10 graph vs. the same model with no graph is −20% cost and
  3.2× clean-onset PR-AUC. The architecture alone (STGNN, no graph) is no
  better than TFT, so the gain is the graph's.
- **Sequence context matters**: models that see the last 12 hours roughly
  double the tabular DisruptNet and quadruple logistic regression on
  clean onsets.
- **FAOC ≈ standard loss** again (identical cost, twice the log loss) —
  the same result at all three horizons.
- Clean-onset PR-AUC of 0.32 at a 0.11% base rate is ~290× chance: real
  early warning, though most alerts would still be false.

### Is the graph detecting already-issued programs? (ablation)

When a GDP is issued at airport A, flights *departing other airports for
A* are held first — their departure-EDCT counts rise before any held
flight reaches A. Through the graph, the model can see A's neighbors'
departure holds: real information at time t, but it detects an issued
program rather than forecasting propagation. Test: retrain STGNN (top-10
and no graph) with `dep_edct_share` neutralized at every airport.

| STGNN (3 seeds) | Cost | Onset PR-AUC | Clean-onset PR-AUC |
|---|---|---|---|
| Top-10 graph, all features | 28.4 ± 1.6 | 0.672 | 0.316 ± 0.041 |
| No graph, all features | 35.5 ± 0.6 | 0.573 | 0.100 ± 0.008 |
| Top-10 graph, departure EDCTs removed | 34.4 ± 0.5 | 0.581 | 0.108 ± 0.006 |
| No graph, departure EDCTs removed | 35.4 ± 0.5 | 0.574 | 0.102 ± 0.008 |

**Almost all of the graph's gain is detecting already-issued programs.**
With departure EDCTs removed, the top-10 graph is worth ~$1 of cost
(~2 seed std) and no measurable clean-onset gain (0.108 vs 0.102).
Removing an airport's *own* departure holds changes nothing for the
no-graph model ($35.5 vs $35.4) — the signal that matters is
*neighbors'* holds on flights bound for this airport, exactly the
issued-program mechanism. Genuine propagation of delays and weather
along rotations adds little at 0–2 h with these inputs.

What the graph does provide is network-wide visibility of programs being
put in place, an hour or two before they show at the airport itself.
That has value for users who can't see EDCT assignments system-wide; it
is not a forecast of the FAA's decision.

## Sensitivity (validation only, `models/short_horizon_sensitivity.py`)

Run after the test evaluation; validation only, nothing selected from it.
Cost per airport-hour; not comparable across thresholds (the label changes).

| Setting (validation positive rate) | Logistic | DisruptNet std | DisruptNet FAOC | STGNN no graph | STGNN top-10 | Clean-onset PR-AUC, no graph → top-10 |
|---|---|---|---|---|---|---|
| GDP hour ≥ 30% (2.0%) | 60.5 | 48.3 ± 0.4 | 47.8 ± 0.4 | 46.0 ± 1.2 | **36.5 ± 0.8** | 0.14 → 0.35 |
| **≥ 50%, main** (1.5%) | 43.2 | 35.6 ± 0.3 | 35.6 ± 0.6 | 35.5 ± 0.6 | **28.4 ± 1.6** | 0.10 → 0.32 |
| ≥ 70% (0.9%) | 32.3 | 24.6 ± 0.5 | 24.5 ± 0.3 | 24.5 ± 0.3 | **22.9 ± 0.9** | 0.07 → 0.12 |
| ≥ 50%, COVID kept in train (1.5%) | 43.7 | 35.0 ± 0.6 | 35.3 ± 0.4 | 35.2 ± 0.7 | **29.5 ± 0.6** | 0.10 → 0.29 |

- The top-10 graph's advantage holds in every setting (−6% to −21% cost),
  smallest at 70% — rarer, more severe programs. The departure-EDCT
  ablation was run at the main setting only; nothing suggests the
  mechanism differs at the others.
- FAOC ties the standard loss everywhere (within $0.5).
- Logistic regression is the weakest learned model in every setting.
- Keeping COVID in train changes nothing material.

## Frozen before the test evaluation (written 2026-10-23, before `evaluate_test.py` ran)

- GDP-hour definition, label, 75 airports, COVID exclusion, features and
  airport identity as above.
- Every model in the validation tables (including the two
  no-departure-EDCT ablations) is evaluated on test from the predictions
  it wrote at training time (`results/predictions/test_*.parquet`, sealed
  — not scored until now). Nothing is retrained or re-selected.
- Main model for the narrative: **STGNN with the top-10 rotation graph**
  (the paper's architecture as designed); no-graph and
  no-departure-EDCT variants are its ablations.
- Reported on all hours, onset hours and clean-onset hours. Decision
  rules: cost-ratio threshold (FAOC: 0.5), plus a threshold tuned on
  validation and applied to test.

## How to reproduce

```bash
cd nas_project && source .venv/bin/activate
python -m common.aspm_cache edct; python -m common.aspm_cache airport_analysis   # once
python -m models.planning.rotation_adjacency          # once
python -m features.tactical                           # ~2 min: hourly UTC tensor (1.2 GB)
python -m features.validate_tactical_features
for s in reference disruptnet sequence graph graph_nodep; do python -m models.tactical.train $s; done   # ~4.5 h
python -m models.tactical.evaluate_test               # ONCE; refuses to re-run
for v in thr30 thr70 covid_included; do python -m models.short_horizon_sensitivity tactical $v; done    # ~4.5 h, validation only
```
