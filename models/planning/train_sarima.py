"""
SARIMA baseline for the Planning tier (models/planning/baselines.py's
SARIMABaseline settings, adapted - see below). Per airport:

- Series: the daily GDP-day indicator (common/disruption.py). COVID days
  and missing EDCT months are NaN - SARIMA's Kalman filter skips them.
- Exog: the airport's train-period disruption climatology for each day's
  calendar month (features/planning.py), so the model has the annual cycle.
  A seasonal period of 365 is impractical for SARIMA; period 7 (weekly)
  stays as in baselines.py.
- order (1,0,1), seasonal (1,0,1,7): no differencing - a 0/1 GDP-day
  series is level-stationary; baselines.py's d=1 default would forecast a
  random walk.
- Fit on train dates only, then filter once over the full series with
  those fixed parameters and compute every 30-day-ahead forecast from the
  stored one-step-ahead states (exact for a linear state-space model; see
  check_against_statsmodels). One get_prediction call per origin would take
  over an hour.
- P(any GDP day in the next 30) = 1 - prod(1 - clip(forecast_k, 0, 1)),
  treating days as independent - an approximation (GDP days cluster), so
  calibration is reported but ranking metrics are the fairer comparison.

Run from the repo root:
    python -m models.planning.train_sarima [--with-test]   # --with-test: also save test forecasts
"""

import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common.disruption import gdp_days
from common.temporal_split import TRAIN_END
from features.planning import build_disruption_climatology, eligible_airports
from features.planning_label import DEFAULT_HORIZON_DAYS
from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.data import in_covid, load_planning_splits
from models.planning.metrics import decision_costs, evaluate
from models.planning.train_disruptnet import RESULTS_DIR

ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 7)
H = DEFAULT_HORIZON_DAYS


def _series(airport: str):
    y = gdp_days()[airport].copy()
    y[in_covid(y.index.to_series()).to_numpy()] = np.nan
    climo = build_disruption_climatology(airport)
    exog = pd.Series(y.index.month.map(climo).to_numpy(), index=y.index, name="climo")
    return y, exog


def _forecast_all_origins(res_full, origins: np.ndarray, horizon: int = H) -> np.ndarray:
    """(n_origins, horizon) forecasts of y[t+1..t+horizon] given data
    through t, from the filter's one-step predicted states."""
    fr = res_full.filter_results
    Z = fr.design[0, :, 0]
    T = fr.transition[:, :, 0]
    d = fr.obs_intercept[0]
    c = fr.state_intercept
    c = np.repeat(c, len(d), axis=1) if c.shape[1] == 1 else c
    A = fr.predicted_state[:, origins + 1]
    out = np.empty((len(origins), horizon))
    for j in range(horizon):
        t = origins + 1 + j
        out[:, j] = d[t] + Z @ A
        A = T @ A + c[:, t]
    return out


def fit_airport(airport: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, exog = _series(airport)
        train_mask = y.index <= TRAIN_END
        model = SARIMAX(y[train_mask], exog=exog[train_mask], order=ORDER,
                        seasonal_order=SEASONAL_ORDER, trend="c")
        res = model.fit(disp=False, maxiter=200)
        full = SARIMAX(y, exog=exog, order=ORDER, seasonal_order=SEASONAL_ORDER,
                       trend="c").filter(res.params)
    return airport, res, full, y.index


def check_against_statsmodels(full, index, origins: list[int]) -> float:
    fc = _forecast_all_origins(full, np.array(origins))
    worst = 0.0
    for k, t in enumerate(origins):
        ref = full.get_prediction(start=t + 1, end=t + H, dynamic=True).predicted_mean.to_numpy()
        worst = max(worst, float(np.abs(fc[k] - ref).max()))
    return worst


def _predictions(fitted, frame: pd.DataFrame) -> pd.DataFrame:
    preds = []
    for airport, _, full, index in fitted:
        rows = frame[frame["airport"] == airport]
        pos = pd.Series(np.arange(len(index)), index=index)
        fc = np.clip(_forecast_all_origins(full, pos.reindex(rows["date"]).to_numpy()), 0, 1)
        preds.append(pd.DataFrame({"airport": airport, "date": rows["date"].to_numpy(),
                                   "label": rows["label"].to_numpy(), "p": 1 - np.prod(1 - fc, axis=1),
                                   "mean_daily_forecast": fc.mean(axis=1)}))
    return pd.concat(preds, ignore_index=True)


def main(with_test: bool = False):
    splits = load_planning_splits(exclude_covid=True)
    val = splits["validation"]
    airports = eligible_airports()
    t0 = time.time()
    with Pool(3) as pool:
        fitted = pool.map(fit_airport, airports)
    print(f"fitted {len(fitted)} airports in {time.time() - t0:.0f}s", flush=True)

    _, _, full0, index0 = fitted[airports.index("EWR")]
    pos0 = pd.Series(np.arange(len(index0)), index=index0)
    probe = [int(pos0[pd.Timestamp(d)]) for d in ("2023-03-01", "2023-07-15", "2023-11-20")]
    print(f"vectorized vs statsmodels get_prediction, max abs diff: "
          f"{check_against_statsmodels(full0, index0, probe):.2e}", flush=True)

    failed = [a for a, res, _, _ in fitted if not res.mle_retvals.get("converged", True)]
    print(f"MLE did not converge for: {failed or 'none'}")

    pred = _predictions(fitted, val)
    RESULTS_DIR.mkdir(exist_ok=True)
    pred.to_parquet(RESULTS_DIR / "sarima_validation_predictions.parquet", index=False)
    if with_test:
        _predictions(fitted, splits["test"]).to_parquet(RESULTS_DIR / "sarima_test_predictions.parquet", index=False)
        print("test predictions saved (scored by models.planning.evaluate_test, not here)")

    faoc = FAOCLoss()
    dm = np.array([faoc.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in pred["airport"]])
    bayes = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    tuned_costs = [decision_costs(pred["label"], pred["p"] >= t, pred["airport"], faoc)["cost_per_airport_day"]
                   for t in grid]

    m = evaluate(pred["label"], pred["p"], pred["airport"])
    m_mean = evaluate(pred["label"], pred["mean_daily_forecast"].clip(1e-6, 1 - 1e-6), pred["airport"])
    m["cost_bayes"] = decision_costs(pred["label"], pred["p"] >= bayes, pred["airport"], faoc)["cost_per_airport_day"]
    m["cost_tuned"] = min(tuned_costs)
    m["tuned_threshold"] = float(grid[int(np.argmin(tuned_costs))])
    pd.Series(m).to_csv(RESULTS_DIR / "sarima_validation.csv")
    print(f"\nVALIDATION rows scored: {len(pred):,}")
    print(pd.Series(m).round(4).to_string())
    print(f"\n(ranking with the mean daily forecast instead: within-airport AUC "
          f"{m_mean['within_airport_auc']:.3f}, pooled AUC {m_mean['roc_auc']:.3f})")


if __name__ == "__main__":
    import sys

    main(with_test="--with-test" in sys.argv)
