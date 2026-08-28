# Shared Data Schema

Phase 0 output. This defines the join keys and canonical column names all
three tiers build on top of, via `common/data_loader.py`. Change this file
only with a heads-up to the other two teammates — everyone's feature
pipeline depends on it.

## Join keys

Every dataset gets aligned to:

| key | type | meaning |
|---|---|---|
| `airport` | str, 3-letter | canonical airport code (see quirks below) |
| `date` | date | calendar date |
| `local_hour` | int, 0-23 | hour in the airport's local time |
| `gmt_hour` | int, 0-23 | hour in UTC (needed to align airports across time zones) |

Weather and BTS don't come pre-aggregated to `(airport, date, hour)` the way
ASPM Airport Analysis and EDCT do — `common/data_loader.py` gives you the
*cleaned raw* table for each source; aggregating BTS (flight-level) and
weather (per-observation) up to your tier's chosen granularity is tier-specific
work, not something the shared loader decides for you.

## Known airport-code quirks (already handled in the loader)

- **PBI / DJT**: West Palm Beach Intl was renamed Donald J Trump Intl in
  2026, after our data's Jan-2026 cutoff. BTS/ASPM/EDCT all use `PBI`
  throughout our downloaded range. The weather files are keyed by `PBI` on
  disk (filename), but the station registry now reports the ICAO code as
  `DJT` internally — the loader normalizes this back to `PBI` so it joins
  cleanly with the other three sources.
- **HNL, OGG, SJU, ANC**: weather files use ICAO codes internally (PHNL,
  PHOG, TJSJ, PANC) but are saved and loaded as the standard 3-letter code
  (HNL, OGG, SJU, ANC) to match BTS/ASPM/EDCT.

## Per-source columns (as exposed by the loader)

### BTS On-Time (`load_bts(year, month)`) — flight-level, not pre-aggregated

Key columns kept: `FlightDate`, `Tail_Number`, `Origin`, `Dest`,
`CRSDepTime`, `DepTime`, `DepDelay`, `DepDel15`, `TaxiOut`, `WheelsOff`,
`WheelsOn`, `TaxiIn`, `CRSArrTime`, `ArrTime`, `ArrDelay`, `ArrDel15`,
`Cancelled`, `CancellationCode`, `Diverted`, `CRSElapsedTime`,
`ActualElapsedTime`, `AirTime`, `Distance`, `NASDelay`, `WeatherDelay`.
(`Tail_Number` + `FlightDate`, sorted, is the basis for the rotation-chain
graph — see Person B's `features/rotation_graph.py`.)

### ASPM Airport Analysis (`load_airport_analysis(year, month)`) — airport-hour level

`airport, date, local_hour, gmt_hour, scheduled_departures,
scheduled_arrivals, departures_for_metric, arrivals_for_metric,
pct_ontime_gate_dep, pct_ontime_airport_dep, pct_ontime_gate_arr,
avg_gate_dep_delay, avg_taxi_out_time, avg_taxi_out_delay,
avg_airport_dep_delay, avg_airborne_delay, avg_taxi_in_delay,
avg_block_delay, avg_gate_arr_delay`

### ASPM EDCT Report (`load_edct(year, month)`) — airport-hour level, GDP proxy label source

`airport, date, local_hour, gmt_hour, arrivals_for_metric,
arrivals_with_edct, pct_arrivals_with_edct, avg_edct_all_arrivals,
avg_edct_arrivals_gt0, arr_edct_early5_count, arr_edct_early5_pct,
arr_edct_late15_count, arr_edct_late15_pct, departures_for_metric,
departures_with_edct, pct_departures_with_edct, avg_edct_all_departures,
avg_edct_departures_gt0, dep_edct_early5_count, dep_edct_early5_pct,
dep_edct_late15_count, dep_edct_late15_pct`

### Weather METAR (`load_weather(airport)`) — per-observation, ~hourly

`airport, date, local_hour_approx, tmpf, dwpf, drct, sknt, gust, vsby,
skyc1, skyl1, skyc2, skyl2, skyc3, skyl3, wxcodes, alti`

Note: METAR timestamps (`valid`) are UTC: the loader derives `date` and an
approximate local hour, but does not do full timezone-correct localization
— if your tier needs exact local-hour alignment with ASPM/EDCT, resolve via
`gmt_hour` instead of the approximate local column.

## Naming convention for your own tier's derived feature tables

Whatever you build on top of the loader, keep the four join key columns
named exactly `airport`, `date`, `local_hour`, `gmt_hour` in your own output
tables too — that's what lets Person C's validation framework (Phase 4)
evaluate all three tiers consistently later.
