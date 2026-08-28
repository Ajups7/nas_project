"""
Phase 0 shared data loader. Gives every tier a clean, typed, consistently-
named table for each of the four raw datasets, aligned to the join keys
defined in schema.md: (airport, date, local_hour, gmt_hour).

This module only cleans and aligns - it does not aggregate BTS (flight-level)
or weather (per-observation) up to any particular granularity. That
aggregation is tier-specific and belongs in each person's own features/ code.

Usage:
    from common.data_loader import load_bts, load_airport_analysis, load_edct, load_weather

    bts_jan16 = load_bts(2016, 1)
    aspm_jan16 = load_airport_analysis(2016, 1)
    edct_jan16 = load_edct(2016, 1)
    wx_atl = load_weather("ATL")
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Weather files are saved under the project's standard 3-letter code, but a
# few stations' internal ICAO identifier differs from that filename - see
# schema.md "Known airport-code quirks". Kept here (not just in the download
# script) since anyone loading the weather CSVs needs the same normalization.
WEATHER_STATION_TO_AIRPORT = {
    "PHNL": "HNL",
    "PHOG": "OGG",
    "TJSJ": "SJU",
    "PANC": "ANC",
    "DJT": "PBI",
}


def load_bts(year: int, month: int) -> pd.DataFrame:
    """Flight-level BTS On-Time Performance for one month. Not aggregated."""
    path = DATA_DIR / "bts_ontime" / "csv" / f"{year}_{month:02d}.csv"
    keep_cols = [
        "FlightDate", "Tail_Number", "Origin", "Dest",
        "CRSDepTime", "DepTime", "DepDelay", "DepDel15",
        "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
        "CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15",
        "Cancelled", "CancellationCode", "Diverted",
        "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance",
        "NASDelay", "WeatherDelay",
    ]
    df = pd.read_csv(path, usecols=keep_cols, low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    return df


def _clean_airport_hour_frame(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Shared cleanup for ASPM Airport Analysis / EDCT: flatten the
    multi-level HTML header, rename to canonical snake_case, fix types.

    Despite requesting "No Sub-Totals" at download time, ASPM still appends
    one grand-total row at the end of each report (Facility/Date/Hour all
    read "Total :") - verified on 2016-01 data. Drop any row that isn't a
    real MM/DD/YYYY date before type conversion."""
    df = df.rename(columns=rename_map)
    df = df[[c for c in rename_map.values() if c in df.columns]]
    df = df[df["date"].astype(str).str.match(r"^\d{2}/\d{2}/\d{4}$")]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["local_hour"] = df["local_hour"].astype(int)
    df["gmt_hour"] = df["gmt_hour"].astype(int)
    return df


def load_airport_analysis(year: int, month: int) -> pd.DataFrame:
    """ASPM Airport Analysis for one month, already airport-hour level."""
    path = DATA_DIR / "aspm_airport_analysis" / f"{year}_{month:02d}.xls"
    raw = pd.read_html(path)[0]
    raw.columns = raw.columns.get_level_values(1)
    rename_map = {
        "Facility": "airport",
        "Date": "date",
        "Hour": "local_hour",
        "GMT Hour": "gmt_hour",
        "Scheduled Departures": "scheduled_departures",
        "Scheduled Arrivals": "scheduled_arrivals",
        "Departures For Metric Computation": "departures_for_metric",
        "Arrivals For Metric Computation": "arrivals_for_metric",
        "% On-Time Gate Departures": "pct_ontime_gate_dep",
        "% On-Time Airport Departures": "pct_ontime_airport_dep",
        "% On-Time Gate Arrivals": "pct_ontime_gate_arr",
        "Average Gate Departure Delay": "avg_gate_dep_delay",
        "Average Taxi Out Time": "avg_taxi_out_time",
        "Average Taxi Out Delay": "avg_taxi_out_delay",
        "Average Airport Departure Delay": "avg_airport_dep_delay",
        "Average Airborne Delay": "avg_airborne_delay",
        "Average Taxi In Delay": "avg_taxi_in_delay",
        "Average Block Delay": "avg_block_delay",
        "Average Gate Arrival Delay": "avg_gate_arr_delay",
    }
    return _clean_airport_hour_frame(raw, rename_map)


def load_edct(year: int, month: int) -> pd.DataFrame:
    """ASPM EDCT Report for one month, already airport-hour level.
    GDP proxy label source - see schema.md for its known coverage gap
    (GDP only, no Ground Stop/MIT signal)."""
    path = DATA_DIR / "aspm_edct_report" / f"{year}_{month:02d}.xls"
    raw = pd.read_html(path)[0]

    # EDCT has a 3-level header: the paired "Count"/"%" sub-columns share a
    # level-1 name (e.g. two columns both named "Flts With EDCT Arriving
    # More Than 5 Min Early"), so combine level 1 + level 2 to disambiguate.
    lvl1 = raw.columns.get_level_values(1)
    lvl2 = raw.columns.get_level_values(2)
    flat_names = [
        l1 if l1 == l2 else f"{l1} ({l2})"
        for l1, l2 in zip(lvl1, lvl2)
    ]
    raw.columns = flat_names

    rename_map = {
        "Facility": "airport",
        "Date": "date",
        "Hour": "local_hour",
        "GMT Hour": "gmt_hour",
        "Arrivals For Metric Computation": "arrivals_for_metric",
        "Arrivals With EDCT": "arrivals_with_edct",
        "% Of Arrivals With EDCT": "pct_arrivals_with_edct",
        "Avg EDCT For All Arrivals": "avg_edct_all_arrivals",
        "Avg EDCT For Arrivals Where EDCT>0": "avg_edct_arrivals_gt0",
        "Flts With EDCT Arriving More Than 5 Min Early (Count)": "arr_edct_early5_count",
        "Flts With EDCT Arriving More Than 5 Min Early (%)": "arr_edct_early5_pct",
        "Flts With EDCT Arriving More Than 15 Min Late (Count)": "arr_edct_late15_count",
        "Flts With EDCT Arriving More Than 15 Min Late (%)": "arr_edct_late15_pct",
        "Departures For Metric Computation": "departures_for_metric",
        "Departures With EDCT": "departures_with_edct",
        "% Of Departures With EDCT": "pct_departures_with_edct",
        "Avg EDCT For All Departures": "avg_edct_all_departures",
        "Avg EDCT For Departures Where EDCT>0": "avg_edct_departures_gt0",
        "Flts With EDCT Departing More Than 5 Min Early (Count)": "dep_edct_early5_count",
        "Flts With EDCT Departing More Than 5 Min Early (%)": "dep_edct_early5_pct",
        "Flts With EDCT Departing More Than 15 Min Late (Count)": "dep_edct_late15_count",
        "Flts With EDCT Departing More Than 15 Min Late (%)": "dep_edct_late15_pct",
    }
    return _clean_airport_hour_frame(raw, rename_map)


def load_weather(airport: str) -> pd.DataFrame:
    """METAR observations for one airport, full 2016-2026 range.
    Per-observation (~hourly, not exactly on the hour) - not pre-aggregated.
    `local_hour_approx` is derived from the UTC timestamp without full
    timezone correction; use `gmt_hour` to align precisely with ASPM/EDCT."""
    path = DATA_DIR / "weather_metar" / f"{airport}.csv"
    df = pd.read_csv(path, low_memory=False)
    df["valid"] = pd.to_datetime(df["valid"])
    df["airport"] = airport
    df["date"] = df["valid"].dt.normalize()  # datetime64, not a python date -
    # keeps the dtype consistent with load_airport_analysis/load_edct's "date"
    df["gmt_hour"] = df["valid"].dt.hour
    df["local_hour_approx"] = df["gmt_hour"]  # placeholder - see docstring
    cols = [
        "airport", "date", "local_hour_approx", "gmt_hour",
        "tmpf", "dwpf", "drct", "sknt", "gust", "vsby",
        "skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3",
        "wxcodes", "alti",
    ]
    return df[cols]
