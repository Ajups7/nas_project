"""
Downloads historical METAR observations (temperature, wind, visibility,
sky cover/ceiling, present weather, altimeter) for the same airport set used
throughout this project, from the Iowa Environmental Mesonet (IEM) ASOS
archive - a free, no-login, scriptable CSV endpoint. Verified working and
much simpler than the ASPM/BTS sources: no browser automation, no session
bootstrap, no monthly chunking. One HTTP request per airport pulls the
entire date range directly.

Endpoint: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
Docs/UI:  https://mesonet.agron.iastate.edu/request/download.phtml

Usage:
  python download_weather_metar.py --start 2016-01-01 --end 2026-01-31
"""

import argparse
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

FIELDS = [
    "tmpf",     # temperature (F)
    "dwpf",     # dewpoint (F)
    "drct",     # wind direction
    "sknt",     # wind speed (knots)
    "gust",     # wind gust (knots)
    "vsby",     # visibility (miles)
    "skyc1", "skyl1",
    "skyc2", "skyl2",
    "skyc3", "skyl3",   # sky cover / ceiling, up to 3 layers
    "wxcodes",  # present weather / precip type
    "alti",     # altimeter setting
]

# Same 82-airport set used for ASPM Airport Analysis / EDCT Report (the
# site's own "ASPM 82" preset), for consistency across all three datasets.
AIRPORTS = [
    "ABQ", "ANC", "APA", "ASE", "ATL", "AUS", "BDL", "BHM", "BJC", "BNA",
    "BOI", "BOS", "BUF", "BUR", "BWI", "CLE", "CLT", "CMH", "CVG", "DAL",
    "DAY", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL", "GYY", "HNL", "HOU",
    "HPN", "IAD", "IAH", "IND", "ISP", "JAX", "JFK", "LAS", "LAX", "LGA",
    "LGB", "MCI", "MCO", "MDW", "MEM", "MHT", "MIA", "MKE", "MSP", "MSY",
    "OAK", "OGG", "OMA", "ONT", "ORD", "OXR", "PBI", "PDX", "PHL", "PHX",
    "PIT", "PSP", "PVD", "RDU", "RFD", "RSW", "SAN", "SAT", "SDF", "SEA",
    "SFO", "SJC", "SJU", "SLC", "SMF", "SNA", "STL", "SWF", "TEB", "TPA",
    "TUS", "VNY",
]

# IEM's ASOS network uses full ICAO identifiers for non-mainland stations
# (verified individually): Hawaii uses a "PH" prefix, Puerto Rico uses "TJ",
# rather than the plain 3-letter FAA/IATA code that works for mainland
# stations. Output files are still named by the project's standard 3-letter
# code (the dict key) for consistency with BTS/ASPM.
STATION_OVERRIDE = {
    "HNL": "PHNL",
    "OGG": "PHOG",
    "SJU": "TJSJ",
    "ANC": "PANC",
    # West Palm Beach Intl (PBI) was renamed Donald J Trump Intl (DJT) in
    # 2026, after our BTS/ASPM/EDCT data's Jan-2026 cutoff, so those
    # datasets are unaffected - but IEM's station registry already applies
    # the new code retroactively, so the weather lookup needs it.
    "PBI": "DJT",
}

MIN_VALID_SIZE = 1000  # a real multi-year pull is MBs; a station with no
# data or an error response is tiny - flag anything under 1KB as suspect.


def build_url(station, start_year, start_month, start_day, end_year, end_month, end_day):
    params = [("station", station)]
    params += [("data", f) for f in FIELDS]
    params += [
        ("year1", start_year), ("month1", start_month), ("day1", start_day),
        ("year2", end_year), ("month2", end_month), ("day2", end_day),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", "3"),
    ]
    req = requests.Request("GET", BASE_URL, params=params)
    return req.prepare().url


def download_airport(session, station, out_dir, start, end):
    dest = out_dir / f"{station}.csv"
    if dest.exists() and dest.stat().st_size >= MIN_VALID_SIZE:
        print(f"[skip] {station} already downloaded")
        return True

    query_station = STATION_OVERRIDE.get(station, station)
    sy, sm, sd = start
    ey, em, ed = end
    url = build_url(query_station, sy, sm, sd, ey, em, ed)

    try:
        resp = session.get(url, timeout=120)
    except requests.RequestException as exc:
        print(f"[fail] {station} request error: {exc}")
        return False

    if resp.status_code != 200:
        print(f"[fail] {station} HTTP {resp.status_code}")
        return False

    content = resp.content
    if len(content) < MIN_VALID_SIZE:
        print(f"[fail] {station} response too small ({len(content)} bytes): {content[:200]!r}")
        return False

    dest.write_bytes(content)
    lines = content.count(b"\n")
    print(f"[ok]   {station} downloaded ({len(content):,} bytes, ~{lines:,} rows)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-31", help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data" / "weather_metar"),
    )
    args = parser.parse_args()

    sy, sm, sd = (int(x) for x in args.start.split("-"))
    ey, em, ed = (int(x) for x in args.end.split("-"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(AIRPORTS)} airports: {args.start} -> {args.end}\n")

    session = requests.Session()
    failures = []
    for i, station in enumerate(AIRPORTS, start=1):
        print(f"--- [{i}/{len(AIRPORTS)}] {station} ---")
        ok = download_airport(session, station, out_dir, (sy, sm, sd), (ey, em, ed))
        if not ok:
            failures.append(station)
        time.sleep(3)

    print("\nDone.")
    if failures:
        print(f"{len(failures)} airport(s) failed: {failures}")
    print(f"Files written to: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
