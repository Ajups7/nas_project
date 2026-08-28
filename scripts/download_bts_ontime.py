"""
Downloads BTS "Reporting Carrier On-Time Performance (1987-present)" monthly
prezipped files (all fields included) for a given year/month range, and
extracts the CSV from each zip.

Source: https://transtats.bts.gov/PREZIP/
URL pattern confirmed working 2026-08-18:
  On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip

Usage:
  python download_bts_ontime.py
  python download_bts_ontime.py --start 2016-01 --end 2026-01
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (research data download script)"}
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_DELAY_SECONDS = 2


def month_range(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_month(session, year, month, raw_dir):
    url = BASE_URL.format(year=year, month=month)
    dest_zip = raw_dir / f"{year}_{month:02d}.zip"

    if dest_zip.exists():
        print(f"[skip] {year}-{month:02d} already downloaded")
        return dest_zip

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=60)
        except requests.RequestException as exc:
            print(f"[error] {year}-{month:02d} attempt {attempt}: {exc}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
            "application/x-zip"
        ):
            dest_zip.write_bytes(resp.content)
            print(f"[ok]   {year}-{month:02d} downloaded ({len(resp.content):,} bytes)")
            return dest_zip

        if resp.status_code == 404:
            print(f"[none] {year}-{month:02d} not published yet (404) — skipping")
            return None

        print(
            f"[warn] {year}-{month:02d} attempt {attempt}: "
            f"unexpected status {resp.status_code}, retrying"
        )
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"[fail] {year}-{month:02d} gave up after {MAX_RETRIES} attempts")
    return None


def extract_csv(zip_path, csv_dir):
    if zip_path is None:
        return
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            print(f"[warn] no CSV found inside {zip_path.name}")
            return
        for member in csv_members:
            target = csv_dir / f"{zip_path.stem}.csv"
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"[ok]   extracted -> {target.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01", help="YYYY-MM, default 2016-01")
    parser.add_argument("--end", default="2026-01", help="YYYY-MM, default 2026-01")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data" / "bts_ontime"),
        help="output base directory (expects raw_zip/ and csv/ subfolders)",
    )
    args = parser.parse_args()

    start_year, start_month = (int(x) for x in args.start.split("-"))
    end_year, end_month = (int(x) for x in args.end.split("-"))

    out_base = Path(args.out)
    raw_dir = out_base / "raw_zip"
    csv_dir = out_base / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    months = list(month_range(start_year, start_month, end_year, end_month))
    print(f"Downloading {len(months)} months: {args.start} -> {args.end}\n")

    failures = []
    for i, (year, month) in enumerate(months, start=1):
        print(f"--- [{i}/{len(months)}] {year}-{month:02d} ---")
        zip_path = download_month(session, year, month, raw_dir)
        if zip_path is not None:
            extract_csv(zip_path, csv_dir)
        else:
            failures.append((year, month))
        time.sleep(REQUEST_DELAY_SECONDS)

    print("\nDone.")
    if failures:
        print(f"{len(failures)} month(s) unavailable or failed: {failures}")
    print(f"CSV files written to: {csv_dir}")


if __name__ == "__main__":
    sys.exit(main())
