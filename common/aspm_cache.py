"""
One-time Parquet caches of the hourly ASPM reports, shared by all three
tiers. load_edct()/load_airport_analysis() parse HTML-format .xls at
~13-20s/month; this parses each month once and later reads take seconds.

Build (resumable - months already cached are skipped):
    python -m common.aspm_cache edct
    python -m common.aspm_cache airport_analysis

Read:
    from common.aspm_cache import load_edct_cache, load_airport_analysis_cache
"""

import sys
import time
from multiprocessing import Pool

import pandas as pd

from common.data_loader import DATA_DIR, load_airport_analysis, load_edct

REPORTS = {
    "edct": (load_edct, "aspm_edct_report", "edct_hourly"),
    "airport_analysis": (load_airport_analysis, "aspm_airport_analysis", "airport_analysis_hourly"),
}


def _cache_dir(report: str):
    return DATA_DIR / "cache" / REPORTS[report][2]


def _cache_month(args) -> str:
    report, stem = args
    loader = REPORTS[report][0]
    out = _cache_dir(report) / f"{stem}.parquet"
    if out.exists():
        return f"{stem}: cached already"
    t = time.time()
    year, month = (int(x) for x in stem.split("_"))
    df = loader(year, month)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    return f"{stem}: {len(df):,} rows in {time.time() - t:.0f}s"


def build_cache(report: str, processes: int = 3) -> None:
    _cache_dir(report).mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / REPORTS[report][1]
    stems = sorted(p.stem for p in source.glob("[0-9][0-9][0-9][0-9]_[0-9][0-9].xls"))
    print(f"{report}: {len(stems)} months found, caching to {_cache_dir(report)}", flush=True)
    with Pool(processes) as pool:
        for msg in pool.imap_unordered(_cache_month, [(report, s) for s in stems]):
            print(msg, flush=True)


def _load(report: str, columns: list[str] | None) -> pd.DataFrame:
    parts = sorted(_cache_dir(report).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no {report} cache at {_cache_dir(report)} - run `python -m common.aspm_cache {report}`")
    return pd.concat((pd.read_parquet(p, columns=columns) for p in parts), ignore_index=True)


def load_edct_cache(columns: list[str] | None = None) -> pd.DataFrame:
    return _load("edct", columns)


def load_airport_analysis_cache(columns: list[str] | None = None) -> pd.DataFrame:
    return _load("airport_analysis", columns)


if __name__ == "__main__":
    build_cache(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
