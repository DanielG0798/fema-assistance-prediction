"""Download and cache the FEMA IHP registrations for Hurricane Ian, Lee County FL.

Team usage — this is the only thing you need to know:

    from src.data.fetch_fema import load_ian_lee
    df = load_ian_lee()

First call downloads from OpenFEMA and caches to data/raw/. Every call after
reads the cache in a second. No API key, no setup, same dataframe for everyone.

Requires: pandas, requests, pyarrow
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATASET = "IndividualsAndHouseholdsProgramValidRegistrations"
BASE = f"https://www.fema.gov/api/open/v2/{DATASET}"
IAN_DR = 4673                      # Hurricane Ian, FL
COUNTY = "Lee (County)"            # verified against the data on first run
PAGE = 5000                        # API caps $top at 10000

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE = RAW_DIR / f"ihp_dr{IAN_DR}_lee.parquet"
CACHE_ALL = RAW_DIR / f"ihp_dr{IAN_DR}_all_counties.parquet"


def _get(params: dict, retries: int = 6) -> dict:
    """One API call, with backoff. OpenFEMA rate-limits under load and throws
    occasional 503s / dropped connections on big pages."""
    for attempt in range(retries):
        try:
            r = requests.get(BASE, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(min(3 * 2 ** attempt, 60))
    raise RuntimeError("unreachable")


def download_disaster(disaster_number: int = IAN_DR, county: str | None = None) -> pd.DataFrame:
    """Every valid IHP registration for one disaster.

    With county=None this is all counties (~910k rows for Ian, ~100 min).
    With a county string the API filters server-side (~195k rows for Lee,
    ~20 min). The county is still re-checked in pandas afterwards, so a wrong
    county string shows up as an empty result we can SEE and get a
    "did you mean" hint for, rather than a silent zero-row API response.
    """
    flt = f"disasterNumber eq {disaster_number}"
    if county is not None:
        flt += f" and county eq '{county}'"
    params = {
        "$filter": flt,
        "$top": PAGE,
        "$skip": 0,
        "$inlinecount": "allpages",
    }
    first = _get(params)
    total = int(first["metadata"]["count"])
    key = next(k for k in first if k != "metadata")
    rows = list(first[key])
    print(f"{total:,} records for DR-{disaster_number}", file=sys.stderr)

    while len(rows) < total:
        params["$skip"] = len(rows)
        batch = _get(params)[key]
        if not batch:
            break
        rows.extend(batch)
        print(f"  {len(rows):,}/{total:,}", file=sys.stderr)

    return pd.DataFrame(rows)


def load_ian_lee(refresh: bool = False, county: str = COUNTY) -> pd.DataFrame:
    """The dataframe the whole project is built on. Cached after the first call."""
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_ALL.exists() and not refresh:
        everything = pd.read_parquet(CACHE_ALL)
    else:
        # County filtered server-side: ~5x fewer rows than pulling the whole state.
        everything = download_disaster(county=county)
        if everything.empty:
            raise ValueError(
                f"OpenFEMA returned no rows for DR-{IAN_DR} county={county!r}. "
                f"Check the spelling (the API uses e.g. 'Lee (County)')."
            )

    col = next((c for c in ("county", "countyName") if c in everything.columns), None)
    if col is None:
        raise KeyError(f"no county column found; got {list(everything.columns)}")

    subset = everything[everything[col] == county]
    if subset.empty:
        near = sorted(v for v in everything[col].dropna().unique() if "lee" in str(v).lower())
        raise ValueError(
            f"No rows for county={county!r}. Did you mean one of {near}? "
            f"Pass the right string as load_ian_lee(county=...)."
        )

    subset.to_parquet(CACHE, index=False)
    return subset


def describe(df: pd.DataFrame) -> None:
    """What the team needs to pick a target variable."""
    print(f"\n{len(df):,} rows x {len(df.columns)} columns\n")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "pct_null": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(dropna=True),
    })
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.sort_values("pct_null"))


if __name__ == "__main__":
    data = load_ian_lee(refresh="--refresh" in sys.argv)
    describe(data)
    print(f"\ncached at {CACHE.relative_to(REPO_ROOT)}")
