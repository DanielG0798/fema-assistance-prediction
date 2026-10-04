"""Download and cache FEMA IHP registrations for any disaster / county.

Team usage — this is the only thing you need to know:

    from src.data.fetch_fema import load_ian_lee
    df = load_ian_lee()

By default this returns Hurricane Ian (DR-4673), Lee County, FL. It checks,
in order:

  1. A committed CSV in `data/raw/` (fastest, no network needed).
  2. A local parquet cache in `data/raw/`.
  3. A live download from the OpenFEMA API (slow, only on first run).

To force a fresh API download:

    python -m src.data.fetch_fema --refresh

You can also point it at a different disaster or county:

    df = load_ian_lee(disaster_number=4575, county="Sarasota (County)")

Requires: pandas, requests, pyarrow (for parquet caching).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATASET = "IndividualsAndHouseholdsProgramValidRegistrations"
BASE = f"https://www.fema.gov/api/open/v2/{DATASET}"
IAN_DR = 4673                      # Hurricane Ian, FL
COUNTY = "Lee (County)"            # verified against the data on first run
PAGE = 1000                        # smaller pages avoid OpenFEMA 503s under load

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _safe_name(county: str) -> str:
    """Turn 'Lee (County)' into 'lee' for filenames.

    FEMA county strings include a parenthetical qualifier such as
    '(County)', '(Parish)', or '(Borough)'. We strip that suffix to keep
    filenames concise and consistent with the committed CSV.
    """
    name = re.sub(r"\s*\([^)]*\)\s*$", "", county.lower()).strip()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _csv_path(disaster_number: int, county: str) -> Path:
    return RAW_DIR / f"ihp_dr{disaster_number}_{_safe_name(county)}.csv"


def _cache_path(disaster_number: int, county: str) -> Path:
    return RAW_DIR / f"ihp_dr{disaster_number}_{_safe_name(county)}.parquet"


def _cache_all_path(disaster_number: int) -> Path:
    return RAW_DIR / f"ihp_dr{disaster_number}_all_counties.parquet"


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
        "$orderby": "id",
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

    df = pd.DataFrame(rows)
    dupes = int(df["id"].duplicated().sum())
    if dupes:
        print(f"warning: dropping {dupes:,} duplicate ids returned by the API", file=sys.stderr)
        df = df.drop_duplicates(subset="id")
    return df


def load_ian_lee(
    disaster_number: int = IAN_DR,
    county: str = COUNTY,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the project dataset, preferring a local CSV, then parquet, then API."""
    csv_path = _csv_path(disaster_number, county)
    cache_path = _cache_path(disaster_number, county)
    cache_all_path = _cache_all_path(disaster_number)

    if csv_path.exists() and not refresh:
        # low_memory=False: a few columns (e.g. occupantsUnderTwo, emergencyNeeds) mix
        # numeric-looking and text/missing values, which trips pandas' chunked dtype
        # inference (DtypeWarning) without this. clean() in src/features/prepare.py
        # already encodes these columns explicitly, so we want them read as plain
        # object dtype here, not whatever pandas half-guesses per chunk.
        return pd.read_csv(csv_path, low_memory=False)

    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if cache_all_path.exists() and not refresh:
        everything = pd.read_parquet(cache_all_path)
    else:
        # County filtered server-side: fewer rows than pulling the whole state.
        everything = download_disaster(disaster_number=disaster_number, county=county)
        if everything.empty:
            raise ValueError(
                f"OpenFEMA returned no rows for DR-{disaster_number} county={county!r}. "
                f"Check the spelling (the API uses e.g. 'Lee (County)')."
            )

    col = next((c for c in ("county", "countyName") if c in everything.columns), None)
    if col is None:
        raise KeyError(f"no county column found; got {list(everything.columns)}")

    subset = everything[everything[col] == county]
    if subset.empty:
        near = sorted(
            v for v in everything[col].dropna().unique()
            if _safe_name(county) in _safe_name(str(v))
        )
        raise ValueError(
            f"No rows for county={county!r}. Did you mean one of {near}? "
            f"Pass the right string as load_ian_lee(county=...)."
        )

    subset.to_parquet(cache_path, index=False)
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
    refresh = "--refresh" in sys.argv
    data = load_ian_lee(refresh=refresh)
    describe(data)
    print(f"\ncached at {_cache_path(IAN_DR, COUNTY).relative_to(REPO_ROOT)}")
