"""Regenerate docs/data_dictionary.md from FEMA's official field descriptions + our column roles.

    python -m src.utils.build_data_dictionary

Run this whenever src/utils/columns.py changes, so the dictionary never drifts from the code.
Needs internet (it reads field descriptions from the OpenFEMA metadata API).
"""

from __future__ import annotations

import re

import pandas as pd
import requests

from src.data.fetch_fema import COUNTY, DATASET, IAN_DR, REPO_ROOT, _cache_path
from src.utils.columns import (
    APPLICATION_FEATURES, ID_OR_CONSTANT, INSPECTION_FEATURES, LEAKAGE, TARGET, UNDER_REVIEW,
)

META_URL = "https://www.fema.gov/api/open/v1/OpenFemaDataSetFields"
OUT = REPO_ROOT / "docs" / "data_dictionary.md"

# The team's own notes: why a column has the role it has, and how cleaning treats it.
NOTES = {
    "ihpEligible": "**Target.** True = received a Housing Assistance and/or Other Needs award. Identical to `ihpAmount > 0` on every row.",
    "appliedDate": "Converted to `daysSinceLandfall` (days after 2022-09-28).",
    "applicantAge": "Ordered numbers 0-4 (`<19` to `65+`).",
    "householdComposition": "Ordered numbers; `>5` becomes 6.",
    "occupantsUnderTwo": "Ordered numbers; `>5` becomes 6.",
    "occupants2to5": "Ordered numbers; `>5` becomes 6.",
    "occupants6to18": "Ordered numbers; `>5` becomes 6.",
    "occupants19to64": "Ordered numbers; `>5` becomes 6.",
    "occupants65andOver": "Ordered numbers; `>5` becomes 6.",
    "grossIncome": "Kept as an unordered category: `0` behaves differently from `<$15,000`.",
    "damagedCity": "**Dropped in cleaning:** user-entered with many spellings. Use ZIP / census block instead.",
    "damagedZipCode": "High-cardinality category (~560 values); frequency-encode inside the model pipeline.",
    "censusGeoid": "High-cardinality category (~3,000 values); ~3% are outside Lee County.",
    "emergencyNeeds": "Strongest Tier 1 signal (34% eligible if No, 66% if Yes).",
    "foodNeed": "Blank effectively means 'need not reported' (nearly always True when present). Keep a missing indicator.",
    "shelterNeed": "Blank effectively means 'need not reported'. Keep a missing indicator.",
    "primaryResidence": "Near-rule: non-primary homes are ~0% eligible.",
    "verifiedOccupancy": "Near-rule: unverified occupancy is <1% eligible.",
    "habitabilityRepairsRequired": "Blank for 72.6% of rows. Eligible 37% when blank vs 88% when answered, so whether it was recorded is itself a strong signal; keep a missing indicator.",
    "currentLocation": "**Moved out of Tier 1** after a leakage check: can change after registration and includes post-aid values (FEMA-provided unit 99% eligible, new rental 87%).",
    "rpfvl": "Zero for 83% of rows; heavy right skew, plan `log1p`.",
    "ppfvl": "Zero for 81% of rows; heavy right skew, plan `log1p`.",
    "waterLevel": "Zero for 88% of rows; the 960-inch maximum is implausible; plan to cap.",
    "floodDamage": "Yes for 14% of rows: eligible 94% if yes, 44% if no.",
    "floodDamageAmount": "Zero for 87% of rows; heavy right skew, plan `log1p`.",
    "inspnIssued": "Yes for 36% of rows: eligible 73% if yes, 38% if no. Identical to `inspnReturned` (r = 1.00).",
    "inspnReturned": "Yes for 36% of rows: eligible 73% if yes, 38% if no. Identical to `inspnIssued` (r = 1.00).",
    "destroyed": "Yes for 1.5% of rows: eligible 99% if yes, 50% if no.",
    "highWaterLocation": "Blank for 88% of rows. Eligible 45% when blank vs 92% when answered; keep a missing indicator.",
    "renterDamageLevel": "Blank for 96% of rows. Eligible 49% when blank vs 89% when answered; keep a missing indicator.",
    "utilitiesOut": "**Moved out of Tier 1** until FEMA confirms when the field is populated. Blank rows (2.7%) are 99.9% eligible and follow a distinct award path (mostly first-week applicants on rental assistance), so the field may be filled in after the decision.",
    "ihpAmount": "Cramér's V = 1.00 with eligibility: same information as the target.",
    "ineligibleReason": "Only filled in for people who were referred and denied. Blank = never referred.",
    "ihpReferral": "False = never referred, and 0% of those are eligible (a process gate, not a cause).",
    "onaEligible": "Cramér's V about 0.95 with eligibility (leakage).",
    "onaAmount": "Cramér's V about 0.95 with eligibility (leakage).",
}

SECTIONS = [
    ("Target (what we predict)", [TARGET]),
    ("Tier 1: Application-time features (known when the person registers)", APPLICATION_FEATURES),
    ("Tier 2: Later-stage features (known only after inspection / verification, or updated after registration)", INSPECTION_FEATURES),
    ("Leakage: results of the aid decision (**never use as model inputs**)", LEAKAGE),
    ("Under review: held out of both tiers until FEMA confirms timing", UNDER_REVIEW),
    ("Identifiers and constants (no information)", ID_OR_CONSTANT),
]


def _fetch_fields() -> dict:
    params = {"$filter": f"openFemaDataSet eq '{DATASET}'", "$top": 500, "$select": "name,type,description"}
    rows = requests.get(META_URL, params=params, timeout=120).json()["OpenFemaDataSetFields"]
    return {r["name"]: r for r in rows}


def _describe(info: dict, name: str) -> str:
    text = re.sub(r"\s+", " ", info[name]["description"]).strip()
    if name == "censusGeoid":
        text = text.split(" Please see")[0].strip() + " (`NO_INTERSECT` = address did not match a census block)."
    return text.replace("|", "/")


def main() -> None:
    info = _fetch_fields()
    df = pd.read_parquet(_cache_path(IAN_DR, COUNTY))
    miss = (df.isna().mean() * 100).round(1)
    n_app, n_insp = len(APPLICATION_FEATURES), len(INSPECTION_FEATURES)

    lines = [f"""# Data Dictionary

*Generated by `python -m src.utils.build_data_dictionary`. Do not edit by hand: change `src/utils/columns.py` and re-run.*

**Dataset:** FEMA OpenFEMA *Individuals and Households Program - Valid Registrations (v2)*, filtered to **Hurricane Ian (DR-4673)**, **Lee County, FL**.
{len(df):,} rows x {df.shape[1]} columns. One row = one household's application. Only *valid* registrations are included.

- Official field definitions: https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v2
- The "Meaning" column is FEMA's own wording (from the OpenFEMA metadata API). The section a column sits in (its role) and the "Our handling" column are the team's decisions.
- Missing % is on the full {len(df):,}-row dataset.

## How the {df.shape[1]} columns split

| Role | Columns | Meaning |
|---|---|---|
| Target | 1 | what we predict |
| Tier 1: application | {n_app} | self-reported at registration |
| Tier 2: later-stage | {n_insp} | known only after an inspector / verification, or updated after registration |
| Leakage | {len(LEAKAGE)} | outcomes of the decision; using them would be cheating |
| Under review | {len(UNDER_REVIEW)} | held out of both tiers until FEMA confirms timing |
| ID / constant | {len(ID_OR_CONSTANT)} | identifiers, or identical on every row |

Cleaning (`src/features/prepare.py`) also drops `damagedCity` and replaces `appliedDate` with `daysSinceLandfall`, so a model sees **{n_app - 1}** Tier 1 features, or **{n_app - 1 + n_insp}** with Tier 2.

## Derived column

| Column | How it is made |
|---|---|
| `daysSinceLandfall` | days between 2022-09-28 (Hurricane Ian landfall) and `appliedDate`; replaces `appliedDate` |
"""]
    for title, cols in SECTIONS:
        lines.append(f"\n## {title}\n\n| Column | Type | Missing % | Meaning (FEMA) | Our handling |\n|---|---|---|---|---|")
        for c in cols:
            lines.append(f"| `{c}` | {info[c]['type']} | {miss[c]} | {_describe(info, c)} | {NOTES.get(c, '')} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({sum(len(c) for _, c in SECTIONS)} columns)")


if __name__ == "__main__":
    main()
