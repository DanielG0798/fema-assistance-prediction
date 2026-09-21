"""Turn the raw FEMA download into one clean, model-ready table.

Team usage:

    from src.features.prepare import load_model_data, split_data

    df = load_model_data()                       # Tier 1: application-time features
    df = load_model_data(tier="inspection")      # Tier 1 + Tier 2 (after FEMA inspection)
    train, val, test = split_data(df)            # 60 / 20 / 20, stratified

What this does (and deliberately does NOT do):
  DOES   drop IDs / constants / leakage, fix data types, decode ordered ranges into numbers,
         turn the date into "days since landfall".
  DOES NOT fill in missing values, scale numbers, or one-hot encode. Those steps "learn"
         from the data, so they must be fit on the TRAINING split only (inside a modeling
         pipeline) or the test score becomes dishonest. Missing values are left as NaN.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.fetch_fema import REPO_ROOT, load_ian_lee
from src.utils.columns import (
    APPLICATION_FEATURES,
    INSPECTION_FEATURES,
    TARGET,
    check_roles,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RANDOM_STATE = 42                       # same split for everyone, every run
LANDFALL = pd.Timestamp("2022-09-28")   # Hurricane Ian landfall, Lee County FL

# Ordered ranges -> numbers (so "older" / "bigger" is preserved for models).
AGE_ORDER = {"<19": 0, "19-34": 1, "35-49": 2, "50-64": 3, "65+": 4}
COUNT_ORDER = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, ">5": 6}   # ">5" -> 6

COUNT_COLS = [
    "householdComposition", "occupantsUnderTwo", "occupants2to5",
    "occupants6to18", "occupants19to64", "occupants65andOver",
]
# Codes / labels with no natural order -> kept as text categories.
CATEGORICAL_COLS = [
    "grossIncome", "ownRent", "residenceType", "registrationMethod", "currentLocation",
    "selfAssessmentInformation", "damagedZipCode", "censusGeoid",
    "highWaterLocation", "renterDamageLevel",
]
# NOTE grossIncome stays unordered on purpose: "0" (no/unreported income) does NOT behave like
# the lowest bracket -- eligibility is 42% for "0" but 61% for "<$15,000".
# damagedCity is dropped: FEMA says it is "user entered with no data checks" (typos).


def clean(raw: pd.DataFrame, tier: str = "application") -> pd.DataFrame:
    """Raw FEMA dataframe -> clean model table (target + chosen feature tier)."""
    if tier not in ("application", "inspection"):
        raise ValueError("tier must be 'application' or 'inspection'")
    check_roles(raw.columns)

    features = list(APPLICATION_FEATURES)
    if tier == "inspection":
        features += INSPECTION_FEATURES
    features.remove("damagedCity")
    df = raw[[TARGET] + features].copy()

    # Target: True/False -> 1/0
    df[TARGET] = df[TARGET].astype("int8")

    # Date -> a number a model can use: days between Ian landfall and the application
    applied = pd.to_datetime(df.pop("appliedDate"), utc=True).dt.tz_localize(None)
    df["daysSinceLandfall"] = (applied - LANDFALL).dt.days.astype("int16")

    # Ordered ranges -> numbers
    df["applicantAge"] = df["applicantAge"].map(AGE_ORDER).astype("int8")
    for c in COUNT_COLS:
        df[c] = df[c].map(COUNT_ORDER).astype("int8")

    # Yes/No columns -> 0/1, keeping "unknown" as a real missing value (not a fake 0)
    for c in df.columns:
        if c == TARGET:
            continue
        if df[c].map(type).isin([bool, type(None)]).all():
            df[c] = df[c].map({True: 1, False: 0}).astype("Int8")

    # Codes with no order -> categories
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df.reset_index(drop=True)


def load_model_data(tier: str = "application", refresh: bool = False) -> pd.DataFrame:
    """Download (first time only), clean, and return the model table."""
    return clean(load_ian_lee(refresh=refresh), tier=tier)


def split_data(df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2):
    """Stratified train / validation / test split (default 60 / 20 / 20).

    Stratified = each piece keeps the same ~51/49 eligible mix as the full data.
    Touch the test set ONCE, at the very end, to report the final score.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET], random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_processed(tier: str = "application") -> None:
    """Write the clean table to data/processed/ (the Milestone 1 'dataset' deliverable)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_model_data(tier=tier)
    path = PROCESSED_DIR / f"ihp_dr4673_lee_{tier}.parquet"
    df.to_parquet(path, index=False)
    print(f"{len(df):,} rows x {df.shape[1] - 1} features (+ target) -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    save_processed("application")
    save_processed("inspection")
