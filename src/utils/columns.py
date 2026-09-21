"""Single source of truth for what every column in the FEMA dataset *is*.

Everyone on the team (EDA, cleaning, modeling) imports from here so we all
agree on which columns are allowed as model inputs. If the team changes its
mind about a column, change it HERE, once.

Why this matters: many columns in this dataset are *outcomes* of the very
decision we want to predict. Feeding them to a model is "data leakage" -- the
model looks perfect on paper and is useless in real life. See docs/data_dictionary.md.

Evidence for these roles: run notebooks/eda/01_initial_eda.ipynb (leakage section).
"""

# --- The thing we predict -----------------------------------------------------
# True  = FEMA awarded the applicant Individuals & Households Program (IHP) aid.
# False = valid registrant, but not awarded IHP aid.
# Balanced (~51% / 49%), so plain accuracy is meaningful. Team can change this.
TARGET = "ihpEligible"

# --- Not useful as inputs: IDs, dates of the disaster, and columns that never vary
# (we filtered to one disaster in one county, so these are identical on every row).
ID_OR_CONSTANT = [
    "id", "lastRefresh", "declarationDate", "disasterNumber", "incidentTypeCode",
    "damagedStateAbbreviation", "county", "fips", "censusYear",
    "foundationDamage", "foundationDamageAmount", "roofDamage", "roofDamageAmount",
    "ihpMax", "onaMax",
]

# --- LEAKAGE: results of the aid decision. NEVER use as model inputs. ---------
# e.g. ihpAmount > 0 is *exactly* the same thing as ihpEligible (AUC = 1.000).
LEAKAGE = [
    "ihpAmount", "haAmount", "onaAmount", "haEligible", "onaEligible", "haStatus", "haMax",
    "ihpReferral", "haReferral", "onaReferral",
    "ineligibleReason", "ineligibleInsurance", "insufficientDamage",
    "fipAmount", "sbaApproved", "tsaEligible", "tsaCheckedIn",
    "transientAccommodEligible", "transientAccommodAmount",
    "onaMedicalAssistEligible", "onaMedicalAssistAmount",
    "onaDentalAssistEligible", "onaDentalAssistAmount",
    "onaFuneralAssistEligible", "onaFuneralAssistAmount",
    "onaMovingAssistEligible", "onaMovingAssistAmount",
    "onaOtherAssistEligible", "onaOtherAssistAmount",
    "rentalAssistanceEligible", "rentalAssistanceAmount", "rentalAssistanceEndDate",
    "rentalResourceCity", "rentalResourceStateAbbrev", "rentalResourceZipCode",
    "repairAssistanceEligible", "repairAmount",
    "replacementAssistanceEligible", "replacementAmount",
    "personalPropertyEligible", "personalPropertyAmount",
    "unmetNeedRp", "unmetNeedPp",
]

# --- TIER 1: known when the applicant registers (self-reported) ---------------
# Use case: "Can FEMA triage applications the moment they arrive?"
APPLICATION_FEATURES = [
    "appliedDate",
    "applicantAge", "householdComposition",
    "occupantsUnderTwo", "occupants2to5", "occupants6to18", "occupants19to64", "occupants65andOver",
    "grossIncome", "ownRent", "primaryResidence", "residenceType",
    "homeOwnersInsurance", "floodInsurance", "registrationMethod",
    "damagedCity", "damagedZipCode", "censusGeoid",
    "homeDamage", "autoDamage", "utilitiesOut", "reportedDamage", "selfAssessmentInformation",
    "emergencyNeeds", "foodNeed", "shelterNeed", "accessFunctionalNeeds",
]

# --- TIER 2: only known after a FEMA inspector / verification step ------------
# Legitimate inputs, but for a *later* prediction moment ("after inspection").
# Add these on top of Tier 1 for a stronger, later-stage model.
INSPECTION_FEATURES = [
    "inspnIssued", "inspnReturned",
    "rpfvl", "ppfvl",                      # FEMA-verified loss: real / personal property ($)
    "floodDamage", "floodDamageAmount", "waterLevel", "highWaterLocation",
    "destroyed", "renterDamageLevel",
    "verifiedOwnership", "verifiedOccupancy", "habitabilityRepairsRequired",
    # Moved out of Tier 1 after a leakage check: this field can be updated after registration and
    # includes post-aid places ("MHU" FEMA Provided Unit = 99% eligible, "R" new rental = 87%).
    # Dropping it from Tier 1 cost only ~0.013 AUC (0.913 -> 0.900).
    "currentLocation",
]

ALL_ROLES = {
    "target": [TARGET],
    "id_or_constant": ID_OR_CONSTANT,
    "leakage": LEAKAGE,
    "application": APPLICATION_FEATURES,
    "inspection": INSPECTION_FEATURES,
}


def role_of(column: str) -> str:
    """Which bucket a column belongs to, e.g. role_of('ihpAmount') -> 'leakage'."""
    for role, cols in ALL_ROLES.items():
        if column in cols:
            return role
    return "unassigned"


def check_roles(columns) -> None:
    """Fail loudly if the data has a column we haven't assigned, or roles overlap."""
    assigned = [c for cols in ALL_ROLES.values() for c in cols]
    dupes = {c for c in assigned if assigned.count(c) > 1}
    if dupes:
        raise ValueError(f"Columns in more than one role: {sorted(dupes)}")
    unassigned = set(columns) - set(assigned)
    if unassigned:
        raise ValueError(f"Columns with no role yet (add them to src/utils/columns.py): {sorted(unassigned)}")
