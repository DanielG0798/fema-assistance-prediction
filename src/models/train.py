"""Fit the models and write the shared predictions file the evaluation and fairness work reads.

    python -m src.models.train            # both tiers -> data/processed/predictions_{tier}.parquet

The file is the hand-off between team members: evaluation and fairness code reads ONLY this
file, never the models themselves. Columns (one row per validation applicant):

    row_id         row position in the cleaned table (load_model_data), same for both tiers
    split          "val" (test rows are added once, at the Milestone 2 model freeze)
    y_true         ihpEligible, 1 = eligible
    prob_<model>   predicted probability of eligible, one column per model
    applicantAge   0-4 code from prepare.AGE_ORDER (for group error rates)
    grossIncome    income bracket as text (for group error rates)

The current models are the untuned baselines; tuned models replace them with the same columns.
"""

from __future__ import annotations

import pandas as pd

from src.features.prepare import PROCESSED_DIR, load_model_data, split_data
from src.models.baseline import add_frequency_features, make_models
from src.utils.columns import TARGET

GROUP_COLS = ["applicantAge", "grossIncome"]


def predict_validation(tier: str = "application") -> pd.DataFrame:
    """Fit every model on the training split and return validation predictions."""
    df = load_model_data(tier)
    df.insert(0, "row_id", range(len(df)))
    train, val, _test = split_data(df)
    train, val = add_frequency_features(train, val)

    X_tr, y_tr = train.drop(columns=[TARGET, "row_id"]), train[TARGET]
    X_va = val.drop(columns=[TARGET, "row_id"])

    out = pd.DataFrame({"row_id": val["row_id"], "split": "val", "y_true": val[TARGET]})
    for name, model in make_models(X_tr).items():
        model.fit(X_tr, y_tr)
        out[f"prob_{name}"] = model.predict_proba(X_va)[:, 1]
    out["applicantAge"] = val["applicantAge"]
    out["grossIncome"] = val["grossIncome"].astype(str)
    return out


def save_predictions(tier: str = "application") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds = predict_validation(tier)
    path = PROCESSED_DIR / f"predictions_{tier}.parquet"
    preds.to_parquet(path, index=False)
    print(f"{len(preds):,} rows, columns {list(preds.columns)} -> {path.name}")


if __name__ == "__main__":
    save_predictions("application")
    save_predictions("inspection")
