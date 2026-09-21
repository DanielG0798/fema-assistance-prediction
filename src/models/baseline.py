"""Quick, honest baseline models so the team knows what "good" looks like.

    from src.models.baseline import run_baselines
    print(run_baselines("application"))     # Tier 1
    print(run_baselines("inspection"))      # Tier 1 + Tier 2

Scored on the VALIDATION split only. The test split is never touched here; save it for
the very end of Milestone 2. These models are untuned on purpose: they are a floor to beat.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.prepare import load_model_data, split_data
from src.utils.columns import TARGET

# ZIP (~500 values) and census block (~3,000) are too many categories for one-hot or for
# scikit-learn's boosting (cap: 255). We replace each by "how common is this place" (frequency),
# counted on the TRAINING rows only so nothing leaks from validation/test.
HIGH_CARDINALITY = ["damagedZipCode", "censusGeoid"]


def add_frequency_features(train: pd.DataFrame, *others: pd.DataFrame):
    """Swap high-cardinality columns for their training-set frequency. Returns copies."""
    frames = [d.copy() for d in (train, *others)]
    for col in HIGH_CARDINALITY:
        if col not in train.columns:
            continue
        freq = train[col].astype(str).value_counts()
        for d in frames:
            d[col + "_freq"] = d[col].astype(str).map(freq).fillna(0).astype(float)
            d.drop(columns=col, inplace=True)
    return frames


def run_baselines(tier: str = "application") -> pd.DataFrame:
    """Train a logistic regression and a gradient-boosting model; score on validation."""
    train, val, _test = split_data(load_model_data(tier))
    train, val = add_frequency_features(train, val)
    X_tr, y_tr = train.drop(columns=TARGET), train[TARGET]
    X_va, y_va = val.drop(columns=TARGET), val[TARGET]

    results = []

    # 1) Gradient boosting: handles missing values, categories, and interactions on its own.
    gb = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=200, random_state=0)
    gb.fit(X_tr, y_tr)
    results.append(_score("Gradient boosting", y_va, gb.predict_proba(X_va)[:, 1]))

    # 2) Logistic regression: simple, interpretable; needs imputing, scaling, one-hot encoding.
    numeric = [c for c in X_tr.columns if str(X_tr[c].dtype) in ("int8", "int16", "Int8", "float64")]
    categorical = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]

    def as_plain(X):
        X = X.copy()
        X[numeric] = X[numeric].astype("float64")
        X[categorical] = X[categorical].astype(str)
        return X

    prep = make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        (OneHotEncoder(handle_unknown="ignore"), categorical),
    )
    lr = make_pipeline(prep, LogisticRegression(max_iter=1000))
    lr.fit(as_plain(X_tr), y_tr)
    results.append(_score("Logistic regression", y_va, lr.predict_proba(as_plain(X_va))[:, 1]))

    majority = max(y_va.mean(), 1 - y_va.mean())
    results.append({"model": "Always guess the majority class", "ROC-AUC": 0.5, "accuracy": majority, "F1": 0.0})
    out = pd.DataFrame(results).set_index("model").round(3)
    out.insert(0, "features", X_tr.shape[1])
    return out


def _score(name: str, y_true, prob) -> dict:
    return {
        "model": name,
        "ROC-AUC": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, prob > 0.5),
        "F1": f1_score(y_true, prob > 0.5),
    }


if __name__ == "__main__":
    for tier in ("application", "inspection"):
        print(f"\n=== {tier}")
        print(run_baselines(tier))
