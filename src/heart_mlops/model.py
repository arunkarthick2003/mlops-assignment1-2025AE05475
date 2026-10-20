"""Model construction and evaluation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-safe preprocessing fitted only inside each training fold."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_model(model_name: str, **parameters: Any) -> Pipeline:
    """Create one of the assignment's candidate classifiers with preprocessing."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {', '.join(models)}.")
    classifier = models[model_name].set_params(**parameters)
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Check inference/training features and return them in the canonical order."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    return frame.loc[:, FEATURES].copy()