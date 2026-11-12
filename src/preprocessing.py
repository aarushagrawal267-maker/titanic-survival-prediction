"""Reusable, leakage-safe preprocessing.

Everything is assembled into one ``Pipeline``::

    TitanicFeatureEngineer → ColumnTransformer(impute → encode → scale) → model

so every statistic (medians, modes, category levels, scaler means) is learned
inside ``fit`` — i.e. on the training fold only during cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

from src.feature_engineering import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TitanicFeatureEngineer,
)

SCALERS = {"none": None, "standard": StandardScaler, "robust": RobustScaler}

# Human-readable names for model features (used across the UI and explanations).
FRIENDLY_NAMES = {
    "Pclass": "Passenger class", "Sex": "Sex", "Age": "Age", "SibSp": "Siblings/spouses",
    "Parch": "Parents/children", "Fare": "Fare", "Embarked": "Port of embarkation",
    "FamilySize": "Family size", "IsAlone": "Travelling alone", "TitleGroup": "Title",
    "FarePerPerson": "Fare per person", "AgeGroup": "Age group", "FamilyCategory": "Family category",
    "CabinKnown": "Cabin recorded", "Deck": "Deck", "TicketGroupSize": "Ticket group size",
    "AgeKnown": "Age recorded",
}


class ColumnsPresent:
    """Picklable column selector: keeps candidates that exist in X (supports ablations)."""

    def __init__(self, candidates):
        self.candidates = list(candidates)

    def __call__(self, X: pd.DataFrame):
        return [c for c in self.candidates if c in X.columns]


def _log1p_fare(X):
    return np.log1p(np.clip(X, 0, None))


def build_preprocessor(scaler: str = "standard", pclass_encoding: str = "onehot", log_fare: bool = False) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if log_fare:
        num_steps.append(("log", FunctionTransformer(_log1p_fare, feature_names_out="one-to-one")))
    if SCALERS.get(scaler) is not None:
        num_steps.append(("scale", SCALERS[scaler]()))
    numeric = Pipeline(num_steps)

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])

    num_cols = list(NUMERIC_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)
    if pclass_encoding == "onehot":
        cat_cols = ["Pclass"] + cat_cols
    else:
        num_cols = ["Pclass"] + num_cols

    return ColumnTransformer(
        transformers=[
            ("num", numeric, ColumnsPresent(num_cols)),
            ("cat", categorical, ColumnsPresent(cat_cols)),
            ("bin", binary, ColumnsPresent(BINARY_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    model,
    feature_set: str = "engineered",
    scaler: str = "standard",
    pclass_encoding: str = "onehot",
    age_strategy: str = "title_group",
    drop_features: tuple = (),
    log_fare: bool = False,
) -> Pipeline:
    return Pipeline([
        ("features", TitanicFeatureEngineer(feature_set=feature_set, drop_features=tuple(drop_features), age_strategy=age_strategy)),
        ("preprocess", build_preprocessor(scaler=scaler, pclass_encoding=pclass_encoding, log_fare=log_fare)),
        ("model", model),
    ])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Column names after the ColumnTransformer, e.g. ``cat__Sex_female``."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def original_feature(name: str) -> str:
    """Map a transformed column (``cat__Sex_female``) back to its model feature (``Sex``)."""
    prefix, _, rest = name.partition("__")
    if prefix in ("num", "bin"):
        return rest
    for col in ["Pclass"] + CATEGORICAL_FEATURES:
        if rest == col or rest.startswith(col + "_"):
            return col
    return rest


def friendly(name: str) -> str:
    return FRIENDLY_NAMES.get(name, name)


def preprocessing_preview(df: pd.DataFrame, n: int = 8, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(raw rows, engineered rows, fully transformed rows) for the Preprocessing page.

    Fitted on ``df`` itself — for display only, never for evaluation.
    """
    fe = TitanicFeatureEngineer(feature_set=kwargs.get("feature_set", "engineered"), age_strategy=kwargs.get("age_strategy", "title_group"))
    eng = fe.fit_transform(df)
    pre = build_preprocessor(scaler=kwargs.get("scaler", "standard"), pclass_encoding=kwargs.get("pclass_encoding", "onehot"))
    mat = pre.fit_transform(eng)
    out = pd.DataFrame(mat, columns=pre.get_feature_names_out(), index=df.index)
    return df.head(n), eng.head(n), out.head(n).round(3)
