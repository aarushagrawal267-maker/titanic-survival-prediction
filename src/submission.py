"""Kaggle competition submission: generation and strict validation."""
from __future__ import annotations

import pandas as pd

from src import config
from src.config import ID_COL, TARGET


def make_submission(pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    pred = pipeline.predict(test_df)
    return pd.DataFrame({ID_COL: test_df[ID_COL].astype(int).to_numpy(), TARGET: pd.Series(pred).astype(int).to_numpy()})


def validate_submission(sub: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    errors = []
    if list(sub.columns) != [ID_COL, TARGET]:
        errors.append(f"Columns must be exactly [{ID_COL}, {TARGET}] — got {list(sub.columns)}.")
    if len(sub) != len(test_df):
        errors.append(f"Row count {len(sub)} does not match test.csv ({len(test_df)}).")
    if ID_COL in sub.columns:
        if sub[ID_COL].duplicated().any():
            errors.append("Duplicate PassengerId values.")
        if set(sub[ID_COL]) != set(test_df[ID_COL]):
            errors.append("PassengerId values do not match test.csv.")
        elif not (sub[ID_COL].to_numpy() == test_df[ID_COL].to_numpy()).all():
            errors.append("PassengerId order differs from test.csv.")
    if TARGET in sub.columns:
        if sub[TARGET].isna().any():
            errors.append("Survived contains missing values.")
        elif not sub[TARGET].isin([0, 1]).all():
            errors.append("Survived must contain only 0 or 1.")
        elif not pd.api.types.is_integer_dtype(sub[TARGET]):
            errors.append("Survived must be an integer column.")
    return errors


def save_submission(sub: pd.DataFrame, path=None):
    path = path or (config.PROJECT_ROOT / config.get("paths", "submission_file"))
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path
