"""Dataset validation: schema, dtypes, missing values, duplicates, invalid values, target.

``validate_dataframe`` never mutates the input. It returns a ``ValidationReport``
with *errors* (dataset unusable), *warnings* (usable but noteworthy) and a
column-level summary table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET

FEATURE_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
TRAIN_COLUMNS = [ID_COL, TARGET] + FEATURE_COLUMNS
TEST_COLUMNS = [ID_COL] + FEATURE_COLUMNS
# Minimum needed to score a passenger; everything else is optional (imputed / engineered).
BATCH_REQUIRED = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
BATCH_OPTIONAL = [ID_COL, "Name", "Ticket", "Cabin", "Title"]

NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
VALID_SEX = {"male", "female"}
VALID_EMBARKED = {"C", "Q", "S"}
VALID_PCLASS = {1, 2, 3}

COLUMN_DESCRIPTIONS = {
    "PassengerId": "Unique row identifier assigned by Kaggle.",
    "Survived": "Target — 1 = survived, 0 = did not survive.",
    "Pclass": "Ticket class: 1 = Upper, 2 = Middle, 3 = Lower (a proxy for socio-economic status).",
    "Name": "Full name including a title (Mr, Mrs, Miss, Master, Dr, …).",
    "Sex": "male / female.",
    "Age": "Age in years (fractional below 1; estimated ages end in .5).",
    "SibSp": "Number of siblings or spouses aboard.",
    "Parch": "Number of parents or children aboard.",
    "Ticket": "Ticket number (shared tickets indicate travelling groups).",
    "Fare": "Ticket fare in pounds sterling (£) — paid per ticket, not per person.",
    "Cabin": "Cabin number; first letter is the deck. Mostly missing.",
    "Embarked": "Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton.",
}


@dataclass
class ValidationReport:
    kind: str
    n_rows: int
    n_cols: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    summary: pd.DataFrame | None = None
    duplicate_rows: int = 0
    duplicate_ids: int = 0

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "rows": self.n_rows, "columns": self.n_cols,
            "valid": self.is_valid, "errors": self.errors, "warnings": self.warnings,
            "info": self.info, "duplicate_rows": self.duplicate_rows, "duplicate_ids": self.duplicate_ids,
        }


def _column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "unique": int(s.nunique(dropna=True)),
            "example": "" if s.dropna().empty else str(s.dropna().iloc[0]),
            "description": COLUMN_DESCRIPTIONS.get(col, ""),
        })
    return pd.DataFrame(rows)


def validate_dataframe(df: pd.DataFrame, kind: str = "train") -> ValidationReport:
    """Validate a Titanic dataframe. ``kind`` ∈ {"train", "test", "batch"}."""
    rep = ValidationReport(kind=kind, n_rows=len(df), n_cols=df.shape[1])
    if df.empty:
        rep.errors.append("The file contains no rows.")
        return rep

    required = {"train": TRAIN_COLUMNS, "test": TEST_COLUMNS, "batch": BATCH_REQUIRED}[kind]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        rep.errors.append(f"Missing required column(s): {', '.join(missing_cols)}")
    known = set(TRAIN_COLUMNS) | set(BATCH_OPTIONAL)
    extra = [c for c in df.columns if c not in known]
    if extra:
        rep.info.append(f"Extra column(s) ignored by the model: {', '.join(map(str, extra))}")

    # Target checks
    if kind == "train" and TARGET in df.columns:
        t = pd.to_numeric(df[TARGET], errors="coerce")
        if t.isna().any():
            rep.errors.append(f"{TARGET} has {int(t.isna().sum())} missing/non-numeric value(s).")
        bad = (~t.dropna().isin([0, 1])).sum()
        if bad:
            rep.errors.append(f"{TARGET} must be 0 or 1 — found {int(bad)} invalid value(s).")
        if t.nunique(dropna=True) < 2:
            rep.errors.append(f"{TARGET} has only one class; a classifier cannot be trained.")
    if kind == "test" and TARGET in df.columns:
        rep.warnings.append(f"Test data contains a {TARGET} column — it will be ignored.")

    # Dtypes / numeric coercion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce")
            non_numeric = int((num.isna() & df[col].notna()).sum())
            if non_numeric:
                rep.errors.append(f"{col}: {non_numeric} value(s) are not numeric.")

    # Invalid values (only checked on non-missing values)
    if "Pclass" in df.columns:
        p = pd.to_numeric(df["Pclass"], errors="coerce").dropna()
        bad = int((~p.isin(list(VALID_PCLASS))).sum())
        if bad:
            rep.errors.append(f"Pclass must be 1, 2 or 3 — {bad} invalid value(s).")
        if df["Pclass"].isna().any():
            rep.errors.append("Pclass has missing values (required).")
    if "Sex" in df.columns:
        s = df["Sex"].dropna().astype(str).str.strip().str.lower()
        bad = int((~s.isin(VALID_SEX)).sum())
        if bad:
            rep.errors.append(f"Sex must be 'male' or 'female' — {bad} invalid value(s).")
        if df["Sex"].isna().any():
            rep.warnings.append(f"Sex is missing for {int(df['Sex'].isna().sum())} row(s) — will be imputed.")
    if "Embarked" in df.columns:
        e = df["Embarked"].dropna().astype(str).str.strip().str.upper()
        bad = int((~e.isin(VALID_EMBARKED)).sum())
        if bad:
            rep.errors.append(f"Embarked must be C, Q or S — {bad} invalid value(s).")
    for col, lo, hi in (("Age", 0, 100), ("Fare", 0, 1000), ("SibSp", 0, 20), ("Parch", 0, 20)):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = int(((v < lo) | (v > hi)).sum())
            if bad:
                rep.errors.append(f"{col} has {bad} value(s) outside the plausible range [{lo}, {hi}].")

    # Missing values
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n and col not in (TARGET,):
            pct = 100 * n / len(df)
            msg = f"{col}: {n} missing ({pct:.1f}%)."
            (rep.info if col in ("Age", "Cabin", "Embarked", "Fare", "Name", "Ticket") else rep.warnings).append(msg)

    # Duplicates
    rep.duplicate_rows = int(df.duplicated().sum())
    if rep.duplicate_rows:
        rep.warnings.append(f"{rep.duplicate_rows} fully duplicated row(s).")
    if ID_COL in df.columns:
        rep.duplicate_ids = int(df[ID_COL].duplicated().sum())
        if rep.duplicate_ids:
            msg = f"{rep.duplicate_ids} duplicated {ID_COL} value(s)."
            (rep.errors if kind in ("train", "test") else rep.warnings).append(msg)

    rep.summary = _column_summary(df)
    return rep


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: numeric columns coerced, Sex lower-cased, Embarked upper-cased.

    Missing optional columns are added as NaN so the pipeline sees a stable schema.
    """
    out = df.copy()
    for col in BATCH_REQUIRED + [c for c in BATCH_OPTIONAL if c != ID_COL]:
        if col not in out.columns:
            out[col] = np.nan
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Sex"] = out["Sex"].map(lambda v: str(v).strip().lower() if pd.notna(v) else np.nan)
    out["Embarked"] = out["Embarked"].map(lambda v: str(v).strip().upper() if pd.notna(v) else np.nan)
    for col in ("Name", "Ticket", "Cabin", "Title"):
        out[col] = out[col].map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else np.nan)
    return out
