"""Batch CSV scoring with validation and a simple input-drift check (PSI)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from src.config import ID_COL
from src.data_validation import BATCH_OPTIONAL, BATCH_REQUIRED, coerce_types, validate_dataframe
from src.prediction import predict_frame


def template_csv() -> str:
    demo = pd.DataFrame([
        {ID_COL: 1, "Pclass": 1, "Name": "Example, Mrs. Fictional", "Sex": "female", "Age": 35, "SibSp": 1, "Parch": 0, "Ticket": "X100", "Fare": 80.0, "Cabin": "C50", "Embarked": "C"},
        {ID_COL: 2, "Pclass": 3, "Name": "Example, Mr. Fictional", "Sex": "male", "Age": 28, "SibSp": 0, "Parch": 0, "Ticket": "X200", "Fare": 7.9, "Cabin": "", "Embarked": "S"},
        {ID_COL: 3, "Pclass": 2, "Name": "", "Sex": "female", "Age": "", "SibSp": 0, "Parch": 1, "Ticket": "", "Fare": 23.0, "Cabin": "", "Embarked": "S"},
    ])
    return demo.to_csv(index=False)


def read_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


def score_batch(pipeline, df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict]:
    """Validate and score. Returns (results, report_dict). Results is None if invalid."""
    rep = validate_dataframe(df, kind="batch")
    report = rep.as_dict()
    report["summary"] = rep.summary
    if not rep.is_valid:
        return None, report
    clean = coerce_types(df)
    preds = predict_frame(pipeline, clean)
    keep = [c for c in [ID_COL] + BATCH_REQUIRED + [c for c in BATCH_OPTIONAL if c != ID_COL] if c in df.columns]
    out = df[keep].copy()
    out["Survival probability"] = preds["prob_survived"].round(4)
    out["Non-survival probability"] = preds["prob_not_survived"].round(4)
    out["Predicted"] = preds["prediction"]
    out["Predicted outcome"] = out["Predicted"].map({1: "Survived", 0: "Did not survive"})
    return out, report


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index between training data and an uploaded batch."""
    e, a = expected.dropna(), actual.dropna()
    if e.empty or a.empty:
        return float("nan")
    if categorical:
        cats = sorted(set(e.astype(str)) | set(a.astype(str)))
        ep = e.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        ap = a.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        ep = np.histogram(e, edges)[0] / len(e)
        ap = np.histogram(a, edges)[0] / len(a)
    ep, ap = np.clip(ep, 1e-4, None), np.clip(ap, 1e-4, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def drift_report(train: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    b = coerce_types(batch)
    rows = []
    for col, cat in (("Pclass", True), ("Sex", True), ("Embarked", True), ("Age", False), ("Fare", False), ("SibSp", False), ("Parch", False)):
        if col in b.columns and col in train.columns:
            v = psi(train[col], b[col], categorical=cat)
            level = "Stable" if v < 0.1 else ("Moderate shift" if v < 0.25 else "Large shift")
            rows.append({"feature": col, "psi": round(v, 3), "status": level})
    return pd.DataFrame(rows)
