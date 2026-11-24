"""Single-passenger prediction: validated input → probability → explanation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from src.config import ID_COL, TARGET
from src.feature_engineering import TitanicFeatureEngineer
from src.preprocessing import build_preprocessor

TRAINING_RANGES = {"Age": (0.42, 80.0), "Fare": (0.0, 512.3292), "SibSp": (0, 8), "Parch": (0, 6)}
PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
TITLES = ["Auto", "Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Col", "Major", "Lady", "Sir", "Countess"]


@dataclass
class PassengerInput:
    pclass: int
    sex: str
    age: float | None
    sibsp: int
    parch: int
    fare: float | None
    embarked: str
    cabin: str | None = None
    title: str | None = None
    ticket: str | None = None
    name: str | None = None

    def validate(self) -> tuple[list[str], list[str]]:
        """Return (errors, warnings)."""
        errors, warnings = [], []
        if self.pclass not in (1, 2, 3):
            errors.append("Passenger class must be 1, 2 or 3.")
        if str(self.sex).lower() not in ("male", "female"):
            errors.append("Sex must be male or female.")
        if self.embarked not in PORTS:
            errors.append("Port must be Southampton (S), Cherbourg (C) or Queenstown (Q).")
        if self.age is not None:
            # 0 is allowed: it stands for an infant under one year (the form's age slider starts at 0
            # and real passengers such as #804, aged 0.42, round to it).
            if not np.isfinite(self.age):
                errors.append("Age must be a number (leave it empty / tick 'Age unknown' if it is not known).")
            elif not 0 <= self.age <= 100:
                errors.append("Age must be between 0 and 100.")
            elif self.age > TRAINING_RANGES["Age"][1]:
                warnings.append(f"Age {self.age:g} is above the oldest passenger in the training data (80) — the model is extrapolating.")
        for name, v in (("Siblings/spouses", self.sibsp), ("Parents/children", self.parch)):
            if v is None or v < 0 or v > 20:
                errors.append(f"{name} must be between 0 and 20.")
        if self.sibsp is not None and self.sibsp > TRAINING_RANGES["SibSp"][1]:
            warnings.append("Siblings/spouses is above the training maximum (8).")
        if self.parch is not None and self.parch > TRAINING_RANGES["Parch"][1]:
            warnings.append("Parents/children is above the training maximum (6).")
        if self.fare is not None:
            if not np.isfinite(self.fare):
                errors.append("Fare must be a finite number (leave it empty if it is not known).")
            elif self.fare < 0:
                errors.append("Fare cannot be negative.")
            elif self.fare > TRAINING_RANGES["Fare"][1]:
                warnings.append("Fare is above the highest fare in the training data (£512.33).")
            typical = {1: (25, 520), 2: (10, 75), 3: (0, 70)}[self.pclass] if self.pclass in (1, 2, 3) and np.isfinite(self.fare) else None
            if typical and not (typical[0] <= self.fare <= typical[1]) and self.fare > 0:
                warnings.append(f"A fare of £{self.fare:.2f} is unusual for {self.pclass}{'st' if self.pclass == 1 else 'nd' if self.pclass == 2 else 'rd'} class.")
        if self.cabin:
            c = self.cabin.strip().upper()
            if not c[0].isalpha() or c[0] not in "ABCDEFGT":
                warnings.append("Cabin should start with a deck letter A–G (e.g. C85); deck will be treated as Unknown.")
        if self.age is not None and np.isfinite(self.age) and self.age < 13 and (self.sibsp or 0) + (self.parch or 0) == 0:
            warnings.append("A child travelling with no family recorded is rare in the data.")
        return errors, warnings

    def to_frame(self) -> pd.DataFrame:
        title = None if (not self.title or self.title == "Auto") else self.title
        return pd.DataFrame([{
            ID_COL: 0, "Pclass": int(self.pclass), "Name": self.name or np.nan, "Sex": str(self.sex).lower(),
            "Age": np.nan if self.age is None else float(self.age), "SibSp": int(self.sibsp), "Parch": int(self.parch),
            "Ticket": self.ticket or np.nan, "Fare": np.nan if self.fare is None else float(self.fare),
            "Cabin": (self.cabin.strip().upper() if self.cabin else np.nan), "Embarked": self.embarked, "Title": title or np.nan,
        }])


def predict_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    prob = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"prob_survived": prob, "prob_not_survived": 1 - prob, "prediction": (prob >= 0.5).astype(int)}, index=X.index)


def predict_one(pipeline, inp: PassengerInput) -> dict:
    errors, warnings = inp.validate()
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    X = inp.to_frame()
    p = float(pipeline.predict_proba(X)[0, 1])
    return {
        "ok": True, "errors": [], "warnings": warnings, "prob_survived": p, "prob_not_survived": 1 - p,
        "prediction": int(p >= 0.5), "label": "Survived" if p >= 0.5 else "Did not survive",
        "confidence": confidence_label(p), "frame": X,
    }


def confidence_label(p: float) -> str:
    d = abs(p - 0.5)
    if d >= 0.35:
        return "High"
    if d >= 0.15:
        return "Moderate"
    return "Low — close to a coin flip"


def what_if_curve(pipeline, inp: PassengerInput, feature: str, values) -> pd.DataFrame:
    """Predicted probability as one input varies (others held fixed) — an ICE curve."""
    base = inp.to_frame()
    rows = pd.concat([base] * len(values), ignore_index=True)
    rows[feature] = list(values)
    prob = pipeline.predict_proba(rows)[:, 1]
    return pd.DataFrame({feature: list(values), "prob_survived": prob})


class SimilarPassengerIndex:
    """Nearest real passengers in a standardised feature space (fit on labelled data)."""

    def __init__(self, df: pd.DataFrame, k: int = 5):
        self.df = df.reset_index(drop=True)
        self.k = k
        self.encoder = Pipeline([
            ("features", TitanicFeatureEngineer(feature_set="base")),
            ("preprocess", build_preprocessor(scaler="standard", pclass_encoding="onehot")),
        ]).fit(self.df.drop(columns=[TARGET], errors="ignore"))
        self.matrix = self.encoder.transform(self.df.drop(columns=[TARGET], errors="ignore"))
        self.nn = NearestNeighbors(n_neighbors=min(k, len(self.df))).fit(self.matrix)

    def query(self, X: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        dist, idx = self.nn.kneighbors(self.encoder.transform(X), n_neighbors=k or self.k)
        out = self.df.iloc[idx[0]].copy()
        out.insert(0, "similarity", 1 / (1 + dist[0]))
        return out.reset_index(drop=True)


# Clearly fictional demonstration profiles (not historical records).
SAMPLE_PROFILES = {
    "Fictional · First-class mother": PassengerInput(1, "female", 38, 1, 2, 120.0, "C", cabin="B45", title="Mrs"),
    "Fictional · Third-class young man": PassengerInput(3, "male", 22, 0, 0, 7.25, "S", title="Mr"),
    "Fictional · Second-class boy": PassengerInput(2, "male", 6, 1, 1, 26.0, "S", title="Master"),
    "Fictional · Third-class large family mother": PassengerInput(3, "female", 41, 1, 5, 34.4, "S", title="Mrs"),
    "Fictional · First-class older gentleman": PassengerInput(1, "male", 64, 0, 0, 52.0, "S", cabin="C23", title="Mr"),
    "Fictional · Queenstown young woman": PassengerInput(3, "female", 19, 0, 0, 7.75, "Q", title="Miss"),
}


def input_from_row(row: pd.Series) -> PassengerInput:
    """Build an input from a real dataset row (used for 'load a real passenger')."""
    return PassengerInput(
        pclass=int(row["Pclass"]), sex=str(row["Sex"]), age=None if pd.isna(row["Age"]) else float(row["Age"]),
        sibsp=int(row["SibSp"]), parch=int(row["Parch"]), fare=None if pd.isna(row["Fare"]) else float(row["Fare"]),
        embarked=str(row["Embarked"]) if pd.notna(row["Embarked"]) else "S",
        cabin=None if pd.isna(row.get("Cabin")) else str(row["Cabin"]), ticket=str(row["Ticket"]) if pd.notna(row.get("Ticket")) else None,
        name=str(row["Name"]) if pd.notna(row.get("Name")) else None,
    )
