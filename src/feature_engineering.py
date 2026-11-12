"""Feature engineering as a scikit-learn transformer.

``TitanicFeatureEngineer`` learns everything it needs (ticket frequencies, age
medians per title) **only in ``fit``**, so inside a Pipeline it is refit on each
training fold and never sees validation/holdout rows. It never uses the target.

Engineered features
-------------------
FamilySize, IsAlone, Title, TitleGroup, FarePerPerson, AgeGroup, FamilyCategory,
CabinKnown, Deck, TicketGroupSize (+ AgeKnown, a missingness indicator).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_GROUPS = {
    "Mr": "Mr",
    "Mrs": "Mrs", "Mme": "Mrs", "Countess": "Mrs", "Lady": "Mrs", "Dona": "Mrs",
    "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Master": "Master",
    # Professional, military, clergy and other rare titles
    "Dr": "Other", "Rev": "Other", "Col": "Other", "Major": "Other", "Capt": "Other",
    "Sir": "Other", "Don": "Other", "Jonkheer": "Other",
}
TITLE_ORDER = ["Mr", "Mrs", "Miss", "Master", "Other"]
AGE_BINS = [0, 12, 17, 30, 50, 200]
AGE_LABELS = ["Child (0–12)", "Teen (13–17)", "Young adult (18–30)", "Adult (31–50)", "Senior (51+)"]
DECKS = list("ABCDEFG")

BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
ENGINEERED_FEATURES = [
    "FamilySize", "IsAlone", "TitleGroup", "FarePerPerson", "AgeGroup",
    "FamilyCategory", "CabinKnown", "Deck", "TicketGroupSize", "AgeKnown",
]
FEATURE_SETS = {
    "base": BASE_FEATURES,
    "engineered": BASE_FEATURES + ENGINEERED_FEATURES,
}

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "FarePerPerson", "TicketGroupSize"]
BINARY_FEATURES = ["IsAlone", "CabinKnown", "AgeKnown"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "TitleGroup", "AgeGroup", "FamilyCategory", "Deck"]
# Pclass is handled separately (one-hot or ordinal — see preprocessing.py)

FEATURE_DESCRIPTIONS = {
    "FamilySize": "SibSp + Parch + 1 — the size of the passenger's family group aboard.",
    "IsAlone": "1 if FamilySize equals 1 (travelling without family).",
    "Title": "Honorific extracted from the name (Mr, Mrs, Miss, Master, Dr, Rev, …).",
    "TitleGroup": "Titles grouped into Mr, Mrs, Miss, Master and Other — encodes sex, age and marital status.",
    "FarePerPerson": "Fare ÷ FamilySize — a rough per-person price of the ticket.",
    "AgeGroup": "Age bucketed into Child, Teen, Young adult, Adult and Senior.",
    "FamilyCategory": "Alone (1), Small family (2–4), Large family (5+).",
    "CabinKnown": "1 if a cabin number is recorded. Missingness itself is informative.",
    "Deck": "First letter of the cabin (A–G); 'Unknown' when no cabin is recorded.",
    "TicketGroupSize": "How many passengers share the ticket number (learned from training data only; no target used).",
    "AgeKnown": "1 if the age was recorded (0 = age was imputed).",
}


def extract_title(name) -> str | None:
    if not isinstance(name, str):
        return None
    m = re.search(r",\s*([^\.]+)\.", name)
    if not m:
        return None
    title = m.group(1).strip()
    if title == "the Countess":
        title = "Countess"
    return title


def group_title(title) -> str | None:
    if not isinstance(title, str) or not title:
        return None
    if title in TITLE_GROUPS:
        return TITLE_GROUPS[title]
    return title if title in TITLE_ORDER else "Other"


def infer_title_group(sex, age, sibsp, parch) -> str:
    """Heuristic title when no name is available (e.g. the prediction form)."""
    is_male = str(sex).lower() == "male"
    young = pd.notna(age) and float(age) < 13
    if is_male:
        return "Master" if young else "Mr"
    if pd.notna(age) and float(age) < 18:
        return "Miss"
    return "Mrs" if (pd.notna(sibsp) and sibsp > 0) or (pd.notna(parch) and parch > 0) else "Miss"


def extract_deck(cabin) -> str:
    if not isinstance(cabin, str) or not cabin.strip():
        return "Unknown"
    letter = cabin.strip()[0].upper()
    return letter if letter in DECKS else "Unknown"


def family_category(size: float) -> str:
    if size <= 1:
        return "Alone"
    return "Small family" if size <= 4 else "Large family"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless features (no fitting needed) — used by EDA and by the transformer."""
    out = df.copy()
    sibsp = pd.to_numeric(out.get("SibSp"), errors="coerce").fillna(0)
    parch = pd.to_numeric(out.get("Parch"), errors="coerce").fillna(0)
    out["FamilySize"] = sibsp + parch + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["FamilyCategory"] = out["FamilySize"].map(family_category)
    names = out["Name"] if "Name" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["Title"] = names.map(extract_title)
    if "Title" in df.columns:  # an explicit Title column (prediction form) wins
        explicit = df["Title"].where(df["Title"].notna() & (df["Title"].astype(str).str.len() > 0))
        out["Title"] = explicit.combine_first(out["Title"])
    groups = out["Title"].map(group_title)
    missing = groups.isna()
    if missing.any():
        groups[missing] = [
            infer_title_group(r.get("Sex"), r.get("Age"), r.get("SibSp"), r.get("Parch"))
            for _, r in out.loc[missing].iterrows()
        ]
    out["TitleGroup"] = groups
    cabin = out["Cabin"] if "Cabin" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    out["CabinKnown"] = cabin.map(lambda c: int(isinstance(c, str) and bool(c.strip())))
    out["Deck"] = cabin.map(extract_deck)
    fare = pd.to_numeric(out.get("Fare"), errors="coerce")
    out["FarePerPerson"] = fare / out["FamilySize"]
    age = pd.to_numeric(out.get("Age"), errors="coerce")
    out["AgeKnown"] = age.notna().astype(int)
    return out


def age_group(age: pd.Series) -> pd.Series:
    cats = pd.cut(pd.to_numeric(age, errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS, right=True, include_lowest=True)
    return cats.astype(object).where(cats.notna(), "Unknown")


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turn raw Titanic columns into model-ready features.

    Parameters
    ----------
    feature_set : "base" | "engineered"
    drop_features : features to exclude (used by the ablation study)
    age_strategy : "title_group" imputes Age with the training median of each
        TitleGroup (learned in ``fit``); "median" leaves Age missing so the
        ColumnTransformer's median imputer handles it.
    """

    def __init__(self, feature_set: str = "engineered", drop_features: tuple = (), age_strategy: str = "title_group"):
        self.feature_set = feature_set
        self.drop_features = drop_features
        self.age_strategy = age_strategy

    # ── fitting ────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y=None):
        X = self._ensure_frame(X)
        basic = add_basic_features(X)
        age = pd.to_numeric(basic["Age"], errors="coerce")
        self.age_median_ = float(age.median()) if age.notna().any() else 28.0
        self.age_by_title_ = age.groupby(basic["TitleGroup"]).median().dropna().to_dict()
        tickets = X["Ticket"] if "Ticket" in X.columns else pd.Series(dtype=object)
        self.ticket_counts_ = tickets.dropna().astype(str).value_counts().to_dict()
        self.output_features_ = self.get_feature_names_out()
        return self

    # ── transforming ───────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_frame(X)
        out = add_basic_features(X)
        out["Pclass"] = pd.to_numeric(out["Pclass"], errors="coerce")
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
        if self.age_strategy == "title_group":
            fill = out["TitleGroup"].map(self.age_by_title_).fillna(self.age_median_)
            out["Age"] = out["Age"].fillna(fill)
        out["AgeGroup"] = age_group(out["Age"])
        out["TicketGroupSize"] = self._ticket_group_size(out)
        for col in ("SibSp", "Parch", "Fare"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in CATEGORICAL_FEATURES:
            if col in out.columns:
                out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
        cols = self.get_feature_names_out()
        return out[cols].copy()

    def _ticket_group_size(self, df: pd.DataFrame) -> pd.Series:
        """Passengers sharing a ticket: max(training count, count within this batch).

        Missing ticket → falls back to FamilySize. Never uses the target.
        """
        if "Ticket" not in df.columns:
            return df["FamilySize"].astype(float)
        tickets = df["Ticket"].astype(object)
        batch_counts = tickets.dropna().astype(str).value_counts()
        res = []
        for t, fam in zip(tickets, df["FamilySize"]):
            if not isinstance(t, str) or not t.strip():
                res.append(float(fam))
            else:
                res.append(float(max(self.ticket_counts_.get(t, 0), batch_counts.get(t, 1))))
        return pd.Series(res, index=df.index)

    def get_feature_names_out(self, input_features=None):
        cols = ["Pclass"] + [c for c in FEATURE_SETS[self.feature_set] if c != "Pclass"]
        return np.array([c for c in cols if c not in set(self.drop_features)], dtype=object)

    @staticmethod
    def _ensure_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TitanicFeatureEngineer expects a pandas DataFrame with raw Titanic columns.")
        return X
