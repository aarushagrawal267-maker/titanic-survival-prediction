"""Exploratory data analysis — every number here is computed from the loaded data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import TARGET
from src.feature_engineering import add_basic_features, age_group

PORT_NAMES = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
CLASS_NAMES = {1: "1st class", 2: "2nd class", 3: "3rd class"}
FARE_ORDER = ["Low (Q1)", "Mid-low (Q2)", "Mid-high (Q3)", "High (Q4)"]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable/derived columns used by EDA charts (display only, not for training)."""
    out = add_basic_features(df)
    out["AgeGroup"] = age_group(out["Age"])
    out["Port"] = out["Embarked"].map(PORT_NAMES).fillna("Unknown")
    out["Class"] = out["Pclass"].map(CLASS_NAMES)
    if out["Fare"].notna().sum() > 4:
        out["FareCategory"] = pd.qcut(out["Fare"], 4, labels=FARE_ORDER).astype(object)
    else:
        out["FareCategory"] = "Unknown"
    if TARGET in out.columns:
        out["Outcome"] = out[TARGET].map({1: "Survived", 0: "Did not survive"})
    return out


def overview(df: pd.DataFrame) -> dict:
    n = len(df)
    surv = int(df[TARGET].sum()) if TARGET in df.columns else None
    return {
        "passengers": n,
        "survivors": surv,
        "non_survivors": (n - surv) if surv is not None else None,
        "survival_rate": (surv / n) if surv is not None and n else None,
        "features": int(df.shape[1] - (1 if TARGET in df.columns else 0)),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(100 * df.isna().sum().sum() / df.size),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def survival_by(df: pd.DataFrame, col: str, order: list | None = None) -> pd.DataFrame:
    g = df.groupby(col, dropna=False, observed=True)[TARGET].agg(["count", "sum", "mean"]).reset_index()
    g.columns = [col, "passengers", "survivors", "survival_rate"]
    g["non_survivors"] = g["passengers"] - g["survivors"]
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
        g[col] = g[col].astype(str)
    return g.reset_index(drop=True)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    strategies = {
        "Age": "Median per title group (learned on training folds) + 'AgeKnown' indicator",
        "Cabin": "Not imputed — converted to 'CabinKnown' and 'Deck' (Unknown kept as a category)",
        "Embarked": "Most frequent port (learned on training folds)",
        "Fare": "Median (learned on training folds)",
    }
    m = df.isna().sum()
    t = pd.DataFrame({"column": m.index, "missing": m.values, "missing_pct": (100 * m.values / len(df)).round(2)})
    t["strategy"] = t["column"].map(strategies).fillna("—")
    return t[t["missing"] > 0].sort_values("missing", ascending=False).reset_index(drop=True)


def correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    e = add_basic_features(df)
    num = pd.DataFrame({
        "Survived": e.get(TARGET), "Pclass": e["Pclass"], "Sex (female=1)": (e["Sex"] == "female").astype(int),
        "Age": e["Age"], "SibSp": e["SibSp"], "Parch": e["Parch"], "Fare": e["Fare"],
        "FamilySize": e["FamilySize"], "IsAlone": e["IsAlone"], "CabinKnown": e["CabinKnown"],
    })
    num = num.dropna(axis=1, how="all")
    return num.corr(method=method)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    e = add_basic_features(df)
    rows = []
    for col in ("Age", "Fare", "FamilySize"):
        s = pd.to_numeric(e[col], errors="coerce").dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = np.abs(stats.zscore(s))
        rows.append({
            "feature": col, "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "max": s.max(),
            "iqr_low": lo, "iqr_high": hi, "iqr_outliers": int(((s < lo) | (s > hi)).sum()),
            "z_gt_3": int((z > 3).sum()), "skew": float(stats.skew(s)),
        })
    return pd.DataFrame(rows)


def extreme_passengers(df: pd.DataFrame, col: str, n: int = 8) -> pd.DataFrame:
    e = add_basic_features(df)
    cols = [c for c in ["PassengerId", "Name", "Pclass", "Sex", "Age", "FamilySize", "Fare", "Ticket", TARGET] if c in e.columns]
    return e.sort_values(col, ascending=False)[cols].head(n).reset_index(drop=True)


def chi_square(df: pd.DataFrame, col: str) -> tuple[float, float]:
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return float(chi2), float(p)


def auto_insights(df: pd.DataFrame) -> list[dict]:
    """Plain-language observations generated from the data (nothing hard-coded)."""
    if TARGET not in df.columns:
        return []
    e = enrich(df)
    out = []
    overall = e[TARGET].mean()
    out.append({"icon": "groups", "title": "Overall survival",
                "text": f"{e[TARGET].sum():,} of {len(e):,} passengers in this dataset survived ({overall:.1%})."})

    sex = survival_by(e, "Sex").set_index("Sex")["survival_rate"]
    if {"female", "male"} <= set(sex.index):
        chi2, p = chi_square(e, "Sex")
        out.append({"icon": "wc", "title": "Sex was strongly associated with survival",
                    "text": f"Female passengers survived at {sex['female']:.1%} vs {sex['male']:.1%} for males "
                            f"({sex['female'] / max(sex['male'], 1e-9):.1f}× higher; χ² p = {p:.1e})."})
    cls = survival_by(e, "Pclass").set_index("Pclass")["survival_rate"]
    if len(cls) == 3:
        chi2, p = chi_square(e, "Pclass")
        out.append({"icon": "stairs", "title": "Survival differed between passenger classes",
                    "text": f"1st class {cls[1]:.1%}, 2nd class {cls[2]:.1%}, 3rd class {cls[3]:.1%} (χ² p = {p:.1e})."})
    kids = e[e["Age"] < 13]
    adults = e[e["Age"] >= 18]
    if len(kids) and len(adults):
        out.append({"icon": "child_care", "title": "Age contains predictive information",
                    "text": f"Children under 13 survived at {kids[TARGET].mean():.1%} vs {adults[TARGET].mean():.1%} for adults "
                            f"(age known for {e['Age'].notna().mean():.0%} of passengers)."})
    fam = survival_by(e, "FamilyCategory").set_index("FamilyCategory")["survival_rate"]
    if {"Alone", "Small family", "Large family"} <= set(fam.index):
        out.append({"icon": "family_restroom", "title": "Family composition mattered",
                    "text": f"Small families (2–4) survived at {fam['Small family']:.1%}, people travelling alone at {fam['Alone']:.1%}, "
                            f"and large families (5+) at only {fam['Large family']:.1%}."})
    cab = survival_by(e, "CabinKnown").set_index("CabinKnown")["survival_rate"]
    if {0, 1} <= set(cab.index):
        out.append({"icon": "door_front", "title": "Missing data can itself be informative",
                    "text": f"Passengers with a recorded cabin survived at {cab[1]:.1%} vs {cab[0]:.1%} without — "
                            "cabin records were far more complete for 1st-class passengers."})
    port = survival_by(e, "Port").set_index("Port")["survival_rate"].drop("Unknown", errors="ignore")
    if len(port) >= 2:
        hi, lo = port.idxmax(), port.idxmin()
        out.append({"icon": "anchor", "title": "Embarkation port differed",
                    "text": f"Highest survival among passengers boarding at {hi} ({port[hi]:.1%}), lowest at {lo} ({port[lo]:.1%}). "
                            "This mostly reflects the class mix at each port — a reminder that correlation is not causation."})
    fare = survival_by(e, "FareCategory", order=FARE_ORDER).set_index("FareCategory")["survival_rate"]
    if len(fare) == 4:
        lo, hi = fare.loc[FARE_ORDER[0]], fare.loc[FARE_ORDER[-1]]
        trend = "rose" if hi > lo else "fell"
        out.append({"icon": "payments", "title": f"Survival {trend} with fare",
                    "text": f"Survival {trend} from {lo:.1%} in the cheapest fare quartile to {hi:.1%} in the most expensive "
                            "(fare is closely tied to passenger class)."})
    return out
