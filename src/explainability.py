"""Explainable AI: SHAP, permutation importance, native importances and a SHAP-free fallback.

Design choice
-------------
SHAP values are computed over the **model features produced by the feature
engineer** (17 human-meaningful columns such as *Sex*, *TitleGroup*, *Fare*),
not over the one-hot columns. Categorical features are integer-coded only for
SHAP's masker and decoded again before scoring, so one-hot groups are never
broken apart. We use ``shap.PermutationExplainer`` on the positive-class
probability for every model, so all explanations share one unit —
*percentage points of predicted survival probability* — and add up exactly to
``prediction − base value``.

These are statistical associations learned from historical data, not causes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_STATE
from src.preprocessing import friendly, original_feature, transformed_feature_names

try:
    import shap  # noqa: F401

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False

RAW_INPUT_LABELS = {
    "Pclass": "Passenger class", "Sex": "Sex", "Age": "Age", "SibSp": "Siblings/spouses",
    "Parch": "Parents/children", "Fare": "Fare", "Embarked": "Port of embarkation",
    "Name": "Name (→ title)", "Ticket": "Ticket (→ group size)", "Cabin": "Cabin (→ deck, cabin recorded)",
}


# ── helpers ────────────────────────────────────────────────────
class _Codec:
    """Integer-codes categorical engineered features so SHAP maskers can handle them."""

    def __init__(self, frames: list[pd.DataFrame]):
        full = pd.concat(frames, axis=0)
        self.columns = list(full.columns)
        self.categorical = [c for c in self.columns if full[c].dtype == object or str(full[c].dtype) in ("str", "string", "category")]
        self.levels = {}
        for c in self.categorical:
            vals = full[c].astype(object).where(full[c].notna(), "__nan__").unique().tolist()
            self.levels[c] = sorted(map(str, vals))

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.columns)), dtype=float)
        for j, c in enumerate(self.columns):
            if c in self.categorical:
                s = df[c].astype(object).where(df[c].notna(), "__nan__").astype(str)
                lut = {v: i for i, v in enumerate(self.levels[c])}
                out[:, j] = s.map(lambda v: lut.get(v, 0)).to_numpy(dtype=float)
            else:
                out[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        return out

    def decode(self, M: np.ndarray) -> pd.DataFrame:
        data = {}
        for j, c in enumerate(self.columns):
            col = M[:, j]
            if c in self.categorical:
                lv = self.levels[c]
                vals = [lv[int(round(v))] if 0 <= int(round(v)) < len(lv) else lv[0] for v in col]
                data[c] = pd.Series(vals, dtype=object).replace("__nan__", np.nan)
            else:
                data[c] = col
        return pd.DataFrame(data, columns=self.columns)


def _prob_fn(pipeline, codec: _Codec):
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    def f(M):
        return model.predict_proba(pre.transform(codec.decode(np.atleast_2d(M))))[:, 1]

    return f


def engineered(pipeline, X_raw: pd.DataFrame) -> pd.DataFrame:
    return pipeline.named_steps["features"].transform(X_raw).reset_index(drop=True)


# ── SHAP ───────────────────────────────────────────────────────
def shap_values(pipeline, X_explain_raw: pd.DataFrame, X_background_raw: pd.DataFrame,
                background_size: int = 80, max_evals: int | None = None) -> dict:
    """SHAP values (probability units) for each row of ``X_explain_raw``.

    Returns dict(values, base_value, features, data, method). Falls back to the
    occlusion method if SHAP is unavailable or fails.
    """
    E = engineered(pipeline, X_explain_raw)
    B = engineered(pipeline, X_background_raw)
    if len(B) > background_size:
        B = B.sample(background_size, random_state=RANDOM_STATE)
    features = list(E.columns)
    if SHAP_AVAILABLE:
        try:
            import shap

            codec = _Codec([E, B])
            f = _prob_fn(pipeline, codec)
            masker = shap.maskers.Independent(codec.encode(B), max_samples=len(B))
            p = len(features)
            explainer = shap.PermutationExplainer(f, masker, seed=RANDOM_STATE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp = explainer(codec.encode(E), max_evals=max_evals or max(2 * p + 1, 8 * p), silent=True)
            vals = np.asarray(exp.values, dtype=float)
            base = float(np.mean(np.atleast_1d(exp.base_values)))
            return {"values": vals, "base_value": base, "features": features, "data": E, "method": "SHAP (PermutationExplainer)"}
        except Exception as exc:  # pragma: no cover - depends on shap internals
            warnings.warn(f"SHAP failed ({exc}); using occlusion fallback.")
    vals, base = occlusion_values(pipeline, E, B)
    return {"values": vals, "base_value": base, "features": features, "data": E, "method": "Occlusion (SHAP fallback)"}


def occlusion_values(pipeline, E: pd.DataFrame, B: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP-free fallback: contribution = p(x) − p(x with feature set to a typical value),
    rescaled so contributions add up to p(x) − mean background prediction."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    predict = lambda df: model.predict_proba(pre.transform(df))[:, 1]  # noqa: E731
    base = float(predict(B).mean())
    typical = {}
    for c in E.columns:
        s = B[c]
        typical[c] = s.mode(dropna=True).iloc[0] if s.dtype == object else float(pd.to_numeric(s, errors="coerce").median())
    p_full = predict(E)
    vals = np.zeros(E.shape, dtype=float)
    for j, c in enumerate(E.columns):
        mod = E.copy()
        mod[c] = typical[c]
        vals[:, j] = p_full - predict(mod)
    # distribute the remainder in proportion to |contribution| so the values add up exactly
    residual = (p_full - base) - vals.sum(axis=1)
    weights = np.abs(vals)
    wsum = weights.sum(axis=1, keepdims=True)
    weights = np.where(wsum > 1e-12, weights / np.where(wsum > 1e-12, wsum, 1), 1.0 / vals.shape[1])
    return vals + residual[:, None] * weights, base


def shap_importance(shap_result: dict) -> pd.DataFrame:
    vals = np.abs(np.asarray(shap_result["values"]))
    df = pd.DataFrame({"feature": shap_result["features"], "mean_abs_shap": vals.mean(axis=0)})
    df["label"] = df["feature"].map(friendly)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_stored(shap_result: dict, passenger_id) -> pd.DataFrame:
    """Contributions for one passenger taken from the SHAP values stored at training time (same layout as
    ``explain_one``). Used where the SHAP library is unavailable (the browser demo), so no values are approximated."""
    i = list(shap_result["passenger_ids"]).index(passenger_id)
    features = list(shap_result["features"])
    data = pd.DataFrame(shap_result["data"])
    row = data.iloc[i]
    df = pd.DataFrame({
        "feature": features,
        "label": [friendly(f) for f in features],
        "value": [_fmt_value(f, row[f]) for f in features],
        "contribution": np.asarray(shap_result["values"])[i],
    })
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    df.attrs["base_value"] = float(shap_result["base_value"])
    df.attrs["method"] = f"{shap_result.get('method', 'SHAP')}, computed when the model was trained"
    return df


def explain_one(pipeline, x_raw: pd.DataFrame, X_background_raw: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """Per-feature contributions for a single passenger, sorted by absolute impact."""
    res = shap_values(pipeline, x_raw, X_background_raw, background_size=60)
    row = res["data"].iloc[0]
    df = pd.DataFrame({
        "feature": res["features"],
        "label": [friendly(f) for f in res["features"]],
        "value": [_fmt_value(f, row[f]) for f in res["features"]],
        "contribution": np.asarray(res["values"])[0],
    })
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    df.attrs["base_value"] = res["base_value"]
    df.attrs["method"] = res["method"]
    return df.head(top_k) if top_k else df


def _fmt_value(feature: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    if feature in ("IsAlone", "CabinKnown", "AgeKnown"):
        return "yes" if int(v) == 1 else "no"
    if feature == "Pclass":
        return f"{int(v)}"
    if feature in ("Fare", "FarePerPerson"):
        return f"£{float(v):.2f}"
    if feature == "Age":
        return f"{float(v):.0f}"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def narrative(contrib: pd.DataFrame, n: int = 3) -> dict[str, list[str]]:
    """Plain-language bullet points for the strongest positive and negative factors."""
    up = contrib[contrib["contribution"] > 0.005].head(n)
    down = contrib[contrib["contribution"] < -0.005].head(n)
    fmt = lambda r: f"**{r.label} = {r.value}** ({r.contribution * 100:+.1f} pts)"  # noqa: E731
    return {"higher": [fmt(r) for r in up.itertuples()], "lower": [fmt(r) for r in down.itertuples()]}


# ── Permutation & native importance ────────────────────────────
def permutation_importance_raw(pipeline, X_raw: pd.DataFrame, y, scoring: str = "roc_auc", n_repeats: int = 15) -> pd.DataFrame:
    """Drop in holdout score when each *raw input column* is shuffled."""
    cols = [c for c in RAW_INPUT_LABELS if c in X_raw.columns]
    Xs = X_raw[cols]
    r = permutation_importance(pipeline, Xs, y, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame({"feature": cols, "label": [RAW_INPUT_LABELS[c] for c in cols],
                       "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def native_importance(pipeline) -> pd.DataFrame | None:
    """Tree impurity importances or |coefficients|, aggregated to model features."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        kind = "Impurity-based importance"
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float).ravel())
        kind = "|Coefficient| (standardised features)"
    else:
        return None
    df = pd.DataFrame({"column": names, "value": raw})
    df["feature"] = df["column"].map(original_feature)
    agg = df.groupby("feature", as_index=False)["value"].sum().sort_values("value", ascending=False)
    agg["label"] = agg["feature"].map(friendly)
    agg.attrs["kind"] = kind
    return agg.reset_index(drop=True)


def coefficients(pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    names = transformed_feature_names(pipeline)
    df = pd.DataFrame({"column": names, "coefficient": np.asarray(model.coef_).ravel()})
    df["feature"] = df["column"].map(original_feature)
    df["level"] = [c.split("__", 1)[1] for c in df["column"]]
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
