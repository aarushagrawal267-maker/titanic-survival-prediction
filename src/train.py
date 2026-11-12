"""Reproducible training workflow.

    train.csv ──stratified 80/20 (seed)──► development set ──► 5-fold StratifiedKFold
                                       │                        RandomizedSearchCV
                                       └──► holdout set ──► touched once, for final metrics

* Every transformation lives inside the Pipeline → refit per CV fold (no leakage).
* Model selection uses cross-validation on the development set only.
* After evaluation the chosen configuration is refit on all labelled rows
  (``pipeline_full``) for predictions and the Kaggle submission.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

from src import config, explainability, model_manager
from src.config import ID_COL, RANDOM_STATE, TARGET
from src.evaluate import METRICS, compute_metrics, full_evaluation, threshold_table
from src.feature_engineering import ENGINEERED_FEATURES
from src.models import MODEL_ORDER, get_spec
from src.preprocessing import build_pipeline

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
ProgressFn = Callable[[float, str], None]


def _noop(frac: float, msg: str) -> None:  # pragma: no cover
    pass


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET].astype(int)


def split_data(df: pd.DataFrame):
    """Stratified development/holdout split with a fixed seed."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=config.get("split", "holdout_size", 0.2), stratify=y, random_state=RANDOM_STATE
    )


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.get("split", "cv_folds", 5), shuffle=True, random_state=RANDOM_STATE)


def default_scaler(key: str) -> str:
    return "standard" if get_spec(key).scale_sensitive else "none"


def load_experiments() -> dict:
    p = config.path("reports_dir") / "experiments.json"
    return json.loads(p.read_text()) if p.exists() else {}


def recommended_setup(key: str) -> dict:
    """Scaler / feature set chosen by the experiments (falls back to sensible defaults)."""
    exp = load_experiments()
    scaler = exp.get("scaling", {}).get("best", {}).get(key, default_scaler(key))
    fset = exp.get("feature_sets", {}).get("best", {}).get(key, config.get("features", "default_set", "engineered"))
    return {"scaler": scaler, "feature_set": fset}


def _cv_summary(fold_scores: dict[str, list[float]]) -> dict:
    out: dict = {"folds": {m: [float(v) for v in fold_scores[m]] for m in METRICS}}
    for m in METRICS:
        arr = np.asarray(fold_scores[m], dtype=float)
        out[f"{m}_mean"] = float(arr.mean())
        out[f"{m}_std"] = float(arr.std())
    return out


def cross_validate_manual(pipeline, X, y, progress: ProgressFn = _noop, start: float = 0.0, span: float = 1.0) -> dict:
    """Fold-by-fold CV (lets the UI show live progress)."""
    cv = get_cv()
    scores: dict[str, list[float]] = {m: [] for m in METRICS}
    fit_times = []
    n = cv.get_n_splits()
    for i, (tr, va) in enumerate(cv.split(X, y)):
        est = clone(pipeline)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y.iloc[tr])
        fit_times.append(time.perf_counter() - t0)
        prob = est.predict_proba(X.iloc[va])[:, 1]
        for m, v in compute_metrics(y.iloc[va], prob).items():
            scores[m].append(v)
        progress(start + span * (i + 1) / n, f"Fold {i + 1}/{n} — ROC-AUC {scores['roc_auc'][-1]:.3f}")
    res = _cv_summary(scores)
    res["fit_time_mean"] = float(np.mean(fit_times))
    return res


def train_model(
    df: pd.DataFrame,
    key: str,
    params: dict | None = None,
    tune: bool = True,
    n_iter: int | None = None,
    feature_set: str | None = None,
    scaler: str | None = None,
    pclass_encoding: str | None = None,
    age_strategy: str | None = None,
    compute_shap: bool = True,
    refit_full: bool = True,
    source: str = "app",
    progress: ProgressFn = _noop,
    save: bool = True,
) -> dict:
    """Train, cross-validate, evaluate on the holdout, explain, and save one model."""
    spec = get_spec(key)
    rec = recommended_setup(key)
    feature_set = feature_set or rec["feature_set"]
    scaler = scaler or rec["scaler"]
    pclass_encoding = pclass_encoding or config.get("features", "pclass_encoding", "onehot")
    age_strategy = age_strategy or config.get("features", "age_strategy", "title_group")
    n_iter = n_iter or config.get("training", "n_iter", 30)

    progress(0.02, "Splitting data (stratified 80/20, fixed seed)…")
    X_dev, X_hold, y_dev, y_hold = split_data(df)
    base_pipe = build_pipeline(spec.make(**(params or {})), feature_set=feature_set, scaler=scaler,
                               pclass_encoding=pclass_encoding, age_strategy=age_strategy)

    t_start = time.perf_counter()
    best_params = dict(params or {})
    if tune:
        progress(0.08, f"RandomizedSearchCV — {n_iter} candidates × {get_cv().get_n_splits()} folds…")
        search = RandomizedSearchCV(
            base_pipe, spec.search_space, n_iter=n_iter, scoring=SCORING, refit="roc_auc", cv=get_cv(),
            random_state=RANDOM_STATE, n_jobs=config.get("training", "n_jobs", -1), error_score=np.nan,
        )
        search.fit(X_dev, y_dev)
        bi = int(search.best_index_)
        k = get_cv().get_n_splits()
        folds = {m: [float(search.cv_results_[f"split{i}_test_{m}"][bi]) for i in range(k)] for m in METRICS}
        cv_res = _cv_summary(folds)
        cv_res["fit_time_mean"] = float(search.cv_results_["mean_fit_time"][bi])
        best_params = {p.replace("model__", ""): v for p, v in search.best_params_.items()}
        search_table = pd.DataFrame(search.cv_results_)[["mean_test_roc_auc", "std_test_roc_auc", "mean_test_accuracy", "mean_test_f1", "params"]]
        search_table = search_table.sort_values("mean_test_roc_auc", ascending=False).head(15)
        search_table["params"] = search_table["params"].map(lambda d: {k.replace("model__", ""): v for k, v in d.items()})
        dev_pipe = clone(base_pipe).set_params(**{f"model__{k}": v for k, v in best_params.items()})
        progress(0.55, f"Best CV ROC-AUC {cv_res['roc_auc_mean']:.3f} — refitting on the development set…")
    else:
        cv_res = cross_validate_manual(base_pipe, X_dev, y_dev, progress, start=0.05, span=0.5)
        search_table = None
        dev_pipe = clone(base_pipe)

    t0 = time.perf_counter()
    dev_pipe.fit(X_dev, y_dev)
    fit_time = time.perf_counter() - t0
    tuning_time = time.perf_counter() - t_start

    progress(0.62, "Evaluating on the untouched holdout set…")
    hold_prob = dev_pipe.predict_proba(X_hold)[:, 1]
    evaluation = full_evaluation(y_hold, hold_prob)
    thr_table = threshold_table(y_hold, hold_prob)

    progress(0.70, "Permutation importance on the holdout set…")
    perm = explainability.permutation_importance_raw(dev_pipe, X_hold, y_hold, n_repeats=10)
    native = explainability.native_importance(dev_pipe)

    shap_res = None
    if compute_shap:
        progress(0.78, "Computing SHAP explanations…")
        n = config.get("training", "shap_sample_size", 120)
        X_explain = X_hold.sample(min(n, len(X_hold)), random_state=RANDOM_STATE)
        shap_res = explainability.shap_values(
            dev_pipe, X_explain, X_dev, background_size=config.get("training", "shap_background_size", 80)
        )
        shap_res["passenger_ids"] = X_explain[ID_COL].tolist() if ID_COL in X_explain else list(range(len(X_explain)))
        shap_res["y_true"] = y_hold.loc[X_explain.index].tolist()
        shap_res["data"] = shap_res["data"].to_dict("list")

    full_pipe = None
    if refit_full:
        progress(0.93, "Refitting the final model on all 891 labelled passengers…")
        X_all, y_all = split_xy(df)
        full_pipe = clone(dev_pipe).fit(X_all, y_all)

    holdout_metrics = evaluation["metrics"]
    meta = {
        "key": key, "name": spec.name, "feature_set": feature_set, "scaler": scaler,
        "pclass_encoding": pclass_encoding, "age_strategy": age_strategy,
        "tuned": bool(tune), "n_iter": n_iter if tune else None, "best_params": best_params,
        "cv": {k: v for k, v in cv_res.items()}, "holdout": holdout_metrics,
        "train_time_s": float(fit_time), "total_time_s": float(tuning_time),
        "n_dev": int(len(X_dev)), "n_holdout": int(len(X_hold)), "source": source,
    }
    eval_blob = {
        **evaluation,
        "threshold_table": thr_table.to_dict("list"),
        "holdout_ids": X_hold[ID_COL].tolist() if ID_COL in X_hold else None,
        "holdout_y": y_hold.tolist(),
        "holdout_prob": hold_prob.tolist(),
        "permutation_importance": perm.to_dict("list"),
        "native_importance": native.to_dict("list") if native is not None else None,
        "native_kind": native.attrs.get("kind") if native is not None else None,
        "shap": shap_res,
        "search_table": search_table.to_dict("list") if search_table is not None else None,
    }
    if save:
        progress(0.97, "Saving model to the registry…")
        model_manager.save_model(key, dev_pipe, full_pipe, meta, eval_blob)
    progress(1.0, "Done.")
    return {"meta": meta, "evaluation": eval_blob, "pipeline": dev_pipe, "pipeline_full": full_pipe}


# ── Experiments (scaling, feature sets, ablation, encodings) ─────────────
def _cv_auc(pipe, X, y) -> tuple[float, float, float]:
    r = cross_validate(pipe, X, y, cv=get_cv(), scoring={"roc_auc": "roc_auc", "accuracy": "accuracy"}, n_jobs=-1)
    return float(r["test_roc_auc"].mean()), float(r["test_roc_auc"].std()), float(r["test_accuracy"].mean())


def scaling_experiment(X_dev, y_dev, keys=None) -> dict:
    keys = keys or [k for k in MODEL_ORDER if get_spec(k).scale_sensitive]
    rows, best = [], {}
    for k in keys:
        scores = {}
        for sc in ("none", "standard", "robust"):
            auc, std, acc = _cv_auc(build_pipeline(get_spec(k).make(), scaler=sc), X_dev, y_dev)
            scores[sc] = auc
            rows.append({"model": get_spec(k).name, "key": k, "scaler": sc, "roc_auc": auc, "roc_auc_std": std, "accuracy": acc})
        best[k] = max(scores, key=scores.get)
    return {"rows": rows, "best": best}


def feature_set_experiment(X_dev, y_dev, scalers: dict) -> dict:
    rows, best = [], {}
    for k in MODEL_ORDER:
        res = {}
        for fs in ("base", "engineered"):
            auc, std, acc = _cv_auc(build_pipeline(get_spec(k).make(), feature_set=fs, scaler=scalers.get(k, default_scaler(k))), X_dev, y_dev)
            res[fs] = auc
            rows.append({"model": get_spec(k).name, "key": k, "feature_set": fs, "roc_auc": auc, "roc_auc_std": std, "accuracy": acc})
        best[k] = "engineered" if res["engineered"] >= res["base"] else "base"
    return {"rows": rows, "best": best}


def ablation_study(X_dev, y_dev, keys=("logistic_regression", "random_forest")) -> dict:
    """Drop one engineered feature at a time; report the change in CV ROC-AUC."""
    rows = []
    for k in keys:
        sc = default_scaler(k)
        full_auc, full_std, _ = _cv_auc(build_pipeline(get_spec(k).make(), scaler=sc), X_dev, y_dev)
        rows.append({"model": get_spec(k).name, "key": k, "dropped": "(none — all features)", "roc_auc": full_auc, "delta": 0.0})
        for feat in ENGINEERED_FEATURES:
            auc, _, _ = _cv_auc(build_pipeline(get_spec(k).make(), scaler=sc, drop_features=(feat,)), X_dev, y_dev)
            rows.append({"model": get_spec(k).name, "key": k, "dropped": feat, "roc_auc": auc, "delta": auc - full_auc})
    return {"rows": rows}


def encoding_experiment(X_dev, y_dev, keys=("logistic_regression", "random_forest")) -> dict:
    rows = []
    for k in keys:
        sc = default_scaler(k)
        for enc in ("onehot", "ordinal"):
            for age in ("title_group", "median"):
                auc, std, acc = _cv_auc(build_pipeline(get_spec(k).make(), scaler=sc, pclass_encoding=enc, age_strategy=age), X_dev, y_dev)
                rows.append({"model": get_spec(k).name, "key": k, "pclass_encoding": enc, "age_strategy": age, "roc_auc": auc, "roc_auc_std": std, "accuracy": acc})
    return {"rows": rows}


def run_experiments(df: pd.DataFrame, log: Callable[[str], None] = print) -> dict:
    X_dev, _, y_dev, _ = split_data(df)
    log("Experiment 1/4 — scaling (none vs StandardScaler vs RobustScaler)…")
    scaling = scaling_experiment(X_dev, y_dev)
    log("Experiment 2/4 — base vs engineered feature sets…")
    fsets = feature_set_experiment(X_dev, y_dev, scaling["best"])
    log("Experiment 3/4 — drop-one-feature ablation…")
    ablation = ablation_study(X_dev, y_dev)
    log("Experiment 4/4 — Pclass encoding & age imputation strategy…")
    enc = encoding_experiment(X_dev, y_dev)
    exp = {"scaling": scaling, "feature_sets": fsets, "ablation": ablation, "encoding": enc,
           "n_dev": int(len(X_dev)), "cv_folds": get_cv().get_n_splits(), "random_state": RANDOM_STATE}
    out = config.path("reports_dir") / "experiments.json"
    out.write_text(json.dumps(model_manager._jsonable(exp), indent=2))
    return exp
