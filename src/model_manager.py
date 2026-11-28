"""Model registry on disk.

Layout::

    models/
      registry.json            # summary of every trained model (for fast page loads)
      run_history.jsonl        # append-only log of every training run (monitoring)
      <model_key>/
        pipeline.joblib        # fitted on the development set — used for evaluation
        pipeline_full.joblib   # refit on all labelled rows — used for predictions/Kaggle
        evaluation.joblib      # holdout predictions, curves, CV details, explanations
        metadata.json          # params, metrics, timings, versions
"""
from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src import config

if config.BROWSER_DEMO:
    # The browser runtime has an older scikit-learn; the shipped models were checked to give the same predictions
    # there (browser_demo/verified_models.json), so the version-mismatch warning would only be noise.
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def models_dir() -> Path:
    return config.path("models_dir")


def _model_dir(key: str) -> Path:
    d = models_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def library_versions() -> dict[str, str]:
    import pandas
    import sklearn

    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                "pandas": pandas.__version__, "numpy": np.__version__}
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except Exception:
        pass
    return versions


def save_model(key: str, pipeline_dev, pipeline_full, metadata: dict, evaluation: dict) -> None:
    d = _model_dir(key)
    joblib.dump(pipeline_dev, d / "pipeline.joblib", compress=3)
    if pipeline_full is not None:
        joblib.dump(pipeline_full, d / "pipeline_full.joblib", compress=3)
    joblib.dump(evaluation, d / "evaluation.joblib", compress=3)
    meta = {**metadata, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": library_versions()}
    (d / "metadata.json").write_text(json.dumps(_jsonable(meta), indent=2))
    _update_registry(key, meta)
    log_run(meta)


def _update_registry(key: str, meta: dict) -> None:
    reg = load_registry()
    reg[key] = _jsonable({
        k: meta.get(k) for k in (
            "key", "name", "feature_set", "scaler", "tuned", "best_params", "cv", "holdout",
            "train_time_s", "saved_at", "n_dev", "n_holdout", "source",
        )
    })
    (models_dir() / "registry.json").write_text(json.dumps(reg, indent=2))


def log_run(meta: dict) -> None:
    rec = {
        "time": meta.get("saved_at"), "key": meta.get("key"), "name": meta.get("name"),
        "source": meta.get("source", "app"), "tuned": meta.get("tuned"),
        "cv_roc_auc": (meta.get("cv") or {}).get("roc_auc_mean"),
        "holdout": meta.get("holdout"), "train_time_s": meta.get("train_time_s"),
        "feature_set": meta.get("feature_set"), "scaler": meta.get("scaler"),
    }
    with open(models_dir() / "run_history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(rec)) + "\n")


def load_registry() -> dict[str, dict]:
    p = models_dir() / "registry.json"
    if not p.exists():
        return {}
    try:
        reg = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    # keep only entries whose files still exist
    return {k: v for k, v in reg.items() if (models_dir() / k / "pipeline.joblib").exists()}


def load_run_history() -> list[dict]:
    p = models_dir() / "run_history.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_metadata(key: str) -> dict:
    return json.loads((models_dir() / key / "metadata.json").read_text())


def load_pipeline(key: str, which: str = "full"):
    """``which='full'`` → refit on all labelled data (predictions); ``'dev'`` → evaluation model."""
    d = models_dir() / key
    fname = "pipeline_full.joblib" if which == "full" else "pipeline.joblib"
    p = d / fname
    if which == "full" and not p.exists():
        p = d / "pipeline.joblib"
    return joblib.load(p)


def load_evaluation(key: str) -> dict:
    return joblib.load(models_dir() / key / "evaluation.joblib")


def has_models() -> bool:
    return bool(load_registry())


def best_model_key(metric: str | None = None) -> str | None:
    """Selection rule: highest mean CV ROC-AUC; ties broken by CV F1, then lower CV std.

    Uses cross-validation on the development set only — never the holdout.
    """
    reg = load_registry()
    if not reg:
        return None
    metric = metric or config.get("training", "primary_metric", "roc_auc")

    def score(item):
        cv = item[1].get("cv") or {}
        return (round(cv.get(f"{metric}_mean") or 0, 4), round(cv.get("f1_mean") or 0, 4), -(cv.get(f"{metric}_std") or 0))

    return max(reg.items(), key=score)[0]


def browser_verified_models() -> dict[str, float]:
    """Browser demo only: {model key: max |probability difference|} from ``models/browser_verified.json``."""
    if not config.BROWSER_DEMO:
        return {}
    try:
        return json.loads((models_dir() / "browser_verified.json").read_text())["max_abs_diff"]
    except Exception:
        return {}


def browser_oof() -> dict[str, list[float]]:
    """Browser demo only: out-of-fold survival probabilities per model for the rows of train.csv, computed by
    browser_demo/build.py with the pinned desktop libraries (refitting is not possible in the browser runtime)."""
    try:
        return json.loads((models_dir() / "browser_oof.json").read_text())
    except Exception:
        return {}


def stale_models() -> list[tuple[str, str]]:
    """Models that should be retrained in this environment: (key, reason).

    Pickled scikit-learn/XGBoost models are only guaranteed to work with the library
    versions that created them, so a mismatch (e.g. after deploying to a server with
    newer packages) or any load error marks the model as stale.
    """
    import warnings

    current = library_versions()
    verified = browser_verified_models()
    out = []
    for key in load_registry():
        if key in verified:
            # Browser demo: library versions differ by design, but this model's predictions were checked
            # against the committed model under the browser's runtime (see browser_demo/build.py).
            if not (models_dir() / key / "evaluation.joblib").exists():
                out.append((key, "missing evaluation.joblib"))
            continue
        try:
            saved = load_metadata(key).get("versions", {})
        except Exception as exc:
            out.append((key, f"unreadable metadata ({exc})"))
            continue
        libs = ["scikit-learn"] + (["xgboost"] if key == "xgboost" else [])
        diff = [f"{lib} {saved.get(lib)} → {current.get(lib)}" for lib in libs if saved.get(lib) != current.get(lib)]
        if diff:
            out.append((key, "trained with " + ", ".join(diff)))
            continue
        # The evaluation, comparison and explainability pages read evaluation.joblib, so a
        # model directory without it (e.g. a partial copy) must be rebuilt, not shown half-broken.
        if not (models_dir() / key / "evaluation.joblib").exists():
            out.append((key, "missing evaluation.joblib"))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # InconsistentVersionWarning → treat as stale
                load_pipeline(key, "full")
                load_pipeline(key, "dev")
        except Exception as exc:
            out.append((key, f"failed to load ({type(exc).__name__})"))
    return out


def delete_model(key: str) -> None:
    d = models_dir() / key
    if d.exists():
        shutil.rmtree(d)
    reg = load_registry()
    reg.pop(key, None)
    (models_dir() / "registry.json").write_text(json.dumps(reg, indent=2))
