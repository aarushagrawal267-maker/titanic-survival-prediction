"""Model evaluation: metrics, curves, calibration, threshold analysis, learning curves."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.config import RANDOM_STATE

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "roc_auc": "ROC-AUC",
}
METRIC_EXPLANATIONS = {
    "accuracy": "Overall share of passengers the model classified correctly.",
    "precision": "Of the passengers predicted to survive, how many actually survived?",
    "recall": "Of the passengers who actually survived, how many did the model identify?",
    "f1": "The harmonic mean of precision and recall — a single balance of the two.",
    "roc_auc": "How well the model ranks survivors above non-survivors across every possible threshold (0.5 = coin flip, 1.0 = perfect).",
}


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def full_evaluation(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Everything the evaluation page needs, as plain Python/numpy objects (joblib-friendly)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)
    n_bins = 8
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "metrics": compute_metrics(y_true, y_prob, threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["Did not survive", "Survived"], output_dict=True, zero_division=0
        ),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": np.nan_to_num(roc_thr, posinf=1.0).tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist(), "thresholds": pr_thr.tolist(),
               "average_precision": float(average_precision_score(y_true, y_prob))},
        "calibration": {"fraction_positive": frac_pos.tolist(), "mean_predicted": mean_pred.tolist()},
        "base_rate": float(y_true.mean()),
    }


def bootstrap_ci(y_true, y_prob, metric: str, n_boot: int = 2000, level: float = 0.95,
                 threshold: float = 0.5, seed: int = RANDOM_STATE) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for one holdout metric.

    Passengers are resampled with replacement; resamples that contain a single class
    (ROC-AUC undefined) are skipped. With ~180 passengers these intervals are wide —
    that width is the honest uncertainty of a single holdout estimate.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("bootstrap_ci needs non-empty, aligned labels and probabilities")
    y_pred = (y_prob >= threshold).astype(int)

    def score(yt, yp, pr) -> float:  # one metric only — computing all five per resample is ~10x slower
        if metric == "roc_auc":
            return float(roc_auc_score(yt, pr))
        tp = int(((yp == 1) & (yt == 1)).sum())
        if metric == "accuracy":
            return float((yp == yt).mean())
        if metric == "precision":
            return tp / max(int(yp.sum()), 1)
        if metric == "recall":
            return tp / max(int(yt.sum()), 1)
        denom = int(yp.sum()) + int(yt.sum())
        return 2 * tp / denom if denom else 0.0

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(score(y_true[idx], y_pred[idx], y_prob[idx]))
    if not vals:
        raise ValueError("bootstrap_ci needs labels from both classes")
    alpha = (1 - level) / 2
    return float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))


def baseline_predictions(X_dev, y_dev, X_eval) -> dict[str, np.ndarray]:
    """Trivial reference "models" that every trained model should be compared against.

    * Majority class — predict the development-set survival rate for everyone
      (so every passenger is classified "did not survive"; ROC-AUC is 0.5 by definition).
    * Sex only — the one-line rule "women survived, men did not" (probability 1 / 0).
    Neither looks at the evaluation labels.
    """
    rate = float(np.asarray(y_dev, dtype=float).mean())
    return {
        "Majority class (everyone did not survive)": np.full(len(X_eval), rate),
        "Sex only (female → survived)": (X_eval["Sex"].astype(str).str.lower() == "female").to_numpy(dtype=float),
    }


def threshold_table(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, t)
        cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= t).astype(int), labels=[0, 1])
        rows.append({"threshold": float(t), **m, "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    return pd.DataFrame(rows)


def compute_learning_curve(pipeline, X, y, cv_folds: int = 5, scoring: str = "roc_auc") -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.15, 1.0, 7), n_jobs=-1, shuffle=True, random_state=RANDOM_STATE
    )
    return pd.DataFrame({
        "train_size": sizes,
        "train_mean": train_scores.mean(1), "train_std": train_scores.std(1),
        "val_mean": val_scores.mean(1), "val_std": val_scores.std(1),
    })
