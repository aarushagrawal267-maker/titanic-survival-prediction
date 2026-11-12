"""The model zoo: estimator factories, search spaces, UI knobs and plain-language notes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE

try:  # XGBoost is optional
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


@dataclass
class ModelSpec:
    key: str
    name: str
    factory: Callable[..., Any]
    scale_sensitive: bool
    search_space: dict
    knobs: list[dict]            # manual hyper-parameter widgets for the Training page
    purpose: str
    pros: list[str]
    cons: list[str]
    tree_based: bool = False
    defaults: dict = field(default_factory=dict)

    def make(self, **params):
        p = {**self.defaults, **params}
        return self.factory(**p)


class CalibratedSVC(ClassifierMixin, BaseEstimator):
    """SVC with Platt-scaled probabilities.

    ``SVC(probability=True)`` is deprecated in recent scikit-learn; this wrapper
    keeps the simple ``C / kernel / gamma`` parameters while producing calibrated
    ``predict_proba`` via ``CalibratedClassifierCV(ensemble=False)``.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, random_state=RANDOM_STATE):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state

    def fit(self, X, y):
        svc = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, random_state=self.random_state)
        self.model_ = CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]


def _sklearn_version() -> tuple[int, int]:
    import sklearn

    parts = sklearn.__version__.split(".")
    return int(parts[0]), int("".join(ch for ch in parts[1] if ch.isdigit()) or 0)


# scikit-learn ≥ 1.8 replaced LogisticRegression(penalty=...) with l1_ratio (0 = L2, 1 = L1).
LR_USES_L1_RATIO = _sklearn_version() >= (1, 8)


def _lr(**p):
    p.setdefault("max_iter", 5000)
    penalty = p.pop("penalty", None)
    if penalty is not None:
        if LR_USES_L1_RATIO:
            p["l1_ratio"] = 1.0 if penalty == "l1" else 0.0
        else:
            p["penalty"] = penalty
    if p.get("solver") == "lbfgs" and (penalty == "l1" or p.get("l1_ratio") == 1.0):
        p["solver"] = "liblinear"  # lbfgs supports only L2
    return LogisticRegression(random_state=RANDOM_STATE, **p)


LR_PENALTY_SPACE = {"model__l1_ratio": [0.0, 1.0]} if LR_USES_L1_RATIO else {"model__penalty": ["l1", "l2"]}


def _xgb(**p):
    return XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1, tree_method="hist", **p)


SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression", name="Logistic Regression", factory=_lr, scale_sensitive=True,
        search_space={
            "model__C": loguniform(1e-3, 1e2),
            **LR_PENALTY_SPACE,
            "model__solver": ["liblinear", "saga"],
        },
        knobs=[
            {"param": "C", "type": "log_slider", "min": 0.001, "max": 100.0, "default": 1.0,
             "help": "Inverse regularisation strength — smaller C means a simpler model."},
            {"param": "penalty", "type": "select", "options": ["l2", "l1"], "default": "l2",
             "help": "L2 shrinks all weights; L1 can set weights exactly to zero."},
            {"param": "solver", "type": "select", "options": ["liblinear", "saga", "lbfgs"], "default": "liblinear",
             "help": "Optimisation algorithm (lbfgs supports only L2)."},
        ],
        purpose="Simple, interpretable baseline: a weighted sum of features squashed into a probability.",
        pros=["Fast and stable", "Coefficients are directly interpretable", "Well-calibrated probabilities"],
        cons=["Only learns linear relationships in log-odds", "Needs scaling and feature engineering for interactions"],
    ),
    "decision_tree": ModelSpec(
        key="decision_tree", name="Decision Tree", factory=lambda **p: DecisionTreeClassifier(random_state=RANDOM_STATE, **p),
        scale_sensitive=False, tree_based=True,
        search_space={
            "model__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "model__min_samples_split": randint(2, 40),
            "model__min_samples_leaf": randint(1, 20),
            "model__criterion": ["gini", "entropy"],
        },
        knobs=[
            {"param": "max_depth", "type": "int", "min": 1, "max": 20, "default": 5, "help": "Maximum number of questions on any path."},
            {"param": "min_samples_split", "type": "int", "min": 2, "max": 50, "default": 10, "help": "Minimum passengers needed to split a node."},
            {"param": "min_samples_leaf", "type": "int", "min": 1, "max": 30, "default": 5, "help": "Minimum passengers in a leaf."},
            {"param": "criterion", "type": "select", "options": ["gini", "entropy"], "default": "gini", "help": "Impurity measure used to choose splits."},
        ],
        purpose="Learns human-readable if/then rules by repeatedly splitting passengers into purer groups.",
        pros=["Very interpretable", "Captures non-linear rules and interactions", "No scaling needed"],
        cons=["High variance — overfits easily", "Blocky, step-shaped probabilities"],
    ),
    "random_forest": ModelSpec(
        key="random_forest", name="Random Forest",
        factory=lambda **p: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **p),
        scale_sensitive=False, tree_based=True,
        search_space={
            "model__n_estimators": randint(150, 600),
            "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", 0.5, None],
        },
        knobs=[
            {"param": "n_estimators", "type": "int", "min": 50, "max": 800, "default": 300, "step": 50, "help": "Number of trees in the forest."},
            {"param": "max_depth", "type": "int", "min": 2, "max": 20, "default": 7, "help": "Maximum depth of each tree."},
            {"param": "min_samples_split", "type": "int", "min": 2, "max": 30, "default": 4, "help": "Minimum passengers needed to split a node."},
            {"param": "min_samples_leaf", "type": "int", "min": 1, "max": 20, "default": 2, "help": "Minimum passengers in a leaf."},
            {"param": "max_features", "type": "select", "options": ["sqrt", "log2", "None"], "default": "sqrt", "help": "Features considered at each split."},
        ],
        purpose="Averages many decorrelated decision trees to reduce variance and generalise better.",
        pros=["Strong out-of-the-box accuracy", "Robust to outliers and scaling", "Built-in feature importance"],
        cons=["Less interpretable than a single tree", "Larger model, slower predictions"],
    ),
    "knn": ModelSpec(
        key="knn", name="K-Nearest Neighbors", factory=lambda **p: KNeighborsClassifier(**p), scale_sensitive=True,
        search_space={
            "model__n_neighbors": randint(3, 40),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        knobs=[
            {"param": "n_neighbors", "type": "int", "min": 1, "max": 60, "default": 15, "help": "How many similar passengers vote."},
            {"param": "weights", "type": "select", "options": ["uniform", "distance"], "default": "uniform", "help": "Closer neighbours can count more."},
            {"param": "p", "type": "select", "options": [2, 1], "default": 2, "help": "Distance: 2 = Euclidean, 1 = Manhattan."},
        ],
        purpose="Predicts from the outcomes of the most similar passengers in the training data.",
        pros=["Intuitive — 'people like you'", "No training phase", "Captures local patterns"],
        cons=["Very sensitive to feature scaling", "Struggles with many one-hot features", "Slow for large datasets"],
    ),
    "svm": ModelSpec(
        key="svm", name="Support Vector Machine",
        factory=lambda **p: CalibratedSVC(**p), scale_sensitive=True,
        search_space={
            "model__C": loguniform(1e-2, 1e2),
            "model__kernel": ["rbf", "linear", "poly"],
            "model__gamma": ["scale", "auto", 0.01, 0.05, 0.1],
        },
        knobs=[
            {"param": "C", "type": "log_slider", "min": 0.01, "max": 100.0, "default": 1.0, "help": "Penalty for misclassified passengers — higher C fits the training data more tightly."},
            {"param": "kernel", "type": "select", "options": ["rbf", "linear", "poly"], "default": "rbf", "help": "Shape of the decision boundary."},
            {"param": "gamma", "type": "select", "options": ["scale", "auto", 0.01, 0.05, 0.1], "default": "scale", "help": "Reach of a single passenger's influence (rbf/poly)."},
        ],
        purpose="Finds the boundary that separates survivors from non-survivors with the widest margin.",
        pros=["Effective with clear margins", "Kernels model non-linear boundaries"],
        cons=["Needs scaling", "Probabilities need an extra calibration step (Platt scaling)", "Harder to interpret"],
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting", name="Gradient Boosting",
        factory=lambda **p: GradientBoostingClassifier(random_state=RANDOM_STATE, **p),
        scale_sensitive=False, tree_based=True,
        search_space={
            "model__n_estimators": randint(80, 400),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__min_samples_leaf": randint(1, 15),
        },
        knobs=[
            {"param": "n_estimators", "type": "int", "min": 20, "max": 600, "default": 150, "step": 10, "help": "Number of boosting stages."},
            {"param": "learning_rate", "type": "log_slider", "min": 0.005, "max": 0.5, "default": 0.1, "help": "How much each new tree corrects the previous ones."},
            {"param": "max_depth", "type": "int", "min": 1, "max": 8, "default": 3, "help": "Depth of each small tree."},
            {"param": "subsample", "type": "float", "min": 0.5, "max": 1.0, "default": 1.0, "help": "Fraction of passengers sampled per tree."},
        ],
        purpose="Builds trees one after another, each fixing the errors of the ensemble so far.",
        pros=["Often top accuracy on tabular data", "Handles mixed feature types", "No scaling needed"],
        cons=["More hyper-parameters to tune", "Can overfit without regularisation", "Sequential training is slower"],
    ),
}

if XGBOOST_AVAILABLE:
    SPECS["xgboost"] = ModelSpec(
        key="xgboost", name="XGBoost", factory=_xgb, scale_sensitive=False, tree_based=True,
        search_space={
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": randint(2, 7),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_lambda": loguniform(0.1, 10),
            "model__min_child_weight": randint(1, 8),
        },
        knobs=[
            {"param": "n_estimators", "type": "int", "min": 50, "max": 800, "default": 300, "step": 50, "help": "Number of boosting rounds."},
            {"param": "learning_rate", "type": "log_slider", "min": 0.005, "max": 0.5, "default": 0.05, "help": "Shrinkage applied to each tree."},
            {"param": "max_depth", "type": "int", "min": 1, "max": 10, "default": 4, "help": "Depth of each tree."},
            {"param": "subsample", "type": "float", "min": 0.5, "max": 1.0, "default": 0.8, "help": "Row sampling per tree."},
            {"param": "colsample_bytree", "type": "float", "min": 0.3, "max": 1.0, "default": 0.8, "help": "Feature sampling per tree."},
        ],
        purpose="A fast, regularised gradient-boosting implementation widely used in competitions.",
        pros=["Excellent accuracy", "Built-in regularisation", "Efficient"],
        cons=["Many hyper-parameters", "Less interpretable", "Extra dependency"],
    )

MODEL_ORDER = [k for k in ["logistic_regression", "decision_tree", "random_forest", "knn", "svm", "gradient_boosting", "xgboost"] if k in SPECS]


def get_spec(key: str) -> ModelSpec:
    if key not in SPECS:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(SPECS)}")
    return SPECS[key]


def coerce_knob_value(knob: dict, value):
    if knob["param"] == "max_features" and value == "None":
        return None
    return value
