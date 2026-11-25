"""Training, predictions, probabilities, explanations, batch scoring and Kaggle submission."""
import json

import numpy as np
import pandas as pd
import pytest

from src import explainability as xai
from src import model_manager
from src.batch_prediction import drift_report, read_csv, score_batch, template_csv
from src.evaluate import baseline_predictions, bootstrap_ci, compute_metrics, full_evaluation, threshold_table
from src.prediction import SAMPLE_PROFILES, PassengerInput, SimilarPassengerIndex, predict_one, what_if_curve
from src.submission import make_submission, validate_submission
from src.train import train_model


# ── training ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def quick_result(train_df):
    return train_model(train_df, "decision_tree", params={"max_depth": 4}, tune=False, compute_shap=False, save=False)


def test_training_produces_sane_metrics(quick_result):
    meta = quick_result["meta"]
    assert meta["n_dev"] == 712 and meta["n_holdout"] == 179
    assert len(meta["cv"]["folds"]["roc_auc"]) == 5
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.5 < meta["holdout"][m] <= 1.0
        assert 0.5 < meta["cv"][f"{m}_mean"] <= 1.0
    assert meta["holdout"]["accuracy"] > 0.70, "a depth-4 tree should beat the 62% majority baseline"


def test_tuned_training_with_randomized_search(train_df):
    res = train_model(train_df, "logistic_regression", tune=True, n_iter=3, compute_shap=False, refit_full=False, save=False)
    assert res["meta"]["tuned"] and "C" in res["meta"]["best_params"]
    assert res["evaluation"]["search_table"] is not None


def test_all_models_can_fit_and_predict(train_df):
    from src.models import MODEL_ORDER, get_spec
    from src.preprocessing import build_pipeline

    X, y = train_df.drop(columns=["Survived"]), train_df["Survived"]
    for key in MODEL_ORDER:
        spec = get_spec(key)
        pipe = build_pipeline(spec.make(), scaler="standard" if spec.scale_sensitive else "none").fit(X.iloc[:600], y.iloc[:600])
        p = pipe.predict_proba(X.iloc[600:])
        assert p.shape == (291, 2) and np.allclose(p.sum(1), 1) and ((p >= 0) & (p <= 1)).all()
        assert set(np.unique(pipe.predict(X.iloc[600:]))) <= {0, 1}


def test_full_refit_uses_all_rows(quick_result, train_df):
    full = quick_result["pipeline_full"]
    assert sum(full.named_steps["features"].ticket_counts_.values()) == train_df["Ticket"].notna().sum()


def test_registry_save_load_and_selection_rule(quick_result, tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager, "models_dir", lambda: tmp_path)
    meta = dict(quick_result["meta"])
    model_manager.save_model("decision_tree", quick_result["pipeline"], quick_result["pipeline_full"], meta, quick_result["evaluation"])
    meta2 = {**meta, "key": "fake_better", "cv": {**meta["cv"], "roc_auc_mean": 0.99}, "holdout": {**meta["holdout"], "accuracy": 0.10}}
    model_manager.save_model("fake_better", quick_result["pipeline"], None, meta2, quick_result["evaluation"])
    reg = model_manager.load_registry()
    assert set(reg) == {"decision_tree", "fake_better"}
    # selection uses CV ROC-AUC, not holdout accuracy
    assert model_manager.best_model_key() == "fake_better"
    pipe = model_manager.load_pipeline("fake_better", "full")  # falls back to dev pipeline when no full model
    assert hasattr(pipe, "predict_proba")
    assert len(model_manager.load_run_history()) == 2
    json.loads((tmp_path / "registry.json").read_text())


# ── evaluation ──────────────────────────────────────────────────────
def test_metrics_and_curves():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(5 / 6) and m["precision"] == 1.0 and m["recall"] == pytest.approx(2 / 3)
    ev = full_evaluation(y, p)
    assert np.array(ev["confusion_matrix"]).sum() == 6
    assert ev["roc"]["fpr"][0] == 0 and ev["roc"]["tpr"][-1] == 1
    tt = threshold_table(y, p)
    assert (tt["tp"] + tt["fn"] == 3).all()


def test_bootstrap_ci_brackets_point_estimate_and_matches_metric_definitions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.25, 200), 0, 1)
    point = compute_metrics(y, p)
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lo, hi = bootstrap_ci(y, p, metric, n_boot=300)
        assert 0 <= lo <= point[metric] <= hi <= 1, metric
    # with a single resample the interval collapses to that resample's score: the fast scorer must match sklearn
    idx = np.random.default_rng(7).integers(0, len(y), len(y))
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        lo, hi = bootstrap_ci(y, p, metric, n_boot=1, seed=7)
        assert lo == pytest.approx(hi) == pytest.approx(compute_metrics(y[idx], p[idx])[metric]), metric
    with pytest.raises(ValueError):
        bootstrap_ci(y, p, "log_loss")


def test_baselines_use_no_labels_and_match_known_holdout_values(train_df):
    from src.train import split_data

    X_dev, X_hold, y_dev, y_hold = split_data(train_df)
    base = baseline_predictions(X_dev, y_dev, X_hold)
    majority, sex_rule = base.values()
    assert np.allclose(majority, y_dev.mean())  # constant → everyone predicted "did not survive"
    assert compute_metrics(y_hold, majority)["accuracy"] == pytest.approx(1 - y_hold.mean())
    assert set(np.unique(sex_rule)) <= {0.0, 1.0}
    assert (sex_rule == (X_hold["Sex"] == "female").to_numpy()).all()
    # documented in README/PROJECT_REPORT: the sex rule scores 0.777 accuracy on the 179-passenger holdout
    assert compute_metrics(y_hold, sex_rule)["accuracy"] == pytest.approx(139 / 179)


# ── single predictions ──────────────────────────────────────────────
def test_predict_one_returns_valid_probabilities(fitted_pipeline):
    for name, inp in SAMPLE_PROFILES.items():
        r = predict_one(fitted_pipeline, inp)
        assert r["ok"], name
        assert 0 <= r["prob_survived"] <= 1 and abs(r["prob_survived"] + r["prob_not_survived"] - 1) < 1e-12
        assert r["prediction"] in (0, 1) and r["prediction"] == int(r["prob_survived"] >= 0.5)


def test_prediction_patterns_match_data(fitted_pipeline):
    rich_woman = predict_one(fitted_pipeline, PassengerInput(1, "female", 30, 0, 0, 80.0, "C"))["prob_survived"]
    poor_man = predict_one(fitted_pipeline, PassengerInput(3, "male", 30, 0, 0, 7.25, "S"))["prob_survived"]
    assert rich_woman > 0.8 and poor_man < 0.2


def test_input_validation(fitted_pipeline):
    r = predict_one(fitted_pipeline, PassengerInput(4, "x", -3, -1, 0, -5.0, "Z"))
    assert not r["ok"] and len(r["errors"]) >= 4
    warn = predict_one(fitted_pipeline, PassengerInput(3, "male", 95, 0, 0, 700.0, "S"))
    assert warn["ok"] and warn["warnings"]


def test_unknown_age_is_allowed(fitted_pipeline):
    assert predict_one(fitted_pipeline, PassengerInput(2, "female", None, 0, 1, 20.0, "S"))["ok"]


def test_infant_age_zero_is_allowed(fitted_pipeline):
    # The form's age slider starts at 0 and real passengers under one year (e.g. #804, 0.42) round to it.
    r = predict_one(fitted_pipeline, PassengerInput(3, "male", 0, 0, 1, 8.5167, "C"))
    assert r["ok"], r["errors"]
    assert 0 <= r["prob_survived"] <= 1


def test_non_finite_inputs_are_rejected_not_crashing(fitted_pipeline):
    for inp in (PassengerInput(1, "male", float("nan"), 0, 0, 30.0, "S"),
                PassengerInput(1, "male", 30, 0, 0, float("inf"), "S"),
                PassengerInput(1, "male", 30, 0, 0, float("nan"), "S")):
        r = predict_one(fitted_pipeline, inp)
        assert not r["ok"] and len(r["errors"]) == 1


def test_what_if_and_similar_passengers(fitted_pipeline, train_df):
    inp = SAMPLE_PROFILES["Fictional · Second-class boy"]
    curve = what_if_curve(fitted_pipeline, inp, "Age", list(range(1, 60, 5)))
    assert curve["prob_survived"].between(0, 1).all() and len(curve) == 12
    sim = SimilarPassengerIndex(train_df, k=5).query(inp.to_frame())
    assert len(sim) == 5 and set(sim["PassengerId"]) <= set(train_df["PassengerId"])


def test_explanations_add_up(fitted_tree_pipeline, train_df):
    X = train_df.drop(columns=["Survived"])
    res = xai.shap_values(fitted_tree_pipeline, X.iloc[:5], X.iloc[100:300], background_size=40)
    pred = fitted_tree_pipeline.predict_proba(X.iloc[:5])[:, 1]
    assert np.allclose(np.asarray(res["values"]).sum(1) + res["base_value"], pred, atol=1e-6)
    c = xai.explain_one(fitted_tree_pipeline, X.iloc[[0]], X.iloc[100:300])
    assert {"higher", "lower"} == set(xai.narrative(c))


def test_occlusion_fallback_adds_up(fitted_pipeline, train_df):
    X = train_df.drop(columns=["Survived"])
    E = xai.engineered(fitted_pipeline, X.iloc[:4])
    B = xai.engineered(fitted_pipeline, X.iloc[200:300])
    vals, base = xai.occlusion_values(fitted_pipeline, E, B)
    pred = fitted_pipeline.predict_proba(X.iloc[:4])[:, 1]
    assert np.allclose(vals.sum(1) + base, pred, atol=1e-6)


def test_permutation_and_native_importance(fitted_pipeline, train_df):
    X, y = train_df.drop(columns=["Survived"]), train_df["Survived"]
    pi = xai.permutation_importance_raw(fitted_pipeline, X.iloc[:200], y.iloc[:200], n_repeats=2)
    assert "PassengerId" not in set(pi["feature"]) and len(pi) == 10
    assert xai.native_importance(fitted_pipeline) is not None


# ── batch ───────────────────────────────────────────────────────────
def test_batch_scoring_on_kaggle_test(fitted_pipeline, test_df):
    out, rep = score_batch(fitted_pipeline, test_df)
    assert rep["valid"] and len(out) == 418
    assert out["Survival probability"].between(0, 1).all()
    assert set(out["Predicted"].unique()) <= {0, 1}
    assert np.allclose(out["Survival probability"] + out["Non-survival probability"], 1, atol=1e-3)


def test_batch_template_and_minimal_columns(fitted_pipeline):
    out, rep = score_batch(fitted_pipeline, read_csv(template_csv().encode()))
    assert rep["valid"] and len(out) == 3
    minimal = pd.DataFrame({"Pclass": [1, 3], "Sex": ["female", "male"], "Age": [np.nan, 40], "SibSp": [0, 1], "Parch": [0, 0],
                            "Fare": [np.nan, 8.0], "Embarked": [np.nan, "Q"]})
    out, rep = score_batch(fitted_pipeline, minimal)
    assert rep["valid"] and len(out) == 2


def test_batch_rejects_invalid_file(fitted_pipeline):
    out, rep = score_batch(fitted_pipeline, pd.DataFrame({"foo": [1], "Sex": ["robot"]}))
    assert out is None and not rep["valid"]


def test_drift_report(train_df, test_df):
    dr = drift_report(train_df, test_df)
    assert set(dr["feature"]) == {"Pclass", "Sex", "Embarked", "Age", "Fare", "SibSp", "Parch"}
    assert (dr["psi"] < 0.25).all(), "Kaggle test should look like train"
    shifted = test_df.assign(Pclass=1, Fare=test_df["Fare"] * 10)
    assert (drift_report(train_df, shifted).set_index("feature").loc[["Pclass", "Fare"], "psi"] > 0.25).all()


# ── Kaggle submission ───────────────────────────────────────────────
def test_submission_format(fitted_pipeline, test_df):
    sub = make_submission(fitted_pipeline, test_df)
    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert len(sub) == 418 and sub["PassengerId"].tolist() == test_df["PassengerId"].tolist()
    assert sub["Survived"].isin([0, 1]).all() and pd.api.types.is_integer_dtype(sub["Survived"])
    assert validate_submission(sub, test_df) == []


def test_submission_validation_catches_problems(fitted_pipeline, test_df):
    sub = make_submission(fitted_pipeline, test_df)
    assert validate_submission(sub.head(400), test_df)
    assert validate_submission(sub.assign(extra=1), test_df)
    assert validate_submission(sub.assign(Survived=0.7), test_df)
    assert validate_submission(sub.iloc[::-1].reset_index(drop=True), test_df)
