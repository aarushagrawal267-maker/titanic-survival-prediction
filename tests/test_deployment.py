"""Deployment helpers: stale-model detection and public demo mode."""
import json

from src import model_manager


def test_committed_models_are_not_stale_in_this_environment():
    if not model_manager.has_models():
        return
    assert model_manager.stale_models() == []


def test_version_mismatch_marks_model_stale(tmp_path, monkeypatch, fitted_pipeline):
    monkeypatch.setattr(model_manager, "models_dir", lambda: tmp_path)
    meta = {"key": "logistic_regression", "name": "LR", "cv": {"roc_auc_mean": 0.8}, "holdout": {"accuracy": 0.8}}
    model_manager.save_model("logistic_regression", fitted_pipeline, fitted_pipeline, meta, {})
    assert model_manager.stale_models() == []
    mpath = tmp_path / "logistic_regression" / "metadata.json"
    m = json.loads(mpath.read_text())
    m["versions"]["scikit-learn"] = "0.0.1"
    mpath.write_text(json.dumps(m))
    stale = model_manager.stale_models()
    assert stale and stale[0][0] == "logistic_regression" and "scikit-learn" in stale[0][1]


def test_corrupt_model_file_marks_model_stale(tmp_path, monkeypatch, fitted_pipeline):
    monkeypatch.setattr(model_manager, "models_dir", lambda: tmp_path)
    model_manager.save_model("knn", fitted_pipeline, fitted_pipeline, {"key": "knn", "cv": {}, "holdout": {}}, {})
    (tmp_path / "knn" / "pipeline_full.joblib").write_bytes(b"not a pickle")
    assert model_manager.stale_models()[0][0] == "knn"


def test_demo_mode_reads_environment(monkeypatch):
    from src.ui import state

    monkeypatch.setenv("TITANIC_DEMO_MODE", "true")
    assert state.demo_mode()
    monkeypatch.setenv("TITANIC_DEMO_MODE", "false")
    assert not state.demo_mode()


def test_requirements_pin_model_libraries_to_trained_versions():
    """On Python ≥ 3.12 the pinned versions must equal those recorded in the committed models."""
    if not model_manager.has_models():
        return
    from src import config

    req = (config.PROJECT_ROOT / "requirements.txt").read_text()
    saved = model_manager.load_metadata("xgboost")["versions"]
    assert f'scikit-learn=={saved["scikit-learn"]}; python_version >= "3.12"' in req
    assert f'xgboost-cpu=={saved["xgboost"]}; python_version >= "3.12"' in req
