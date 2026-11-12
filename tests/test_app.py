"""Smoke tests: every Streamlit page runs without raising (Streamlit AppTest).

Pages that need trained models show an onboarding panel instead of failing when
``models/`` is empty, so these tests pass on a fresh clone too.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted(p.name for p in (ROOT / "app_pages").glob("*.py"))


def _app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    return at


def test_home_runs():
    at = _app()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = _app()
    at.switch_page(f"app_pages/{page}")
    at.run()
    assert not at.exception, f"{page}: {at.exception}"


def _has_models() -> bool:
    import sys

    sys.path.insert(0, str(ROOT))
    from src import model_manager

    return model_manager.has_models()


@pytest.mark.skipif(not _has_models(), reason="needs trained models (python scripts/train_all.py)")
def test_passenger_prediction_flow():
    at = _app()
    at.switch_page("app_pages/9_Passenger_Prediction.py")
    at.run()
    at.selectbox(key="preset").select("Fictional · First-class mother").run()
    submit = [b for b in at.button if "Estimate survival" in str(b.label)]
    assert submit, "submit button missing"
    submit[0].click().run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Est. survival" in text and "Why the model estimated" in text


@pytest.mark.skipif(not _has_models(), reason="needs trained models")
def test_kaggle_submission_flow():
    at = _app()
    at.switch_page("app_pages/11_Kaggle_Submission.py")
    at.run()
    [b for b in at.button if "Generate submission" in str(b.label)][0].click().run()
    assert not at.exception, at.exception
    assert any("418" in m.value for m in at.markdown)


@pytest.mark.skipif(not _has_models(), reason="needs trained models")
def test_batch_example_flow():
    at = _app()
    at.switch_page("app_pages/10_Batch_Prediction.py")
    at.run()
    [b for b in at.button if "Kaggle test.csv" in str(b.label)][0].click().run()
    assert not at.exception, at.exception
    assert any(df.value.shape[0] == 418 for df in at.dataframe)


@pytest.mark.skipif(not _has_models(), reason="needs trained models")
def test_beat_the_model_round():
    at = _app()
    at.switch_page("app_pages/14_Beat_the_Model.py")
    at.run()
    [b for b in at.button if "survived" in str(b.label)][0].click().run()
    assert not at.exception, at.exception
    assert any("What actually happened" in m.value for m in at.markdown)
