import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_test, load_train  # noqa: E402
from src.models import get_spec  # noqa: E402
from src.preprocessing import build_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def train_df():
    return load_train()


@pytest.fixture(scope="session")
def test_df():
    return load_test()


@pytest.fixture(scope="session")
def fitted_pipeline(train_df):
    """A quick Logistic Regression pipeline fitted on the real training data."""
    X, y = train_df.drop(columns=["Survived"]), train_df["Survived"]
    return build_pipeline(get_spec("logistic_regression").make(), scaler="standard").fit(X, y)


@pytest.fixture(scope="session")
def fitted_tree_pipeline(train_df):
    X, y = train_df.drop(columns=["Survived"]), train_df["Survived"]
    return build_pipeline(get_spec("random_forest").make(n_estimators=60, max_depth=6), scaler="none").fit(X, y)
