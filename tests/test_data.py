"""Dataset loading and validation."""
import numpy as np
import pandas as pd

from src import config, data_loader
from src.data_validation import TRAIN_COLUMNS, coerce_types, validate_dataframe


def test_train_loads_with_expected_shape(train_df):
    assert train_df.shape == (891, 12)
    assert list(train_df.columns) == TRAIN_COLUMNS
    assert set(train_df["Survived"].unique()) == {0, 1}


def test_test_loads_without_target(test_df):
    assert test_df.shape == (418, 11)
    assert "Survived" not in test_df.columns
    assert test_df["PassengerId"].is_unique


def test_files_match_official_kaggle_checksums():
    for kind in ("train", "test"):
        st = data_loader.dataset_status(kind)
        assert st.exists and st.is_official, f"{kind}.csv is not the official Kaggle file"


def test_known_missing_value_counts(train_df):
    assert train_df["Age"].isna().sum() == 177
    assert train_df["Cabin"].isna().sum() == 687
    assert train_df["Embarked"].isna().sum() == 2


def test_real_data_is_valid(train_df, test_df):
    assert validate_dataframe(train_df, "train").is_valid
    assert validate_dataframe(test_df, "test").is_valid


def test_validation_catches_missing_columns(train_df):
    rep = validate_dataframe(train_df.drop(columns=["Sex", "Fare"]), "train")
    assert not rep.is_valid
    assert any("Sex" in e and "Fare" in e for e in rep.errors)


def test_validation_catches_invalid_values(train_df):
    bad = train_df.head(20).copy()
    bad.loc[0, "Sex"] = "unknown"
    bad.loc[1, "Pclass"] = 5
    bad.loc[2, "Embarked"] = "X"
    bad.loc[3, "Age"] = -4
    bad.loc[4, "Survived"] = 2
    errors = " ".join(validate_dataframe(bad, "train").errors)
    for word in ("Sex", "Pclass", "Embarked", "Age", "Survived"):
        assert word in errors


def test_validation_catches_duplicate_ids_and_rows(train_df):
    dup = pd.concat([train_df.head(5), train_df.head(2)])
    rep = validate_dataframe(dup, "train")
    assert rep.duplicate_rows == 2 and rep.duplicate_ids == 2
    assert not rep.is_valid


def test_validation_requires_both_target_classes(train_df):
    rep = validate_dataframe(train_df[train_df["Survived"] == 1], "train")
    assert any("only one class" in e for e in rep.errors)


def test_batch_validation_needs_only_core_columns():
    df = pd.DataFrame({"Pclass": [1], "Sex": ["female"], "Age": [30], "SibSp": [0], "Parch": [0], "Fare": [50.0], "Embarked": ["S"]})
    assert validate_dataframe(df, "batch").is_valid


def test_coerce_types_normalises_and_adds_optional_columns():
    df = pd.DataFrame({"Pclass": ["2"], "Sex": [" Female "], "Age": [""], "SibSp": [0], "Parch": [1], "Fare": ["13.5"], "Embarked": ["s"]})
    out = coerce_types(df)
    assert out.loc[0, "Sex"] == "female" and out.loc[0, "Embarked"] == "S"
    assert out.loc[0, "Pclass"] == 2 and np.isnan(out.loc[0, "Age"]) and out.loc[0, "Fare"] == 13.5
    for col in ("Name", "Ticket", "Cabin", "Title"):
        assert col in out.columns


def test_save_uploaded_rejects_invalid_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "raw_file", lambda kind: tmp_path / f"{kind}.csv")
    ok, msgs = data_loader.save_uploaded(b"a,b\n1,2\n", "train")
    assert not ok and msgs
    assert not (tmp_path / "train.csv").exists()


def test_save_uploaded_accepts_valid_csv(train_df, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "raw_file", lambda kind: tmp_path / f"{kind}.csv")
    ok, _ = data_loader.save_uploaded(train_df.head(50).to_csv(index=False).encode(), "train")
    assert ok and (tmp_path / "train.csv").exists()


def test_ensure_dataset_is_noop_when_present():
    assert data_loader.ensure_dataset() == ["Dataset already present locally."]
