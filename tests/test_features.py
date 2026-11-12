"""Feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (ENGINEERED_FEATURES, TitanicFeatureEngineer, add_basic_features, extract_deck, extract_title,
                                     family_category, group_title, infer_title_group)


@pytest.mark.parametrize("name,title", [
    ("Braund, Mr. Owen Harris", "Mr"), ("Heikkinen, Miss. Laina", "Miss"), ("Palsson, Master. Gosta Leonard", "Master"),
    ("Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)", "Countess"), ("Uruchurtu, Don. Manuel E", "Don"), (None, None),
])
def test_extract_title(name, title):
    assert extract_title(name) == title


def test_title_grouping():
    assert group_title("Mlle") == "Miss" and group_title("Mme") == "Mrs" and group_title("Capt") == "Other"
    assert group_title("Jonkheer") == "Other" and group_title("Master") == "Master" and group_title(None) is None


def test_inferred_title_when_no_name():
    assert infer_title_group("male", 6, 1, 1) == "Master"
    assert infer_title_group("male", 40, 0, 0) == "Mr"
    assert infer_title_group("female", 15, 0, 0) == "Miss"
    assert infer_title_group("female", 35, 1, 0) == "Mrs"


def test_family_features(train_df):
    e = add_basic_features(train_df)
    assert (e["FamilySize"] == train_df["SibSp"] + train_df["Parch"] + 1).all()
    assert ((e["FamilySize"] == 1) == (e["IsAlone"] == 1)).all()
    assert family_category(1) == "Alone" and family_category(3) == "Small family" and family_category(7) == "Large family"
    assert np.allclose(e["FarePerPerson"], train_df["Fare"] / e["FamilySize"])


def test_cabin_features():
    assert extract_deck("C85") == "C" and extract_deck(np.nan) == "Unknown" and extract_deck("T") == "Unknown"
    e = add_basic_features(pd.DataFrame({"Name": ["A, Mr. B"], "Sex": ["male"], "Age": [30], "SibSp": [0], "Parch": [0], "Fare": [10.0], "Cabin": ["B22"]}))
    assert e.loc[0, "CabinKnown"] == 1 and e.loc[0, "Deck"] == "B"


def test_transformer_outputs_expected_columns_and_no_target(train_df):
    fe = TitanicFeatureEngineer().fit(train_df)
    out = fe.transform(train_df)
    assert "Survived" not in out.columns
    for f in ENGINEERED_FEATURES:
        assert f in out.columns
    assert out["Age"].isna().sum() == 0, "title-group imputation should fill every age"
    assert out["AgeKnown"].sum() == train_df["Age"].notna().sum()


def test_base_feature_set_and_drop(train_df):
    assert list(TitanicFeatureEngineer(feature_set="base").fit(train_df).transform(train_df).columns) == \
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    out = TitanicFeatureEngineer(drop_features=("Deck", "TitleGroup")).fit(train_df).transform(train_df)
    assert "Deck" not in out.columns and "TitleGroup" not in out.columns


def test_median_strategy_leaves_age_for_imputer(train_df):
    out = TitanicFeatureEngineer(age_strategy="median").fit(train_df).transform(train_df)
    assert out["Age"].isna().sum() == 177
    assert (out.loc[train_df["Age"].isna(), "AgeGroup"] == "Unknown").all()


def test_ticket_group_size_learned_only_from_fit_data(train_df):
    fit_part, other = train_df.iloc[:400], train_df.iloc[400:]
    fe = TitanicFeatureEngineer().fit(fit_part)
    assert set(fe.ticket_counts_) <= set(fit_part["Ticket"].astype(str))
    single = other.head(1).copy()
    out = fe.transform(single)
    expected = max(fe.ticket_counts_.get(single["Ticket"].iloc[0], 0), 1)
    assert out["TicketGroupSize"].iloc[0] == expected


def test_ticket_group_falls_back_to_family_size():
    df = pd.DataFrame({"Pclass": [3], "Name": [np.nan], "Sex": ["male"], "Age": [20.0], "SibSp": [1], "Parch": [2],
                       "Ticket": [np.nan], "Fare": [20.0], "Cabin": [np.nan], "Embarked": ["S"]})
    fe = TitanicFeatureEngineer().fit(df)
    assert fe.transform(df)["TicketGroupSize"].iloc[0] == 4


def test_transformer_requires_dataframe(train_df):
    with pytest.raises(TypeError):
        TitanicFeatureEngineer().fit(train_df.to_numpy())
