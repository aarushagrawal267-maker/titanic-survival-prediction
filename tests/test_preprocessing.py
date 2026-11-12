"""Preprocessing pipeline and data-leakage prevention."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.preprocessing import build_pipeline, build_preprocessor, original_feature, transformed_feature_names
from src.feature_engineering import TitanicFeatureEngineer
from src.train import get_cv, split_data, split_xy


def test_transformed_matrix_has_no_missing_values(fitted_pipeline, train_df, test_df):
    pre = fitted_pipeline[:-1]
    for df in (train_df.drop(columns=["Survived"]), test_df):
        mat = pre.transform(df)
        assert not np.isnan(mat).any()
        assert mat.shape[1] == len(transformed_feature_names(fitted_pipeline))


def test_unseen_categories_do_not_crash(fitted_pipeline, test_df):
    odd = test_df.head(3).copy()
    odd["Cabin"] = ["Z99", np.nan, "Q1"]
    odd["Name"] = ["X, Sir. Unknown", np.nan, "Y, Mr. Z"]
    p = fitted_pipeline.predict_proba(odd)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_missing_values_in_every_imputable_column(fitted_pipeline, test_df):
    row = test_df.head(1).copy()
    for c in ("Age", "Fare", "Embarked", "Cabin", "Ticket", "Name"):
        row[c] = np.nan
    assert 0 <= fitted_pipeline.predict_proba(row)[0, 1] <= 1


def test_scalers_and_encodings(train_df):
    eng = TitanicFeatureEngineer().fit_transform(train_df)
    std = build_preprocessor("standard").fit(eng)
    mat = std.transform(eng)
    names = list(std.get_feature_names_out())
    age = mat[:, names.index("num__Age")]
    assert abs(age.mean()) < 1e-9 and abs(age.std() - 1) < 1e-6
    ordinal = build_preprocessor("none", pclass_encoding="ordinal").fit(eng)
    assert "num__Pclass" in ordinal.get_feature_names_out()
    onehot = build_preprocessor("none", pclass_encoding="onehot").fit(eng)
    assert {"cat__Pclass_1", "cat__Pclass_2", "cat__Pclass_3"} <= set(onehot.get_feature_names_out())


def test_original_feature_mapping():
    assert original_feature("cat__Sex_female") == "Sex"
    assert original_feature("cat__TitleGroup_Master") == "TitleGroup"
    assert original_feature("num__FarePerPerson") == "FarePerPerson"
    assert original_feature("bin__CabinKnown") == "CabinKnown"


# ── leakage ─────────────────────────────────────────────────────────
def test_holdout_is_disjoint_stratified_and_reproducible(train_df):
    X_dev, X_hold, y_dev, y_hold = split_data(train_df)
    assert len(X_dev) == 712 and len(X_hold) == 179
    assert set(X_dev["PassengerId"]).isdisjoint(X_hold["PassengerId"])
    assert abs(y_dev.mean() - y_hold.mean()) < 0.01
    X_dev2, *_ = split_data(train_df)
    assert X_dev.index.equals(X_dev2.index)


def test_imputers_learn_only_from_training_fold(train_df):
    """Fit the pipeline on one CV training fold: every learned statistic must equal the fold's own statistic."""
    X, y = split_xy(train_df)
    tr, va = next(get_cv().split(X, y))
    pipe = build_pipeline(LogisticRegression(max_iter=2000), scaler="standard", age_strategy="median").fit(X.iloc[tr], y.iloc[tr])
    fold = X.iloc[tr]
    num_imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["impute"]
    cols = list(pipe.named_steps["preprocess"].transformers_[0][2])
    age_median_learned = num_imputer.statistics_[cols.index("Age")]
    assert age_median_learned == fold["Age"].median()
    assert age_median_learned != X["Age"].median() or fold["Age"].median() == X["Age"].median()
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["scale"]
    fold_fare = fold["Fare"].fillna(fold["Fare"].median())
    assert np.isclose(scaler.mean_[cols.index("Fare")], fold_fare.mean())


def test_feature_engineer_statistics_come_from_fit_data_only(train_df):
    X, y = split_xy(train_df)
    tr, va = next(get_cv().split(X, y))
    fe = TitanicFeatureEngineer().fit(X.iloc[tr])
    fold_ages = X.iloc[tr].assign(T=TitanicFeatureEngineer().fit(X.iloc[tr]).transform(X.iloc[tr])["TitleGroup"])
    assert fe.age_median_ == X.iloc[tr]["Age"].median()
    assert sum(fe.ticket_counts_.values()) == X.iloc[tr]["Ticket"].notna().sum()
    # validation rows never contribute to learned ticket counts
    val_only = set(X.iloc[va]["Ticket"]) - set(X.iloc[tr]["Ticket"])
    assert val_only.isdisjoint(fe.ticket_counts_)


def test_target_is_never_used_by_preprocessing(train_df):
    """Shuffling the labels must not change any transformed value."""
    X, y = split_xy(train_df)
    pipe_a = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)
    pipe_b = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y.sample(frac=1, random_state=1).reset_index(drop=True).set_axis(y.index))
    assert np.allclose(pipe_a[:-1].transform(X), pipe_b[:-1].transform(X))


def test_pipeline_clone_is_unfitted():
    pipe = build_pipeline(LogisticRegression())
    cloned = clone(pipe)
    assert not hasattr(cloned.named_steps["features"], "ticket_counts_")
