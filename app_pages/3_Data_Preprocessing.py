"""Data Preprocessing — leakage-safe pipeline, imputation, encoding, scaling, outliers."""
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import RobustScaler, StandardScaler

from src import eda
from src.feature_engineering import TitanicFeatureEngineer, add_basic_features
from src.preprocessing import build_pipeline, preprocessing_preview
from src.train import load_experiments
from src.ui import charts, state
from src.ui.components import flow, learn, note, page_header, section
from src.ui.theme import AQUA, BRASS, PERISHED, SURVIVED, VIOLET

page_header("Data · 03", "Data Preprocessing",
            "Missing values, categorical encoding, scaling and outliers — all wrapped in a scikit-learn <code>Pipeline</code> + "
            "<code>ColumnTransformer</code> so nothing is learned from validation data.")
df = state.require_data()
exp = load_experiments()

section("The leakage-safe pipeline", "01", "account_tree")
flow([
    ("Raw passenger row", "PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.", "input"),
    ("Feature engineer", "Title, family, deck, ticket groups; Age imputed by title median (learned in fit).", "auto_fix_high"),
    ("Numeric branch", "Median imputer → optional StandardScaler / RobustScaler.", "straighten"),
    ("Categorical branch", "Most-frequent imputer → OneHotEncoder(handle_unknown='ignore').", "category"),
    ("Binary flags", "IsAlone, CabinKnown, AgeKnown — passed through.", "toggle_on"),
    ("Model", "Any of the 7 classifiers, fitted on the same transformed matrix.", "model_training"),
])
c1, c2 = st.columns([1.1, 1])
with c1:
    learn("What is data leakage?", "Leakage happens when information from the validation/test data sneaks into training — e.g. computing the median age "
          "on <i>all</i> passengers before splitting. Scores then look better than they will be on new passengers. Because every imputer, encoder "
          "and scaler sits <b>inside</b> the Pipeline, cross-validation refits them on each training fold only.")
    note("The Kaggle <code>test.csv</code> is never used for fitting anything — it is only scored at the very end on the Kaggle Submission page.", "lock")
with c2:
    with st.expander("Show the pipeline code", expanded=False):
        st.code(inspect.getsource(build_pipeline), language="python")
    st.code(repr(build_pipeline(__import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression())), language="text")

section("Try the preprocessing", "02", "science")
o1, o2, o3 = st.columns(3)
scaler = o1.selectbox("Numeric scaling", ["standard", "robust", "none"], format_func={"standard": "StandardScaler", "robust": "RobustScaler", "none": "No scaling"}.get)
penc = o2.selectbox("Pclass encoding", ["onehot", "ordinal"], format_func={"onehot": "One-hot (3 columns)", "ordinal": "Ordinal (1, 2, 3)"}.get)
age_s = o3.selectbox("Age imputation", ["title_group", "median"], format_func={"title_group": "Median per title group", "median": "Global median"}.get)
raw, eng, out = preprocessing_preview(df, n=8, scaler=scaler, pclass_encoding=penc, age_strategy=age_s)
t1, t2, t3 = st.tabs(["① Raw rows", "② After feature engineering", f"③ Model matrix ({out.shape[1]} columns)"])
t1.dataframe(raw, hide_index=True, width="stretch")
t2.dataframe(eng, hide_index=True, width="stretch")
t3.dataframe(out, hide_index=True, width="stretch")
st.caption("Preview fitted on the full training file purely for display. Rows with missing Age (e.g. passenger 6) show the imputed value in step ②.")

section("Missing-value treatment", "03", "healing")
i1, i2 = st.columns([1.3, 1])
basic = add_basic_features(df)
fe = TitanicFeatureEngineer(age_strategy="title_group").fit(df)
imputed = fe.transform(df)["Age"]
with i1:
    fig = go.Figure()
    fig.add_histogram(x=df["Age"].dropna(), name="Recorded ages", nbinsx=40, marker=dict(color=SURVIVED, line=dict(color="#050b1a", width=1)), opacity=0.85)
    fig.add_histogram(x=imputed[df["Age"].isna()], name="Imputed (title-group median)", nbinsx=40, marker=dict(color=BRASS, line=dict(color="#050b1a", width=1)), opacity=0.9)
    fig.add_histogram(x=[df["Age"].median()] * int(df["Age"].isna().sum()), name="Imputed (global median)", nbinsx=40,
                      marker=dict(color="rgba(232,105,58,0.55)"), visible="legendonly")
    fig.update_layout(barmode="overlay", title="Age: recorded vs imputed values", height=380)
    fig.update_xaxes(title="Age")
    fig.update_yaxes(title="Passengers")
    charts.show(fig)
with i2:
    med = pd.DataFrame({"Title group": list(fe.age_by_title_.keys()), "Median age (train)": list(fe.age_by_title_.values())})
    med["Passengers with missing age"] = med["Title group"].map(basic[df["Age"].isna()]["TitleGroup"].value_counts()).fillna(0).astype(int)
    st.dataframe(med, hide_index=True, width="stretch")
    note("A single global median (28) would make every missing-age 'Master' (young boy) look like an adult. The title-group median keeps "
         "children young — a better guess that the ablation and encoding experiments test rather than assume.", "lightbulb")
st.dataframe(eda.missing_table(df), hide_index=True, width="stretch", column_config={"strategy": st.column_config.TextColumn(width="large")})

if exp.get("encoding"):
    e = pd.DataFrame(exp["encoding"]["rows"])
    st.markdown("**Experiment — Pclass encoding × age-imputation strategy** (5-fold CV on the development set)")
    piv = e.pivot_table(index=["model", "pclass_encoding"], columns="age_strategy", values="roc_auc").round(4)
    st.dataframe(piv, width="stretch")

section("Scaling — does it help?", "04", "straighten")
s1, s2 = st.columns([1, 1.2])
with s1:
    fare = df[["Fare"]].dropna()
    std = StandardScaler().fit_transform(fare).ravel()
    rob = RobustScaler().fit_transform(fare).ravel()
    fig = go.Figure()
    for name, vals, color in (("StandardScaler", std, SURVIVED), ("RobustScaler", rob, AQUA)):
        fig.add_box(x=vals, name=name, marker=dict(color=color, size=4), line=dict(width=1.3), boxpoints="outliers")
    fig.update_layout(title="Fare after scaling (outliers stay extreme)", height=320, showlegend=False)
    charts.show(fig)
    learn("Why scale?", "Distance- and margin-based models (KNN, SVM) and regularised Logistic Regression treat a £500 fare and a 5-year age difference "
          "on the same numeric footing. Scaling puts features on comparable ranges. Tree models split on thresholds, so they don't need it.")
with s2:
    if exp.get("scaling"):
        sc = pd.DataFrame(exp["scaling"]["rows"])
        fig = go.Figure()
        for i, (s_name, color) in enumerate((("none", "rgba(180,194,217,0.5)"), ("standard", SURVIVED), ("robust", AQUA))):
            sub = sc[sc["scaler"] == s_name]
            fig.add_bar(name={"none": "No scaling", "standard": "StandardScaler", "robust": "RobustScaler"}[s_name], x=sub["model"], y=sub["roc_auc"],
                        error_y=dict(type="data", array=sub["roc_auc_std"], color="rgba(255,255,255,0.35)", thickness=1),
                        marker=dict(color=color, cornerradius=3), hovertemplate="%{x}<br>CV ROC-AUC %{y:.3f}<extra></extra>")
        fig.update_layout(barmode="group", title="Scaling experiment — CV ROC-AUC (default hyper-parameters)", height=380)
        fig.update_yaxes(range=[max(0.5, sc["roc_auc"].min() - 0.06), min(1, sc["roc_auc"].max() + 0.04)], title="CV ROC-AUC")
        charts.show(fig)
        best = exp["scaling"]["best"]
        st.caption("Chosen scaler per model: " + " · ".join(f"**{state.model_name(k)}** → {v}" for k, v in best.items()))
    else:
        st.info("Run `python scripts/train_all.py` to execute the scaling experiment.")

section("Outlier investigation", "05", "scatter_plot")
rep = eda.outlier_report(df).round(2)
st.dataframe(rep, hide_index=True, width="stretch")
b1, b2, b3 = st.columns(3)
for col, feat in zip((b1, b2, b3), ("Age", "Fare", "FamilySize")):
    with col:
        fig = go.Figure()
        for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
            fig.add_box(y=basic[basic["Survived"] == val][feat], name=name, marker=dict(color=color, size=4), line=dict(width=1.3), boxpoints="outliers")
        fig.update_layout(title=f"{feat} by outcome", height=330, showlegend=False)
        charts.show(fig, key=f"box_{feat}")
st.markdown("**The most extreme fares** — these are real, legitimate passengers")
st.dataframe(eda.extreme_passengers(df, "Fare", 8), hide_index=True, width="stretch",
             column_config={"Fare": st.column_config.NumberColumn("Fare (£)", format="£%.2f")})
note("Decision: <b>no passengers are removed.</b> Extreme fares belong to real first-class suites (and fares are per ticket, shared by groups); "
     "large families are real families. Instead the pipeline uses median imputation, an optional RobustScaler, and per-person fare, "
     "and tree models are inherently robust to extreme values.", "gavel")
