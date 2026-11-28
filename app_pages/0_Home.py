"""Home dashboard."""
import streamlit as st

from src import eda
from src.evaluate import METRIC_LABELS
from src.ui import charts, state
from src.ui.components import disclaimer, flow, kpi_tiles, learn, note, pill, section
from src.ui.ship import class_zones, render_ship
from src.ui.theme import PERISHED, SURVIVED

df = state.require_data()
ov = eda.overview(df)
reg = state.get_registry()
best = state.best_key()

st.markdown(f"""
<div class="tml-hero">
  <div class="tml-eyebrow">Passenger survival prediction</div>
  <h1>Titanic ML Lab</h1>
  <div class="tag">An end-to-end machine-learning laboratory built on the historical passenger records of RMS Titanic —
  explore the data, train and compare seven algorithms, look inside their decisions, and estimate survival chances for any passenger profile.</div>
  <div class="chips">{pill(f"{ov['passengers']:,} real passenger records")}{pill("7 algorithms")}{pill("Leakage-safe pipelines")}{pill("SHAP explainability")}{pill("3D ship explorer")}</div>
</div>""", unsafe_allow_html=True)

st.write("")
render_ship("hero", height=500, key="hero_ship", zones=class_zones(df),
            stats={"total": ov["passengers"], "survived": ov["survivors"]})

best_meta = reg.get(best, {}) if best else {}
kpi_tiles([
    {"label": "Passenger records", "count_to": ov["passengers"], "sub": "Kaggle train.csv (labelled)"},
    {"label": "Survivors", "count_to": ov["survivors"], "sub": f"{ov['non_survivors']:,} did not survive"},
    {"label": "Survival rate", "value": f"{ov['survival_rate']:.1%}", "sub": "in the labelled data"},
    {"label": "Models trained", "count_to": len(reg), "sub": "tuned with 5-fold CV"},
    {"label": "Best model", "value": state.model_name(best) if best else "—",
     "sub": f"CV ROC-AUC {best_meta.get('cv', {}).get('roc_auc_mean', 0):.3f}" if best else "train models to see"},
    {"label": "Best · holdout accuracy", "value": f"{best_meta.get('holdout', {}).get('accuracy', 0):.1%}" if best else "—",
     "sub": f"F1 {best_meta.get('holdout', {}).get('f1', 0):.3f} · AUC {best_meta.get('holdout', {}).get('roc_auc', 0):.3f}" if best else ""},
])

c1, c2, c3, c4 = st.columns(4)
c1.page_link("app_pages/9_Passenger_Prediction.py", label="Predict a passenger", icon=":material/person_search:", width="stretch")
c2.page_link("app_pages/13_Ship_Explorer_3D.py", label="Explore the ship in 3D", icon=":material/directions_boat:", width="stretch")
c3.page_link("app_pages/14_Beat_the_Model.py", label="Play: Beat the Model", icon=":material/sports_esports:", width="stretch")
c4.page_link("app_pages/2_Exploratory_Analysis.py", label="Explore the data", icon=":material/insights:", width="stretch")

section("The machine-learning journey", "01", "route")
learn("What is this app?", "A binary classification project: the model learns from 891 passengers whose fate is known (Survived = 1 or 0) "
      "and estimates the probability of survival for new passenger profiles. Follow the pages in the sidebar from top to bottom to walk the full workflow.")
flow([
    ("Acquire & validate", "Kaggle CSVs, checksum-verified; schema, types, missing values, duplicates.", "dataset"),
    ("Explore", "Survival by sex, class, age, port, family, fare; correlations; automatic insights.", "insights"),
    ("Preprocess", "Imputation, encoding and scaling inside a leakage-safe Pipeline.", "tune"),
    ("Engineer features", "Title, family size, deck, ticket groups… and ablation tests.", "auto_fix_high"),
    ("Train & tune", "7 algorithms, stratified 5-fold CV, RandomizedSearchCV.", "model_training"),
    ("Evaluate & explain", "Holdout metrics, ROC/PR, calibration, SHAP & permutation importance.", "psychology"),
    ("Predict", "Single passengers, batch CSVs and the Kaggle submission.", "person_search"),
])

section("At a glance", "02", "monitoring")
e = eda.enrich(df)
g1, g2, g3 = st.columns([1, 1.15, 1.15])
with g1:
    charts.show(charts.donut([ov["non_survivors"], ov["survivors"]], ["Did not survive", "Survived"], [PERISHED, SURVIVED],
                                 "Overall outcome", center=f"{ov['survival_rate']:.0%}<br><span style='font-size:11px'>survived</span>"))
with g2:
    charts.show(charts.survival_rate_bar(eda.survival_by(e, "Class"), "Class", "Survival rate by class", overall=ov["survival_rate"]))
with g3:
    sx = eda.survival_by(e, "Sex")
    sx["Sex"] = sx["Sex"].str.capitalize()
    charts.show(charts.survival_rate_bar(sx, "Sex", "Survival rate by sex", overall=ov["survival_rate"]))

if reg:
    section("Model leaderboard", "03", "leaderboard")
    import pandas as pd

    rows = []
    for k, m in reg.items():
        rows.append({"Model": state.model_name(k), "CV ROC-AUC": m["cv"]["roc_auc_mean"], "± std": m["cv"]["roc_auc_std"],
                     **{f"Holdout {METRIC_LABELS[x]}": m["holdout"][x] for x in ("accuracy", "f1", "roc_auc")}})
    lb = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
    lb.insert(0, "Rank", range(1, len(lb) + 1))
    st.dataframe(lb, hide_index=True, width="stretch", column_config={
        "CV ROC-AUC": st.column_config.ProgressColumn("CV ROC-AUC", min_value=0.5, max_value=1.0, format="%.3f"),
        "± std": st.column_config.NumberColumn(format="%.3f"),
        **{c: st.column_config.NumberColumn(format="%.3f") for c in lb.columns if c.startswith("Holdout")},
    })
    note("Ranking uses <b>cross-validated ROC-AUC on the development set</b> (never the holdout and never accuracy alone). "
         "Holdout scores are measured once, on 179 passengers no model saw during tuning.", "rule")
else:
    state.require_models()

disclaimer()
