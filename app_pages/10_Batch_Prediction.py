"""Batch Prediction — score a CSV of passengers, check drift, download results."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.batch_prediction import drift_report, read_csv, score_batch, template_csv
from src.data_validation import BATCH_OPTIONAL, BATCH_REQUIRED
from src.ui import charts, state
from src.ui.components import disclaimer, kpi_tiles, learn, note, page_header, section
from src.ui.theme import BRASS, PERISHED, SURVIVED, VIOLET

page_header("Predict · 11", "Batch Prediction",
            "Upload a CSV with many passengers. Each row is validated, missing values are handled by the saved pipeline, and you get "
            "probabilities and predictions to download. No <code>Survived</code> column needed.")
df = state.require_data()
key = state.require_models()
pipe = state.get_pipeline(key, "full")

section("1 · Prepare your file", "01", "description")
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown(f"**Required columns:** `{'`, `'.join(BATCH_REQUIRED)}`  \n**Optional:** `{'`, `'.join(BATCH_OPTIONAL)}` "
                "(Name → title, Cabin → deck, Ticket → group size). Missing Age, Fare or Embarked values are imputed.")
    st.download_button("Download CSV template", template_csv(), "batch_template.csv", "text/csv", icon=":material/download:")
with c2:
    learn("Same pipeline, many rows", "Batch scoring runs exactly the same saved Pipeline as single predictions — feature engineering, imputation and "
          "encoding learned from the training data are re-applied, never re-fitted on your file.")

section("2 · Upload", "02", "upload_file")
u1, u2 = st.columns([2, 1])
up = u1.file_uploader("Passenger CSV", type="csv", key="batch_up")
if u2.button("Use Kaggle test.csv as an example", width="stretch"):
    st.session_state["batch_example"] = True
src_df, src_name = None, None
if up is not None:
    try:
        src_df, src_name = read_csv(up.getvalue()), up.name
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
elif st.session_state.get("batch_example"):
    try:
        src_df, src_name = state.get_test(), "test.csv (Kaggle competition)"
    except FileNotFoundError:
        st.error("test.csv is not available.")
if src_df is None:
    note("Upload a file or use the example to continue.", "touch_app")
    st.stop()

results, report = score_batch(pipe, src_df)
section("3 · Validation", "03", "rule")
v1, v2 = st.columns([1, 1.6])
with v1:
    kpi_tiles([{"label": "Rows", "count_to": report["rows"], "icon": "table_rows", "accent": BRASS},
               {"label": "Status", "value": "Valid" if report["valid"] else "Invalid", "icon": "rule", "tone": "survived" if report["valid"] else "perished"}])
with v2:
    for e in report["errors"]:
        st.error(e)
    for w in report["warnings"]:
        st.warning(w)
    for i in report["info"]:
        st.info(i)
    if report["valid"] and not (report["warnings"] or report["info"]):
        st.success("All checks passed.")
if results is None:
    st.stop()

section("4 · Results", "04", "query_stats")
rate = results["Predicted"].mean()
kpi_tiles([
    {"label": "Passengers scored", "count_to": len(results), "icon": "groups", "accent": BRASS},
    {"label": "Predicted survivors", "count_to": int(results["Predicted"].sum()), "icon": "health_and_safety", "accent": SURVIVED},
    {"label": "Predicted survival rate", "value": f"{rate:.1%}", "icon": "percent", "sub": f"training data: {df['Survived'].mean():.1%}", "accent": SURVIVED},
    {"label": "Mean probability", "value": f"{results['Survival probability'].mean():.1%}", "icon": "functions", "accent": VIOLET},
    {"label": "Uncertain (40–60%)", "count_to": int(results["Survival probability"].between(0.4, 0.6).sum()), "icon": "help", "accent": PERISHED},
])
r1, r2 = st.columns([1.4, 1])
with r1:
    st.dataframe(results, hide_index=True, width="stretch", height=420, column_config={
        "Survival probability": st.column_config.ProgressColumn("Survival probability", min_value=0, max_value=1, format="%.2f"),
        "Non-survival probability": st.column_config.NumberColumn(format="%.2f"),
    })
with r2:
    fig = go.Figure(go.Histogram(x=results["Survival probability"], nbinsx=20, marker=dict(color=SURVIVED, line=dict(color="#050b1a", width=1)),
                                 hovertemplate="%{x}<br>%{y} passengers<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=BRASS, dash="dot", width=1.5), annotation_text="threshold", annotation_font_color=BRASS)
    fig.update_layout(title="Distribution of survival probabilities", height=420)
    fig.update_xaxes(title="Survival probability", range=[0, 1])
    fig.update_yaxes(title="Passengers")
    charts.show(fig)
st.download_button("Download predictions (CSV)", results.to_csv(index=False), f"predictions_{(src_name or 'batch').split('.')[0]}.csv", "text/csv",
                   type="primary", icon=":material/download:")
st.caption(f"Model: {state.model_name(key)} · source: {src_name}")

section("5 · Input drift check", "05", "monitor_heart")
dr = drift_report(df, src_df)
st.dataframe(dr, hide_index=True, width="stretch", column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")})
note("The Population Stability Index compares each input's distribution in your file with the training data: < 0.1 stable, 0.1–0.25 moderate shift, "
     "> 0.25 large shift. Predictions for passengers unlike the training data are less reliable.", "monitor_heart")
disclaimer()
