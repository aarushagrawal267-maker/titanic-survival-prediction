"""Model Evaluation — holdout metrics, threshold analysis, confusion matrix, ROC/PR, calibration, learning curve."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import METRIC_EXPLANATIONS, METRIC_LABELS, METRICS, compute_learning_curve, compute_metrics
from src.models import MODEL_ORDER
from src.train import split_data
from src.ui import charts, state
from src.ui.components import kpi_tiles, learn, note, page_header, section
from src.ui.theme import BRASS, PERISHED, SURVIVED, VIOLET

page_header("Modeling · 07", "Model Evaluation",
            "How good is the model, really? Every number here is measured on the <b>179-passenger holdout set</b> that was never used for training or tuning.")
df = state.require_data()
key = state.require_models()
reg = state.get_registry()
keys = [k for k in MODEL_ORDER if k in reg]
key = st.selectbox("Model to evaluate", keys, index=keys.index(key), format_func=state.model_name, key="eval_model")
ev = state.get_evaluation(key)
meta = reg[key]
y = np.asarray(ev["holdout_y"])
p = np.asarray(ev["holdout_prob"])

section("Holdout performance", "01", "verified")
thr = st.slider("Decision threshold — predict 'survived' when probability ≥ threshold", 0.05, 0.95, 0.50, 0.01, key="eval_thr")
m = compute_metrics(y, p, thr)
m50 = ev["metrics"]
kpi_tiles([{"label": METRIC_LABELS[k], "value": f"{m[k]:.3f}", "icon": ic,
            "sub": (f"{(m[k] - m50[k]) * 100:+.1f} pts vs 0.50" if abs(thr - 0.5) > 1e-9 and k != "roc_auc" else METRIC_EXPLANATIONS[k][:62] + "…"),
            "accent": col} for k, ic, col in zip(METRICS, ["check_circle", "target", "radar", "balance", "show_chart"], [BRASS, SURVIVED, SURVIVED, VIOLET, BRASS])])
learn("The threshold trade-off", "Models output a probability; the threshold turns it into a yes/no. Lowering it finds more survivors (recall ↑) "
      "but raises false alarms (precision ↓). ROC-AUC does not depend on the threshold at all.")

c1, c2 = st.columns([1, 1.2])
with c1:
    norm = st.toggle("Show row percentages", value=False, key="cm_norm")
    pred = (p >= thr).astype(int)
    cm = np.array([[((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()], [((y == 1) & (pred == 0)).sum(), ((y == 1) & (pred == 1)).sum()]])
    charts.show(charts.confusion(cm, f"Confusion matrix @ threshold {thr:.2f}", normalize=norm))
with c2:
    tt = pd.DataFrame(ev["threshold_table"])
    fig = go.Figure()
    for i, k in enumerate(["precision", "recall", "f1", "accuracy"]):
        fig.add_scatter(x=tt["threshold"], y=tt[k], name=METRIC_LABELS[k], mode="lines", line=dict(width=2.4, color=charts.CATEGORICAL[i]))
    fig.add_vline(x=thr, line=dict(color="#ffffff", dash="dot", width=1))
    fig.update_layout(title="Metrics across thresholds", height=380)
    fig.update_xaxes(title="Threshold")
    fig.update_yaxes(title="Score", range=[0, 1.02])
    charts.show(fig)

section("Ranking quality", "02", "show_chart")
r1, r2 = st.columns(2)
with r1:
    charts.show(charts.roc_curves({state.model_name(key): {"fpr": ev["roc"]["fpr"], "tpr": ev["roc"]["tpr"], "auc": ev["metrics"]["roc_auc"]}}))
with r2:
    charts.show(charts.pr_curve(ev["pr"]["precision"], ev["pr"]["recall"], ev["pr"]["average_precision"], ev["base_rate"]))

section("Probabilities & calibration", "03", "tune")
q1, q2 = st.columns(2)
with q1:
    fig = go.Figure()
    for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
        fig.add_histogram(x=p[y == val], name=f"Actually: {name}", nbinsx=20, marker=dict(color=color, line=dict(color="#050b1a", width=1)), opacity=0.8)
    fig.add_vline(x=thr, line=dict(color="#ffffff", dash="dot", width=1))
    fig.update_layout(barmode="overlay", title="Predicted probability by actual outcome", height=400)
    fig.update_xaxes(title="Predicted survival probability", range=[0, 1])
    fig.update_yaxes(title="Passengers")
    charts.show(fig)
with q2:
    charts.show(charts.calibration_plot(ev["calibration"]["mean_predicted"], ev["calibration"]["fraction_positive"]))
learn("Calibration", "If a model says 70% for a group of passengers, about 70% of them should have survived. Points on the diagonal = well calibrated. "
      "Calibration matters here because the app reports probabilities, not just yes/no.")

section("Classification report & cross-validation", "04", "table")
k1, k2 = st.columns([1, 1.2])
with k1:
    rep = pd.DataFrame(ev["classification_report"]).T.rename(columns={"f1-score": "f1"})
    st.dataframe(rep.round(3), width="stretch")
    st.caption("Classification report at threshold 0.50.")
with k2:
    folds = meta["cv"]["folds"]
    cvdf = pd.DataFrame({METRIC_LABELS[x]: folds[x] for x in METRICS}, index=[f"Fold {i + 1}" for i in range(len(folds["roc_auc"]))])
    cvdf.loc["Mean"] = cvdf.mean()
    cvdf.loc["Std"] = cvdf.iloc[:-1].std(ddof=0)
    st.dataframe(cvdf.round(3), width="stretch")
    st.caption("5-fold stratified cross-validation on the 712-passenger development set.")

section("Learning curve", "05", "trending_up")
if st.button("Compute learning curve (≈10–60 s)", key="lc_btn"):
    with st.spinner("Re-fitting on growing subsets…"):
        X_dev, _, y_dev, _ = split_data(df)
        st.session_state[f"lc_{key}"] = compute_learning_curve(state.get_pipeline(key, "dev"), X_dev, y_dev)
if f"lc_{key}" in st.session_state:
    charts.show(charts.learning_curve_plot(st.session_state[f"lc_{key}"], f"Learning curve — {state.model_name(key)}"))
    st.caption("A large, persistent gap between training and validation suggests overfitting; curves that are still rising suggest more data would help.")

note("<b>Validation vs Kaggle leaderboard.</b> These metrics come from a holdout slice of Kaggle's labelled train.csv. "
     "The Kaggle leaderboard scores accuracy on test.csv, whose labels are private — your leaderboard score will differ and can only be obtained by submitting.", "emoji_events")
