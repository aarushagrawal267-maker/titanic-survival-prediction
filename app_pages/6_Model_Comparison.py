"""Model Comparison — leaderboard, charts, CV stability, training time, algorithm notes, run history."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import METRIC_EXPLANATIONS, METRIC_LABELS, METRICS
from src.models import MODEL_ORDER, SPECS
from src import model_manager
from src.ui import charts, state
from src.ui.components import learn, note, page_header, pill, section
from src.ui.theme import BRASS, MUTED, PERISHED, SURVIVED, VIOLET

page_header("Modeling · 06", "Model Comparison",
            "All trained algorithms evaluated with the identical methodology: same split, same 5 stratified folds, same metrics, same holdout set.")
state.require_data()
state.require_models()
reg = state.get_registry()
best = state.best_key()
keys = [k for k in MODEL_ORDER if k in reg]

rows = []
for k in keys:
    m = reg[k]
    rows.append({
        "Model": state.model_name(k), "key": k,
        **{f"CV {METRIC_LABELS[x]}": m["cv"][f"{x}_mean"] for x in METRICS},
        "CV ROC-AUC std": m["cv"]["roc_auc_std"],
        **{f"Holdout {METRIC_LABELS[x]}": m["holdout"][x] for x in METRICS},
        "Fit time (s)": m.get("train_time_s"), "Features": m.get("feature_set"), "Scaler": m.get("scaler"),
        "Tuned": "yes" if m.get("tuned") else "manual",
    })
tbl = pd.DataFrame(rows).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
tbl.insert(0, "Rank", np.arange(1, len(tbl) + 1))

section("Leaderboard", "01", "leaderboard")
st.markdown(f"Best model: {pill('★ ' + state.model_name(best), 'brass')} &nbsp; selected by **mean cross-validated ROC-AUC** "
            "(tie-break: CV F1, then lower fold variance).", unsafe_allow_html=True)
show_cols = ["Rank", "Model"] + [f"CV {METRIC_LABELS[x]}" for x in METRICS] + ["CV ROC-AUC std"] + [f"Holdout {METRIC_LABELS[x]}" for x in METRICS] + ["Fit time (s)", "Features", "Scaler", "Tuned"]
cfg = {c: st.column_config.NumberColumn(format="%.3f") for c in show_cols if c.startswith(("CV", "Holdout"))}
cfg["CV ROC-AUC"] = st.column_config.ProgressColumn("CV ROC-AUC", min_value=0.5, max_value=1.0, format="%.3f")
cfg["Fit time (s)"] = st.column_config.NumberColumn(format="%.2f")
st.dataframe(tbl[show_cols], hide_index=True, width="stretch", column_config=cfg)
st.download_button("Download comparison (CSV)", tbl[show_cols].to_csv(index=False), "model_comparison.csv", icon=":material/download:")
learn("Why not just pick the highest accuracy?", "Only 38% of passengers survived, so a model that predicts 'did not survive' for everyone already gets 62% accuracy. "
      "ROC-AUC measures how well a model <i>ranks</i> survivors above non-survivors across all thresholds, and F1 balances precision and recall. "
      "We also prefer models whose scores are stable across folds (low std).")

section("Holdout metrics", "02", "bar_chart")
view = st.segmented_control("Scores", ["Holdout", "Cross-validation"], default="Holdout", key="cmp_view") or "Holdout"
prefix = "Holdout" if view == "Holdout" else "CV"
mdf = tbl[["Model"] + [f"{prefix} {METRIC_LABELS[x]}" for x in METRICS]].rename(columns={f"{prefix} {METRIC_LABELS[x]}": x for x in METRICS})
charts.show(charts.metric_bars(mdf, METRICS, METRIC_LABELS, f"{view} performance by model"))
with st.expander("What do these metrics mean?"):
    for m in METRICS:
        st.markdown(f"**{METRIC_LABELS[m]}** — {METRIC_EXPLANATIONS[m]}")

c1, c2 = st.columns(2)
with c1:
    section("Cross-validation stability", "03", "stacked_line_chart")
    folds = {state.model_name(k): reg[k]["cv"]["folds"]["roc_auc"] for k in tbl["key"]}
    charts.show(charts.cv_box(folds, "ROC-AUC on each of the 5 folds"))
with c2:
    section("Generalisation check", "04", "compare_arrows")
    fig = go.Figure()
    for _, r in tbl.iterrows():
        is_best = r["key"] == best
        fig.add_scatter(x=[r["CV ROC-AUC"]], y=[r["Holdout ROC-AUC"]], mode="markers+text", text=[r["Model"]], textposition="top center",
                        textfont=dict(color="#eaf0fa" if is_best else MUTED, size=11), showlegend=False,
                        marker=dict(size=14 if is_best else 11, color=BRASS if is_best else SURVIVED, line=dict(color="#050b1a", width=2)),
                        hovertemplate=f"<b>{r['Model']}</b><br>CV %{{x:.3f}} · holdout %{{y:.3f}}<extra></extra>")
    lo = min(tbl["CV ROC-AUC"].min(), tbl["Holdout ROC-AUC"].min()) - 0.02
    hi = max(tbl["CV ROC-AUC"].max(), tbl["Holdout ROC-AUC"].max()) + 0.02
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dot", width=1), showlegend=False, hoverinfo="skip")
    fig.update_layout(title="CV ROC-AUC vs holdout ROC-AUC", height=400)
    fig.update_xaxes(title="Cross-validated ROC-AUC", range=[lo, hi])
    fig.update_yaxes(title="Holdout ROC-AUC", range=[lo, hi])
    charts.show(fig)
    st.caption("Points below the dotted line did worse on the holdout than in CV. With only 179 holdout passengers, differences of ~0.02 are within noise.")

section("ROC curves on the holdout set", "05", "show_chart")
hl = st.selectbox("Highlight", ["All"] + [state.model_name(k) for k in tbl["key"]], key="cmp_hl")
curves = {}
for k in tbl["key"]:
    ev = state.get_evaluation(k)
    curves[state.model_name(k)] = {"fpr": ev["roc"]["fpr"], "tpr": ev["roc"]["tpr"], "auc": ev["metrics"]["roc_auc"]}
if hl == "All":
    note("Top 3 models by CV ROC-AUC in colour, the rest muted — pick a model in <b>Highlight</b> to focus on it.", "visibility")
    charts.show(charts.roc_curves(curves, highlight=list(curves)[:3]))
else:
    charts.show(charts.roc_curves(curves, highlight=hl))

section("Training time", "06", "timer")
tt = tbl.sort_values("Fit time (s)", ascending=False)
charts.show(charts.hbar(tt["Model"], tt["Fit time (s)"], "Time to fit the final model on 712 passengers (seconds)", color=VIOLET, fmt=".2f", xlabel="seconds", height=330))

section("Algorithms at a glance", "07", "menu_book")
cols = st.columns(3)
for i, k in enumerate(keys):
    s = SPECS[k]
    with cols[i % 3], st.container(border=True):
        st.markdown(f"**{s.name}** {'★' if k == best else ''}")
        st.caption(s.purpose)
        st.markdown("✔ " + "  \n✔ ".join(s.pros))
        st.markdown("✘ " + "  \n✘ ".join(s.cons))
        with st.popover("Tuned hyper-parameters"):
            st.json({p: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for p, v in (reg[k].get("best_params") or {}).items()})

section("Run history & monitoring", "08", "monitor_heart")
hist = model_manager.load_run_history()
if hist:
    h = pd.DataFrame(hist)
    h["time"] = pd.to_datetime(h["time"])
    h["Model"] = h["key"].map(state.model_name)
    h["holdout_accuracy"] = h["holdout"].map(lambda d: (d or {}).get("accuracy"))
    fig = go.Figure()
    for i, (name, g) in enumerate(h.sort_values("time").groupby("Model")):
        fig.add_scatter(x=g["time"], y=g["cv_roc_auc"], mode="lines+markers", name=name, line=dict(width=2),
                        hovertemplate=f"{name}<br>%{{x}}<br>CV ROC-AUC %{{y:.3f}}<extra></extra>")
    fig.update_layout(title="CV ROC-AUC of every training run", height=340, legend=dict(orientation="h", y=-0.25, x=0, xanchor="left", yanchor="top"))
    charts.show(fig)
    st.dataframe(h.sort_values("time", ascending=False)[["time", "Model", "source", "tuned", "cv_roc_auc", "holdout_accuracy", "train_time_s", "feature_set", "scaler"]],
                 hide_index=True, width="stretch", column_config={"cv_roc_auc": st.column_config.NumberColumn(format="%.3f"),
                                                                           "holdout_accuracy": st.column_config.NumberColumn(format="%.3f"),
                                                                           "train_time_s": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Every training run (CLI or app) is appended to `models/run_history.jsonl`, so you can track how changes affect performance over time.")
note("These are <b>validation</b> scores on labelled Kaggle training data. Kaggle leaderboard scores are computed on hidden labels for test.csv and "
     "will differ — see the Kaggle Submission page.", "info")
