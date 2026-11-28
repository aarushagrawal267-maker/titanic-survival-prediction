"""Kaggle Submission — generate and validate submission.csv for the Titanic competition."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.models import MODEL_ORDER
from src.submission import make_submission, save_submission, validate_submission
from src.ui import charts, state
from src.ui.components import kpi_tiles, learn, note, page_header, section
from src.ui.theme import BRASS, PERISHED, SURVIVED, VIOLET

page_header("Predict · 12", "Kaggle Submission",
            "Score every passenger in Kaggle's competition <code>test.csv</code> and produce the official two-column file "
            "<code>PassengerId,Survived</code>, ready to upload at kaggle.com/competitions/titanic.")
df = state.require_data()
key = state.require_models()
reg = state.get_registry()
try:
    test = state.get_test()
except FileNotFoundError:
    st.error("test.csv not found in data/raw — download it on the Dataset Explorer page.")
    st.stop()

note("<b>Kaggle's test labels are not public.</b> The real competition score can only be obtained by submitting the file on Kaggle. "
     "Validation scores in this app are estimates from the labelled training data and will differ from the leaderboard.", "emoji_events")

keys = [k for k in MODEL_ORDER if k in reg]
c1, c2 = st.columns([1.2, 1])
with c1:
    sel = st.selectbox("Model", keys, index=keys.index(state.best_key()), format_func=lambda k: f"{state.model_name(k)} — CV ROC-AUC {reg[k]['cv']['roc_auc_mean']:.3f}"
                       f"{'  ★ best' if k == state.best_key() else ''}", key="kg_model")
    st.caption("Uses the model refit on all 891 labelled passengers with the tuned hyper-parameters.")
    go_btn = st.button("Generate submission.csv", type="primary", icon=":material/emoji_events:", width="stretch")
with c2:
    learn("The competition workflow", "1) Train on train.csv · 2) predict test.csv · 3) upload a CSV with exactly 418 rows and the columns "
          "PassengerId and Survived (0/1) · 4) Kaggle reports accuracy on hidden labels.")

if go_btn:
    sub = make_submission(state.get_pipeline(sel, "full"), test)
    errs = validate_submission(sub, test)
    st.session_state["kg_sub"] = (sel, sub, errs)
    if not errs:
        save_submission(sub)

if "kg_sub" in st.session_state:
    sel_k, sub, errs = st.session_state["kg_sub"]
    section("Validation checks", "01", "rule")
    checks = [
        ("Columns are exactly PassengerId, Survived", list(sub.columns) == ["PassengerId", "Survived"]),
        (f"Row count matches test.csv ({len(test)})", len(sub) == len(test)),
        ("Every PassengerId preserved, same order, no duplicates", sub["PassengerId"].tolist() == test["PassengerId"].tolist()),
        ("Survived contains only integers 0 or 1", sub["Survived"].isin([0, 1]).all() and pd.api.types.is_integer_dtype(sub["Survived"])),
        ("No missing values", not sub.isna().any().any()),
    ]
    st.markdown("\n".join(f"- {':green[✔]' if ok else ':red[✘]'} {label}" for label, ok in checks))
    if errs:
        for e in errs:
            st.error(e)
        st.stop()
    section("Submission summary", "02", "summarize")
    kpi_tiles([
        {"label": "Rows", "count_to": len(sub), "icon": "table_rows", "accent": BRASS},
        {"label": "Predicted survivors", "count_to": int(sub["Survived"].sum()), "icon": "health_and_safety", "accent": SURVIVED},
        {"label": "Predicted survival rate", "value": f"{sub['Survived'].mean():.1%}", "icon": "percent", "sub": f"train: {df['Survived'].mean():.1%}", "accent": SURVIVED},
        {"label": "Model", "value": state.model_name(sel_k), "icon": "model_training", "accent": VIOLET},
    ])
    s1, s2 = st.columns([1, 1.3])
    with s1:
        st.dataframe(sub, hide_index=True, width="stretch", height=360)
    with s2:
        merged = test.assign(Survived=sub["Survived"].values)
        g = merged.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
        fig = go.Figure()
        for sex, color in (("female", SURVIVED), ("male", PERISHED)):
            gg = g[g["Sex"] == sex]
            fig.add_bar(name=sex.capitalize(), x=[f"Class {c}" for c in gg["Pclass"]], y=gg["Survived"], marker=dict(color=color, cornerradius=4),
                        hovertemplate=f"{sex} · %{{x}}: %{{y:.0%}} predicted to survive<extra></extra>")
        fig.update_layout(barmode="group", title="Predicted survival rate in test.csv by class and sex", height=360)
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        charts.show(fig)
    path = config.PROJECT_ROOT / config.get("paths", "submission_file")
    st.download_button("Download submission.csv", sub.to_csv(index=False), "submission.csv", "text/csv", type="primary", icon=":material/download:")
    st.caption(f"Also saved to `{path.relative_to(config.PROJECT_ROOT)}`. CLI equivalent: `python scripts/make_submission.py --model {sel_k}`")
