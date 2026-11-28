"""TITANIC ML LAB — Streamlit entry point.

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.ui import state  # noqa: E402
from src.ui.theme import inject_theme  # noqa: E402

st.set_page_config(page_title="Titanic ML Lab", page_icon="🚢", layout="wide", initial_sidebar_state="expanded")
inject_theme()

P = "app_pages/"
pages = {
    "Overview": [
        st.Page(P + "0_Home.py", title="Home Dashboard", icon=":material/space_dashboard:", default=True),
        st.Page(P + "13_Ship_Explorer_3D.py", title="3D Ship Explorer", icon=":material/directions_boat:"),
    ],
    "Data": [
        st.Page(P + "1_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/table_view:"),
        st.Page(P + "2_Exploratory_Analysis.py", title="Exploratory Analysis", icon=":material/insights:"),
        st.Page(P + "3_Data_Preprocessing.py", title="Data Preprocessing", icon=":material/tune:"),
        st.Page(P + "4_Feature_Engineering.py", title="Feature Engineering", icon=":material/auto_fix_high:"),
    ],
    "Modeling": [
        st.Page(P + "5_Model_Training.py", title="Model Training", icon=":material/model_training:"),
        st.Page(P + "6_Model_Comparison.py", title="Model Comparison", icon=":material/leaderboard:"),
        st.Page(P + "7_Model_Evaluation.py", title="Model Evaluation", icon=":material/fact_check:"),
        st.Page(P + "8_Explainable_AI.py", title="Explainable AI", icon=":material/psychology:"),
    ],
    "Predict": [
        st.Page(P + "9_Passenger_Prediction.py", title="Passenger Prediction", icon=":material/person_search:"),
        st.Page(P + "14_Beat_the_Model.py", title="Beat the Model", icon=":material/sports_esports:"),
        st.Page(P + "10_Batch_Prediction.py", title="Batch Prediction", icon=":material/upload_file:"),
        st.Page(P + "11_Kaggle_Submission.py", title="Kaggle Submission", icon=":material/emoji_events:"),
    ],
    "About": [
        st.Page(P + "12_About.py", title="About the Project", icon=":material/info:"),
    ],
}
nav = st.navigation(pages, position="sidebar", expanded=True)
state.ensure_dataset_present()
state.ensure_models_ready()

# ── global sidebar controls ──────────────────────────────────────────
st.logo(str(ROOT / "assets" / "logo.svg"), size="large", icon_image=str(ROOT / "assets" / "logo_icon.svg"))
with st.sidebar:
    if state.demo_mode():
        st.markdown('<div class="tml-status"><b>Public demo</b> · you can train and experiment freely; '
                    'shared models and data are never overwritten.</div>', unsafe_allow_html=True)
    st.toggle("Learn mode", value=st.session_state.get("learn_mode", True), key="learn_mode",
              help="Show plain-language explanations of machine-learning concepts on every page.")
    reg = state.get_registry() if state.data_ready() else {}
    if reg:
        from src.models import MODEL_ORDER

        keys = [k for k in MODEL_ORDER if k in reg]
        best = state.best_key()
        if st.session_state.get("active_model") not in keys:
            st.session_state["active_model"] = best
        st.selectbox("Active model", keys, key="active_model",
                     format_func=lambda k: f"{state.model_name(k)}{'  ★' if k == best else ''}",
                     help="Used by prediction, evaluation and explainability pages. ★ = best by cross-validated ROC-AUC.")
        cv = (reg[st.session_state['active_model']].get("cv") or {})
        ho = (reg[st.session_state['active_model']].get("holdout") or {})
        st.markdown(f"""<div class="tml-status">
            <b>{len(reg)}</b> models trained · best <b>{state.model_name(best)}</b><br>
            Active · CV ROC-AUC <b>{(cv.get('roc_auc_mean') or 0):.3f}</b> · holdout acc <b>{(ho.get('accuracy') or 0):.3f}</b>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="tml-status">No trained models yet — open <b>Model Training</b> or run <code>python scripts/train_all.py</code>.</div>',
                    unsafe_allow_html=True)

nav.run()
