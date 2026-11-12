"""About the project."""
import platform

import streamlit as st

from src.models import XGBOOST_AVAILABLE
from src.explainability import SHAP_AVAILABLE
from src.ui import state
from src.ui.components import card, note, page_header, section

page_header("About", "About the Project",
            "Titanic ML Lab is an educational, end-to-end machine-learning application: from raw historical records to tuned, explained and "
            "deployed models — all running locally on your computer.")

section("The problem", "01", "help")
c1, c2 = st.columns(2)
with c1:
    card("Binary classification", "Predict whether a passenger survived (1) or did not survive (0) the sinking of RMS Titanic on 15 April 1912, "
         "from information such as ticket class, sex, age, family aboard, fare and port of embarkation.", "target")
with c2:
    card("The dataset", "Kaggle's <i>Titanic — Machine Learning from Disaster</i> competition: <code>train.csv</code> (891 passengers with known outcomes) and "
         "<code>test.csv</code> (418 passengers whose outcomes are hidden for the leaderboard). Files are verified against their SHA-256 checksums.", "dataset")

section("How it works", "02", "route")
st.markdown("""
1. **Acquisition & validation** — local files → Kaggle API → checksum-verified mirror → manual upload. Schema, types, ranges, duplicates and target checks.
2. **EDA** — survival by sex, class, age, port, family and fare; distributions; correlations; missing-data analysis; automatic insights computed from the data.
3. **Preprocessing** — a scikit-learn `Pipeline` with a custom feature engineer and a `ColumnTransformer` (median / most-frequent imputation, one-hot encoding, optional scaling). Fitted inside each CV fold — no leakage.
4. **Feature engineering** — FamilySize, IsAlone, Title/TitleGroup, FarePerPerson, AgeGroup, FamilyCategory, CabinKnown, Deck, TicketGroupSize, AgeKnown — each tested with ablation experiments.
5. **Modelling** — Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting and XGBoost; stratified 80/20 split, 5-fold StratifiedKFold CV, RandomizedSearchCV.
6. **Selection & evaluation** — best model by mean CV ROC-AUC (never accuracy alone, never the holdout); final metrics on an untouched holdout; ROC, PR, calibration, threshold analysis.
7. **Explainability** — SHAP (permutation explainer, probability units), permutation importance, native importances, per-passenger explanations.
8. **Prediction** — individual (with what-if and nearest real passengers), batch CSV (with drift check) and the Kaggle submission file.
""")

section("Limitations", "03", "report")
st.markdown("""
- **Small dataset** — 891 labelled passengers; holdout metrics on 179 people carry roughly ±3 percentage points of uncertainty.
- **Historical, not causal** — models learn associations in 1912 data (e.g. sex, class). They cannot explain *why* anyone survived, and should not be used to judge individuals.
- **Missing information** — cabin is unknown for 77% of passengers, age for 20%; crew are not in the dataset; cabin location on the ship is only approximated.
- **Not representative of everyone aboard** — the Kaggle split covers 1,309 of the ≈2,224 people aboard.
- **Leaderboard ≠ validation** — the Kaggle test labels are hidden; local scores are estimates.
- **3D ship** — a stylised, procedurally built model; class zones are approximate.
""")

section("Future improvements", "04", "rocket_launch")
st.markdown("""
- Stacking / soft-voting ensembles and probability calibration (isotonic) comparison.
- Nested cross-validation for fully unbiased tuning estimates.
- Fairness analysis across sex and class subgroups.
- Bayesian hyper-parameter optimisation (e.g. Optuna).
- Uncertainty intervals for individual predictions (bootstrap or conformal prediction).
- Containerised deployment (Docker) and CI running the test-suite.
""")

section("Technology", "05", "memory")
import numpy, pandas, plotly, sklearn  # noqa: E401,E402

cols = st.columns(4)
items = [("Python", platform.python_version()), ("pandas", pandas.__version__), ("NumPy", numpy.__version__), ("scikit-learn", sklearn.__version__),
         ("Streamlit", st.__version__), ("Plotly", plotly.__version__), ("XGBoost", "available" if XGBOOST_AVAILABLE else "not installed"),
         ("SHAP", "available" if SHAP_AVAILABLE else "fallback in use")]
for i, (k, v) in enumerate(items):
    cols[i % 4].metric(k, v)
st.caption("Also: SciPy, Matplotlib, Seaborn, Joblib, PyYAML, Pytest, and Three.js (vendored, MIT licence) for the 3D ship.")

section("Run it yourself", "06", "terminal")
st.code("pip install -r requirements.txt\npython scripts/download_data.py\npython scripts/train_all.py\nstreamlit run app.py\npytest", language="bash")
note("In memory of the more than 1,500 passengers and crew who lost their lives. This project uses their records to teach data science; "
     "we have tried to present them with care.", "favorite")
