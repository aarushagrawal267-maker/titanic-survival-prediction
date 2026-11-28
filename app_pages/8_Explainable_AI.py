"""Explainable AI — SHAP, permutation importance, native importances, individual explanations, 3D landscape."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import explainability as xai
from src.models import MODEL_ORDER, SPECS
from src.preprocessing import friendly
from src.ui import charts, state
from src.ui.components import causation_note, kpi_tiles, learn, note, page_header, section
from src.ui.theme import BRASS, GOLD, PERISHED, SURVIVED, VIOLET

page_header("Modeling · 08", "Explainable AI",
            "Which features drive the predictions — globally, and for one passenger? SHAP values, permutation importance and the models' own internals.")
df = state.require_data()
key = state.require_models()
reg = state.get_registry()
keys = [k for k in MODEL_ORDER if k in reg]
key = st.selectbox("Model to explain", keys, index=keys.index(key), format_func=state.model_name, key="xai_model")
ev = state.get_evaluation(key)
shp = ev.get("shap")
pipe_dev = state.get_pipeline(key, "dev")

learn("What is SHAP?", "SHAP (SHapley Additive exPlanations) splits one prediction into contributions from each feature, using ideas from game theory. "
      "Starting from the average predicted survival probability (the <i>base value</i>), each feature pushes the estimate up or down; "
      "the pushes add up exactly to the model's prediction. Here contributions are in <b>percentage points of survival probability</b>.")
causation_note()

tabs = st.tabs(["Global importance", "SHAP summary", "Dependence", "Explain a passenger", "3D probability landscape", "Model internals"])

with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        if shp:
            imp = xai.shap_importance(shp)
            charts.show(charts.hbar(imp["label"], imp["mean_abs_shap"] * 100, "Mean |SHAP| — average impact on the estimate", color=SURVIVED, fmt=".1f",
                                    xlabel="percentage points"))
            st.caption(f"Method: {shp.get('method', 'SHAP')} on {len(shp['values'])} holdout passengers.")
        else:
            st.info("SHAP was not computed for this model (re-train with 'Compute SHAP' on).")
    with c2:
        pi = pd.DataFrame(ev["permutation_importance"])
        charts.show(charts.hbar(pi["label"], pi["importance_mean"], "Permutation importance (holdout ROC-AUC drop)", color=BRASS,
                                fmt=".3f", error=pi["importance_std"], xlabel="ROC-AUC decrease"))
        st.caption("Each raw input column is shuffled 10 times; bigger drops mean the model relies on it more. Name → title, Ticket → group size, Cabin → deck.")
    note("The two methods answer different questions: SHAP shows how much each <i>model feature</i> moves individual predictions; permutation "
         "importance shows how much <i>performance</i> depends on each raw input. Correlated inputs (Sex and Title) share credit, so neither should be read alone.", "info")

with tabs[1]:
    if shp:
        vals = np.asarray(shp["values"]) * 100
        feats = shp["features"]
        data = pd.DataFrame(shp["data"])
        order = np.argsort(np.abs(vals).mean(0))
        fig = go.Figure()
        rng = np.random.default_rng(0)
        for rank, j in enumerate(order):
            f = feats[j]
            col = data[f]
            if col.dtype == object:
                codes = col.astype("category").cat.codes.astype(float)
                cval = (codes - codes.min()) / max(codes.max() - codes.min(), 1)
                hover = col.astype(str)
            else:
                num = pd.to_numeric(col, errors="coerce")
                cval = num.rank(pct=True).fillna(0.5)
                hover = num.round(2).astype(str)
            fig.add_scatter(x=vals[:, j], y=rank + rng.uniform(-0.28, 0.28, len(vals)), mode="markers", showlegend=False,
                            marker=dict(size=6, color=cval, colorscale=[[0, VIOLET], [1, GOLD]], cmin=0, cmax=1, opacity=0.85,
                                        line=dict(color="#050b1a", width=0.5), showscale=(rank == len(order) - 1),
                                        colorbar=dict(title="feature value", tickvals=[0, 1], ticktext=["low", "high"], thickness=10, len=0.6)),
                            text=hover, hovertemplate=f"<b>{friendly(f)}</b> = %{{text}}<br>SHAP %{{x:+.1f}} pts<extra></extra>")
        fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
        fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[friendly(feats[j]) for j in order], showgrid=False)
        fig.update_xaxes(title="SHAP value (percentage points of survival probability)")
        fig.update_layout(title="SHAP summary (beeswarm) — one dot per passenger per feature", height=620)
        charts.show(fig)
        st.caption("Dots right of zero pushed the estimate up; left pushed it down. Colour shows the feature's value (violet = low, gold = high; "
                   "for categories, alphabetical order). E.g. for Sex, 'female' (low) sits on the right.")
    else:
        st.info("No SHAP values stored for this model.")

with tabs[2]:
    if shp:
        feats = shp["features"]
        imp_order = [feats[j] for j in np.argsort(-np.abs(np.asarray(shp["values"])).mean(0))]
        f = st.selectbox("Feature", imp_order, format_func=friendly, key="dep_feat")
        j = feats.index(f)
        data = pd.DataFrame(shp["data"])
        yv = np.asarray(shp["values"])[:, j] * 100
        outcome = np.asarray(shp["y_true"])
        fig = go.Figure()
        for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
            m = outcome == val
            fig.add_scatter(x=data[f][m], y=yv[m], mode="markers", name=f"Actually: {name}", marker=dict(size=9, color=color, opacity=0.8, line=dict(color="#050b1a", width=1)),
                            hovertemplate=f"{friendly(f)} = %{{x}}<br>SHAP %{{y:+.1f}} pts<extra></extra>")
        fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
        fig.update_layout(title=f"How {friendly(f)} affects the estimate", height=440)
        fig.update_xaxes(title=friendly(f))
        fig.update_yaxes(title="SHAP value (pts)")
        charts.show(fig)
    else:
        st.info("No SHAP values stored for this model.")

with tabs[3]:
    if shp:
        ids = shp["passenger_ids"]
        lookup = df.set_index("PassengerId")
        options = [i for i in ids if i in lookup.index]
        pid = st.selectbox("Holdout passenger (real record, unseen during training)", options,
                           format_func=lambda i: f"#{i} · {lookup.loc[i, 'Name']} · class {lookup.loc[i, 'Pclass']} · {lookup.loc[i, 'Sex']}", key="xai_pid")
        row = lookup.loc[[pid]].reset_index()
        X_dev = df.sample(frac=1.0, random_state=0)
        contrib = xai.explain_one(pipe_dev, row.drop(columns=["Survived"]), df.drop(columns=["Survived"]).sample(200, random_state=1))
        prob = float(pipe_dev.predict_proba(row.drop(columns=["Survived"]))[0, 1])
        actual = int(row["Survived"].iloc[0])
        kpi_tiles([
            {"label": "Model estimate", "value": f"{prob:.0%}", "icon": "query_stats", "tone": "survived" if prob >= 0.5 else "perished",
             "sub": "Predicted: survived" if prob >= 0.5 else "Predicted: did not survive"},
            {"label": "Actual outcome", "value": "Survived" if actual else "Did not survive", "icon": "history_edu", "tone": "survived" if actual else "perished"},
            {"label": "Base value", "value": f"{contrib.attrs['base_value']:.0%}", "icon": "horizontal_rule", "sub": "average estimate", "accent": BRASS},
            {"label": "Model was", "value": "Right" if int(prob >= 0.5) == actual else "Wrong", "icon": "check_circle" if int(prob >= 0.5) == actual else "cancel",
             "tone": "survived" if int(prob >= 0.5) == actual else "perished"},
        ])
        w1, w2 = st.columns([1.4, 1])
        with w1:
            charts.show(charts.waterfall(contrib.attrs["base_value"], contrib, prob, "From the average passenger to this passenger"))
        with w2:
            nar = xai.narrative(contrib)
            st.markdown("**Associated with a higher estimate**")
            st.markdown("\n".join(f"- {t}" for t in nar["higher"]) or "- none")
            st.markdown("**Associated with a lower estimate**")
            st.markdown("\n".join(f"- {t}" for t in nar["lower"]) or "- none")
            st.dataframe(row.drop(columns=["PassengerId"]).T.rename(columns={0: "value"}).astype(str), width="stretch")
    else:
        st.info("No SHAP values stored for this model.")

with tabs[4]:
    learn("Probability landscape", "The surface shows the model's survival estimate for every combination of age and fare, holding sex, class, port and "
          "family fixed. Smooth hills = Logistic Regression; staircases = trees. Drag to rotate.")
    a, b, c, d = st.columns(4)
    land_key = a.selectbox("Model", keys, index=keys.index(key), format_func=state.model_name, key="land_model")
    sex = b.selectbox("Sex", ["female", "male"], key="land_sex")
    pcl = c.selectbox("Class", [1, 2, 3], index=2, key="land_cls")
    fam = d.selectbox("Family aboard", ["Alone", "With spouse", "Parent + 2 kids"], key="land_fam")
    sib, par = {"Alone": (0, 0), "With spouse": (1, 0), "Parent + 2 kids": (1, 2)}[fam]

    @st.cache_data(show_spinner="Evaluating the model on a 40 × 40 grid…")
    def landscape(k, sex, pcl, sib, par, mtime):
        ages = np.linspace(1, 80, 40)
        fares = np.geomspace(5, 300, 40)
        A, F = np.meshgrid(ages, fares)
        grid = pd.DataFrame({"PassengerId": 0, "Pclass": pcl, "Name": np.nan, "Sex": sex, "Age": A.ravel(), "SibSp": sib, "Parch": par,
                             "Ticket": np.nan, "Fare": F.ravel(), "Cabin": np.nan, "Embarked": "S"})
        Z = state.get_pipeline(k, "full").predict_proba(grid)[:, 1].reshape(A.shape)
        return ages, fares, Z

    ages, fares, Z = landscape(land_key, sex, pcl, sib, par, state.registry_mtime())
    charts.show(charts.surface3d(ages, fares, Z, f"{state.model_name(land_key)} · {sex}, class {pcl}, {fam.lower()}, Southampton"))
    st.caption("Titles are inferred from sex and age (e.g. a boy under 13 → 'Master'), so jumps at age 13/18 reflect the title rule, not just age.")

with tabs[5]:
    nat = ev.get("native_importance")
    if nat:
        nd = pd.DataFrame(nat)
        charts.show(charts.hbar(nd["label"], nd["value"], f"{ev.get('native_kind', 'Built-in importance')} (summed over one-hot columns)", color=VIOLET, fmt=".3f"))
    coefs = xai.coefficients(pipe_dev)
    if coefs is not None:
        st.markdown("**Logistic Regression coefficients** — log-odds change per unit of the (standardised) feature; odds ratio = e^coef.")
        st.dataframe(coefs[["level", "coefficient", "odds_ratio"]].round(3), hide_index=True, width="stretch", height=420)
    if not nat and coefs is None:
        st.info(f"{SPECS[key].name} has no built-in importance — use SHAP and permutation importance instead.")
