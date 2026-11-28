"""Feature Engineering — what each feature means, whether it helps, and ablation evidence."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_selection import mutual_info_classif

from src import eda
from src.feature_engineering import (AGE_LABELS, ENGINEERED_FEATURES, FEATURE_DESCRIPTIONS, TITLE_GROUPS,
                                     TitanicFeatureEngineer, add_basic_features)
from src.preprocessing import friendly
from src.train import load_experiments
from src.ui import charts, state
from src.ui.components import card, kpi_tiles, learn, note, page_header, section
from src.ui.theme import BRASS, SURVIVED, VIOLET

page_header("Data · 04", "Feature Engineering",
            "Ten new features derived from names, tickets, cabins and family counts — and honest experiments showing which of them actually help.")
df = state.require_data()
exp = load_experiments()
fe = TitanicFeatureEngineer().fit(df)
E = fe.transform(df)
E["Survived"] = df["Survived"].values
basic = add_basic_features(df)
E["Title"] = basic["Title"].values

learn("Feature engineering", "Models only see numbers. Feature engineering turns raw text and counts into signals a model can use — e.g. "
      "the word <i>Master</i> in a name tells us the passenger is a young boy, even when the age is missing. Good features encode domain knowledge; "
      "bad ones add noise, so we test them.")

section("Feature gallery", "01", "auto_fix_high")
sel = st.pills("Inspect a feature", ["FamilySize", "IsAlone", "Title", "TitleGroup", "FarePerPerson", "AgeGroup", "FamilyCategory", "CabinKnown", "Deck", "TicketGroupSize"],
               default="TitleGroup", format_func=lambda f: friendly(f) if f != "Title" else "Raw title")
sel = sel or "TitleGroup"
c1, c2 = st.columns([1, 1.6])
with c1:
    card(friendly(sel) if sel != "Title" else "Raw title", FEATURE_DESCRIPTIONS.get(sel, ""), "auto_fix_high")
    s = E[sel]
    if sel in ("FarePerPerson",):
        stats = s.describe()[["mean", "50%", "min", "max"]].round(2)
        st.dataframe(stats.rename("value").to_frame(), width="stretch")
    else:
        vc = s.value_counts().rename("passengers").to_frame()
        st.dataframe(vc, width="stretch", height=min(360, 40 + 35 * len(vc)))
with c2:
    if sel == "FarePerPerson":
        tmp = E.assign(bucket=pd.qcut(E["FarePerPerson"].fillna(E["FarePerPerson"].median()), 5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (priciest)"]).astype(str))
        g = eda.survival_by(tmp, "bucket")
        col_, label = "bucket", "Fare per person quintile"
    elif sel == "TicketGroupSize":
        tmp = E.assign(bucket=E["TicketGroupSize"].clip(upper=6).astype(int).astype(str).replace({"6": "6+"}))
        g = eda.survival_by(tmp, "bucket")
        col_, label = "bucket", "Passengers sharing the ticket"
    elif sel == "Title":
        g = eda.survival_by(E, "Title")
        g = g[g["passengers"] >= 2].sort_values("passengers", ascending=False)
        col_, label = "Title", "Raw title (≥ 2 passengers)"
    else:
        order = AGE_LABELS + ["Unknown"] if sel == "AgeGroup" else None
        g = eda.survival_by(E, sel, order=order)
        if sel == "FamilySize":
            g = g.sort_values("FamilySize")
        col_, label = sel, friendly(sel)
    charts.show(charts.survival_rate_bar(g, col_, f"Survival rate by {label.lower()}", label, overall=float(df["Survived"].mean())))

st.markdown("**Title extraction → grouping**")
tmap = basic[["Name", "Title", "TitleGroup"]].head(8)
t1, t2 = st.columns([1.4, 1])
t1.dataframe(tmap, hide_index=True, width="stretch")
t2.dataframe(pd.DataFrame({"title": list(TITLE_GROUPS), "group": list(TITLE_GROUPS.values())}), hide_index=True, width="stretch", height=300)

section("Signal strength (mutual information)", "02", "network_check")
X_mi = E[[c for c in E.columns if c not in ("Survived", "Title")]].copy()
disc = []
for c in X_mi.columns:
    if X_mi[c].dtype == object:
        X_mi[c] = X_mi[c].astype("category").cat.codes
        disc.append(True)
    else:
        X_mi[c] = X_mi[c].fillna(X_mi[c].median())
        disc.append(c in ("Pclass", "IsAlone", "CabinKnown", "AgeKnown", "SibSp", "Parch", "FamilySize"))
mi = mutual_info_classif(X_mi, E["Survived"], discrete_features=np.array(disc), random_state=42)
mi_df = pd.DataFrame({"feature": X_mi.columns, "mi": mi}).sort_values("mi", ascending=False)
mi_df["kind"] = np.where(mi_df["feature"].isin(ENGINEERED_FEATURES), "Engineered", "Original")
fig = go.Figure()
for kind, color in (("Original", "rgba(180,194,217,0.55)"), ("Engineered", BRASS)):
    sub = mi_df[mi_df["kind"] == kind]
    fig.add_bar(y=[friendly(f) for f in sub["feature"]], x=sub["mi"], orientation="h", name=kind, marker=dict(color=color, cornerradius=4),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>")
fig.update_layout(title="Mutual information with survival (higher = more information)", height=560, barmode="overlay")
fig.update_yaxes(categoryorder="total ascending")
fig.update_xaxes(title="mutual information (nats)")
charts.show(fig)
note("Mutual information measures how much knowing a feature reduces uncertainty about survival — on its own. It ignores redundancy: "
     "<i>Title</i> and <i>Sex</i> both score high but carry overlapping information. That is why the ablation below matters.", "info")

section("Do engineered features improve the models?", "03", "science")
if exp.get("feature_sets"):
    fs = pd.DataFrame(exp["feature_sets"]["rows"])
    fig = go.Figure()
    for name, color in (("base", "rgba(180,194,217,0.55)"), ("engineered", SURVIVED)):
        sub = fs[fs["feature_set"] == name]
        fig.add_bar(name="Base (7)" if name == "base" else "Engineered (17)", x=sub["model"], y=sub["roc_auc"],
                    error_y=dict(type="data", array=sub["roc_auc_std"], color="rgba(255,255,255,0.35)", thickness=1),
                    marker=dict(color=color, cornerradius=3), hovertemplate="%{x}<br>CV ROC-AUC %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", title="Base vs engineered features · CV ROC-AUC", height=400)
    fig.update_yaxes(range=[max(0.5, fs["roc_auc"].min() - 0.05), min(1.0, fs["roc_auc"].max() + 0.03)], title="CV ROC-AUC")
    charts.show(fig)
    piv = fs.pivot_table(index="model", columns="feature_set", values="roc_auc")
    piv["gain"] = piv["engineered"] - piv["base"]
    kpi_tiles([{"label": m, "value": f"{g * 100:+.1f} pts", "sub": f"{piv.loc[m, 'base']:.3f} → {piv.loc[m, 'engineered']:.3f}",
                "icon": "trending_up" if g > 0 else "trending_down", "tone": "survived" if g > 0 else "perished"} for m, g in piv["gain"].items()])

    section("Ablation — remove one feature at a time", "04", "content_cut")
    ab = pd.DataFrame(exp["ablation"]["rows"])
    ab = ab[ab["dropped"] != "(none — all features)"]
    a1, a2 = st.columns(2)
    for col, model in zip((a1, a2), ab["model"].unique()):
        sub = ab[ab["model"] == model].sort_values("delta")
        with col:
            fig = charts.signed_hbar([friendly(f) for f in sub["dropped"]], list(-sub["delta"]), f"{model} — AUC lost without each feature",
                                     xlabel="Δ CV ROC-AUC × 100 (positive = feature helps)", fmt="+.2f")
            charts.show(fig, key=f"abl_{model}")
    helps = ab.groupby("dropped")["delta"].mean().sort_values()
    useful = [friendly(f) for f, d in helps.items() if d < -0.002]
    neutral = [friendly(f) for f, d in helps.items() if abs(d) <= 0.002]
    harmful = [friendly(f) for f, d in helps.items() if d > 0.002]
    note(f"<b>Findings (average over both models):</b> clearly useful — {', '.join(useful) or 'none'}; roughly neutral — {', '.join(neutral) or 'none'}; "
         f"removal slightly <i>improved</i> CV — {', '.join(harmful) or 'none'}. Differences of a few thousandths are within fold-to-fold noise "
         f"(CV std ≈ 0.02), and many engineered features are redundant with each other (FamilySize / IsAlone / FamilyCategory), "
         "so removing one rarely matters on its own. Engineered features are not assumed to help — this is the evidence.", "fact_check")
else:
    st.info("Run `python scripts/train_all.py` to execute the feature-set and ablation experiments.")

section("Ticket groups without leakage", "05", "confirmation_number")
note("<code>TicketGroupSize</code> counts how many passengers share a ticket number. The counts are learned in <code>fit()</code> from the training "
     "fold only (validation rows never contribute), use no survival labels, and fall back to <code>FamilySize</code> when a ticket is missing "
     "(e.g. on the prediction form). It captures travelling groups that are not families — friends, servants, companions.", "verified_user")
grp = E.assign(bucket=E["TicketGroupSize"].clip(upper=5).astype(int))
st.dataframe(pd.crosstab(grp["bucket"].rename("Ticket group size (5 = 5+)"), basic["IsAlone"].rename("IsAlone (family)")), width="stretch")
