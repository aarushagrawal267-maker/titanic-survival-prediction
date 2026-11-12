"""Exploratory Data Analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import eda
from src.feature_engineering import AGE_LABELS
from src.ui import charts, state
from src.ui.components import causation_note, insight_cards, kpi_tiles, learn, note, page_header, section
from src.ui.theme import AQUA, BRASS, GOLD, PERISHED, SURVIVED, VIOLET

page_header("Data · 02", "Exploratory Data Analysis",
            "Who survived, and which passenger characteristics were associated with survival? Every number and insight on this page is computed live from the data.")
df = state.require_data()
full = eda.enrich(df)

# global filter bar
with st.container(border=True):
    st.markdown("**Filter bar** — every chart below updates")
    a, b, c, d = st.columns(4)
    f_cls = a.multiselect("Class", [1, 2, 3], default=[1, 2, 3], key="eda_cls")
    f_sex = b.multiselect("Sex", ["female", "male"], default=["female", "male"], key="eda_sex")
    f_port = c.multiselect("Port", ["Southampton", "Cherbourg", "Queenstown", "Unknown"], default=["Southampton", "Cherbourg", "Queenstown", "Unknown"], key="eda_port")
    f_age = d.slider("Age", 0, 80, (0, 80), key="eda_age")
mask = full["Pclass"].isin(f_cls) & full["Sex"].isin(f_sex) & full["Port"].isin(f_port) & (full["Age"].between(*f_age) | full["Age"].isna())
e = full[mask]
if len(e) < 5:
    st.warning("Fewer than 5 passengers match — widen the filters.")
    st.stop()
raw = df[mask.values]
ov = eda.overview(raw)
overall = ov["survival_rate"]

tabs = st.tabs(["Overview", "Survival analysis", "Distributions", "Correlations", "Missing data", "Auto insights"])

with tabs[0]:
    kpi_tiles([
        {"label": "Passengers", "count_to": ov["passengers"], "icon": "groups", "accent": BRASS},
        {"label": "Survivors", "count_to": ov["survivors"], "icon": "health_and_safety", "accent": SURVIVED},
        {"label": "Non-survivors", "count_to": ov["non_survivors"], "icon": "person_off", "accent": PERISHED},
        {"label": "Survival rate", "value": f"{overall:.1%}", "icon": "percent", "accent": SURVIVED},
        {"label": "Raw features", "count_to": ov["features"], "icon": "view_column", "sub": "excl. target", "accent": VIOLET},
        {"label": "Missing cells", "count_to": ov["missing_cells"], "icon": "help", "sub": f"{ov['missing_pct']:.1f}% · {ov['duplicates']} duplicates", "accent": GOLD},
    ])
    o1, o2 = st.columns([1, 1.4])
    with o1:
        charts.show(charts.donut([ov["non_survivors"], ov["survivors"]], ["Did not survive", "Survived"], [PERISHED, SURVIVED],
                                 "Overall survival distribution", center=f"{overall:.0%}<br><span style='font-size:11px'>survived</span>", height=340))
    with o2:
        dt = pd.DataFrame({"column": list(ov["dtypes"].keys()), "dtype": list(ov["dtypes"].values()),
                           "non-null": [int(raw[c].notna().sum()) for c in ov["dtypes"]], "unique": [int(raw[c].nunique()) for c in ov["dtypes"]]})
        st.markdown("**Column data types**")
        st.dataframe(dt, hide_index=True, width="stretch", height=330)

with tabs[1]:
    learn("How to read these charts", "Left: the <b>survival rate</b> within each group (the dotted line is the overall rate). "
          "Right: the <b>number</b> of passengers in each group split by outcome — small groups have noisy rates.")
    groups = [
        ("Sex", "Sex", None, "By gender"),
        ("Class", "Class", ["1st class", "2nd class", "3rd class"], "By passenger class"),
        ("AgeGroup", "Age group", AGE_LABELS + ["Unknown"], "By age group"),
        ("Port", "Port of embarkation", ["Southampton", "Cherbourg", "Queenstown", "Unknown"], "By embarkation port"),
        ("FamilySize", "Family size", None, "By family size"),
        ("FareCategory", "Fare quartile", eda.FARE_ORDER, "By fare category"),
    ]
    for col, label, order, title in groups:
        g = eda.survival_by(e, col, order=order)
        if col == "FamilySize":
            g = g.sort_values("FamilySize")
        g = g[g["passengers"] > 0]
        st.markdown(f"##### {title}")
        l, r = st.columns(2)
        with l:
            charts.show(charts.survival_rate_bar(g, col, f"Survival rate {title.lower()}", label, overall=overall), key=f"rate_{col}")
        with r:
            charts.show(charts.outcome_stack(g, col, f"Passengers {title.lower()}", label), key=f"stack_{col}")
    st.markdown("##### Interaction: class × sex")
    pv = e.pivot_table(index="Sex", columns="Class", values="Survived", aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=[s.capitalize() for s in pv.index], zmin=0, zmax=1, colorscale=charts.DIVERGING,
                               text=[[f"{v:.0%}" for v in row] for row in pv.values], texttemplate="%{text}", textfont=dict(size=16, color="#fff"),
                               xgap=4, ygap=4, colorbar=dict(tickformat=".0%", thickness=10), hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>"))
    fig.update_layout(title="Survival rate by sex and class", height=300)
    charts.show(fig)
    causation_note()

with tabs[2]:
    d1, d2 = st.columns(2)
    with d1:
        charts.show(charts.histogram_by_outcome(e, "Age", "Age distribution by outcome", nbins=32))
    with d2:
        logx = st.toggle("Log scale for fare", value=True)
        ef = e.copy()
        if logx:
            ef = ef[ef["Fare"] > 0]
        charts.show(charts.histogram_by_outcome(ef, "Fare", "Fare distribution by outcome (£)", nbins=40, log_x=logx))
    d3, d4 = st.columns(2)
    with d3:
        fs = e["FamilySize"].value_counts().sort_index()
        fig = go.Figure(go.Bar(x=fs.index, y=fs.values, marker=dict(color=VIOLET, cornerradius=4), hovertemplate="Family size %{x}: %{y} passengers<extra></extra>"))
        fig.update_layout(title="Family size distribution", height=340)
        fig.update_xaxes(title="Family size (SibSp + Parch + 1)", dtick=1)
        fig.update_yaxes(title="Passengers")
        charts.show(fig)
    with d4:
        pc = e["Class"].value_counts().reindex(["1st class", "2nd class", "3rd class"]).fillna(0)
        charts.show(charts.donut(pc.values.tolist(), pc.index.tolist(), [GOLD, VIOLET, AQUA], "Passenger class distribution", center=f"{len(e)}<br><span style='font-size:11px'>passengers</span>", height=340))
    st.markdown("##### Age vs fare")
    fig = go.Figure()
    for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
        s = e[(e["Survived"] == val) & e["Age"].notna()]
        fig.add_scatter(x=s["Age"], y=s["Fare"], mode="markers", name=name, marker=dict(color=color, size=8, opacity=0.75, line=dict(color="#050b1a", width=1)),
                        text=s["Name"], hovertemplate="%{text}<br>Age %{x} · £%{y:.2f}<extra></extra>")
    fig.update_yaxes(type="log", title="Fare (£, log scale)")
    fig.update_xaxes(title="Age")
    fig.update_layout(title="Each dot is a passenger", height=430)
    charts.show(fig)
    st.markdown("##### Summary statistics")
    st.dataframe(raw[["Age", "SibSp", "Parch", "Fare"]].assign(FamilySize=e["FamilySize"].values).describe().T.round(2), width="stretch")

with tabs[3]:
    method = st.segmented_control("Method", ["pearson", "spearman"], default="pearson", format_func=str.capitalize)
    corr = eda.correlation(raw, method=method or "pearson")
    charts.show(charts.heatmap(corr, f"{(method or 'pearson').capitalize()} correlation matrix", height=520))
    tc = corr["Survived"].drop("Survived").sort_values(key=np.abs, ascending=False)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        charts.show(charts.signed_hbar(tc.index, tc.values, "Correlation with survival", xlabel="correlation coefficient", fmt="+.2f", scale=1))
    with c2:
        learn("Correlation", "A number from −1 to +1 describing how two variables move together <i>linearly</i>. "
              "Sex (female = 1) and Pclass show the strongest relationships with survival. Correlation only captures linear association — "
              "trees and other models can also pick up non-linear patterns.")
        causation_note()

with tabs[4]:
    mt = eda.missing_table(raw)
    m1, m2 = st.columns([1, 1.2])
    with m1:
        charts.show(charts.hbar(mt["column"], mt["missing_pct"], "Missing values (%)", color=PERISHED, fmt=".1f", xlabel="% missing", height=260))
    with m2:
        st.dataframe(mt, hide_index=True, width="stretch", column_config={"strategy": st.column_config.TextColumn(width="large")})
    a_known = e.groupby("Class")["AgeKnown"].mean()
    fig = go.Figure(go.Bar(x=a_known.index, y=1 - a_known.values, marker=dict(color=PERISHED, cornerradius=4), text=[f"{1 - v:.0%}" for v in a_known.values],
                           textposition="outside", hovertemplate="%{x}: %{y:.1%} missing age<extra></extra>"))
    fig.update_layout(title="Share of passengers with unknown age, by class", height=320)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.5])
    charts.show(fig)
    note("Ages are missing more often in 3rd class, so 'age unknown' is not random. The model gets an <code>AgeKnown</code> flag and imputes age "
         "with the median <i>of the passenger's title group</i> (Master → young boys, Miss → young women…), learned on training folds only.", "lightbulb")

with tabs[5]:
    st.caption("Generated from the currently filtered data — nothing is hard-coded.")
    insight_cards(eda.auto_insights(raw))
    causation_note()
