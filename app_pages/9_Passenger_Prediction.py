"""Passenger Prediction — the interactive survival estimator."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src import config
from src import explainability as xai
from src.prediction import PORTS, SAMPLE_PROFILES, TITLES, PassengerInput, SimilarPassengerIndex, input_from_row, predict_one, what_if_curve
from src.ui import charts, state
from src.ui.components import boarding_pass, disclaimer, kpi_tiles, learn, note, page_header, ring, section
from src.ui.ship import render_ship
from src.ui.theme import BRASS, PERISHED, SURVIVED

page_header("Predict · 09", "Passenger Prediction",
            "Describe a passenger and the model estimates their probability of survival — then shows <i>why</i>, how the estimate would change, "
            "and which real passengers were most similar.")
df = state.require_data()
key = state.require_models()
pipe = state.get_pipeline(key, "full")
disclaimer()

# ── form state & presets ───────────────────────────────────────────────
DEFAULTS = {"f_pclass": "3rd", "f_sex": "male", "f_age": 28, "f_age_unknown": False, "f_sibsp": 0, "f_parch": 0, "f_fare": 8.05,
            "f_port": "S", "f_cabin": "", "f_title": "Auto", "f_name": None, "f_ticket": None}
CLASS_OPTS = {"1st": 1, "2nd": 2, "3rd": 3}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_profile(inp: PassengerInput, name: str | None = None):
    st.session_state.update({
        "f_pclass": {1: "1st", 2: "2nd", 3: "3rd"}[inp.pclass], "f_sex": inp.sex, "f_age": int(round(inp.age)) if inp.age is not None else 28,
        "f_age_unknown": inp.age is None, "f_sibsp": int(inp.sibsp), "f_parch": int(inp.parch), "f_fare": float(inp.fare or 0.0),
        "f_port": inp.embarked, "f_cabin": inp.cabin or "", "f_title": inp.title or "Auto", "f_name": name, "f_ticket": inp.ticket,
    })


def on_preset():
    choice = st.session_state.get("preset")
    if choice in SAMPLE_PROFILES:
        load_profile(SAMPLE_PROFILES[choice])
        st.session_state["pred_result_input"] = None


def on_real():
    pid = st.session_state.get("real_pid")
    if pid:
        row = df.set_index("PassengerId").loc[pid]
        inp = input_from_row(row.rename(None))
        load_profile(inp, name=str(row["Name"]))
        st.session_state["pred_result_input"] = None


p1, p2 = st.columns(2)
p1.selectbox("Start from a sample profile (clearly fictional)", ["—"] + list(SAMPLE_PROFILES), key="preset", on_change=on_preset)
names = df.sort_values("Name")
p2.selectbox("…or load a real passenger from train.csv", [None] + names["PassengerId"].tolist(), key="real_pid", on_change=on_real,
             format_func=lambda i: "—" if i is None else f"{names.set_index('PassengerId').loc[i, 'Name']} (#{i})")

with st.form("passenger_form", border=True):
    st.markdown("##### Passenger details")
    a1, a2, a3 = st.columns(3)
    a1.segmented_control("Passenger class", list(CLASS_OPTS), key="f_pclass", help="1 = upper, 2 = middle, 3 = lower class.")
    a2.segmented_control("Sex", ["female", "male"], key="f_sex", format_func=str.capitalize)
    a3.selectbox("Port of embarkation", list(PORTS), key="f_port", format_func=lambda c: f"{PORTS[c]} ({c})")
    b1, b2 = st.columns([2, 1])
    b1.slider("Age (years)", 0, 80, key="f_age", help="Training data covers ages 0.42–80.")
    b2.checkbox("Age unknown", key="f_age_unknown", help="Like 20% of real passengers — the model imputes it from the title.")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Siblings / spouses aboard", 0, 8, step=1, key="f_sibsp")
    c2.number_input("Parents / children aboard", 0, 6, step=1, key="f_parch")
    c3.number_input("Ticket fare (£)", 0.0, 600.0, step=0.5, format="%.2f", key="f_fare",
                    help="Typical fares: 1st class £30–£150 · 2nd class £10–£30 · 3rd class £7–£20.")
    d1, d2 = st.columns(2)
    d1.text_input("Cabin (optional)", key="f_cabin", placeholder="e.g. C85 — leave blank if unknown", max_chars=15)
    d2.selectbox("Title (optional)", TITLES, key="f_title", help="'Auto' infers Mr/Mrs/Miss/Master from sex, age and family.")
    submitted = st.form_submit_button("Estimate survival", type="primary", icon=":material/query_stats:", width="stretch")


def current_input() -> PassengerInput:
    s = st.session_state
    return PassengerInput(
        pclass=CLASS_OPTS.get(s["f_pclass"] or "3rd", 3), sex=s["f_sex"] or "male", age=None if s["f_age_unknown"] else float(s["f_age"]),
        sibsp=int(s["f_sibsp"]), parch=int(s["f_parch"]), fare=float(s["f_fare"]), embarked=s["f_port"],
        cabin=(s["f_cabin"] or "").strip() or None, title=None if s["f_title"] == "Auto" else s["f_title"],
        name=s.get("f_name"), ticket=s.get("f_ticket"),
    )


if submitted:
    st.session_state["pred_result_input"] = current_input()
    st.session_state["log_pending"] = True
inp = st.session_state.get("pred_result_input")
if inp is None:
    note("Fill in the form (or load a profile) and press <b>Estimate survival</b>.", "touch_app")
    st.stop()

res = predict_one(pipe, inp)
if not res["ok"]:
    for e in res["errors"]:
        st.error(e)
    st.stop()
for w in res["warnings"]:
    st.warning(w, icon=":material/warning:")
p = res["prob_survived"]


@st.cache_data(show_spinner=False, max_entries=64)
def explain_cached(model_key: str, records: list, mtime: float):
    X = pd.DataFrame(records)
    for c in ("Name", "Ticket", "Cabin", "Title"):
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    bg = df.drop(columns=["Survived"]).sample(150, random_state=7)
    c = xai.explain_one(state.get_pipeline(model_key, "full"), X, bg)
    return c, c.attrs["base_value"], c.attrs["method"]


contrib, base, method = explain_cached(key, res["frame"].to_dict("records"), state.registry_mtime())
contrib.attrs["base_value"] = base

# log the prediction once per submission (monitoring)
if st.session_state.pop("log_pending", False):
    log_path = config.path("reports_dir") / "prediction_log.csv"
    rec = res["frame"].drop(columns=["PassengerId"]).assign(model=key, prob_survived=round(p, 4), time=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rec.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

# ── result ──────────────────────────────────────────────────────────────
section("Survival estimate", "01", "confirmation_number")
cls_name = {1: "First", 2: "Second", 3: "Third"}[inp.pclass]
age_txt = "unknown" if inp.age is None else f"{inp.age:g}"
fields = {"Class": cls_name, "Sex": inp.sex.capitalize(), "Age": age_txt, "Port": PORTS[inp.embarked], "Sib/Sp": inp.sibsp, "Par/Ch": inp.parch,
          "Fare": f"£{inp.fare:.2f}", "Cabin": inp.cabin or "—"}
who = inp.name or f"{inp.title or contrib.loc[contrib['feature'] == 'TitleGroup', 'value'].iloc[0] if (contrib['feature'] == 'TitleGroup').any() else 'Passenger'} · custom profile"
top = contrib.head(6)[["label", "value", "contribution"]].to_dict("records")
boarding_pass({"title": who, "fields": fields, "from": PORTS[inp.embarked]}, p, res["confidence"], top, state.model_name(key))
r1, r2 = st.columns([1, 1.15])
with r1:
    kpi_tiles([
        {"label": "Survival probability", "value": f"{p:.1%}", "icon": "health_and_safety", "accent": SURVIVED},
        {"label": "Non-survival probability", "value": f"{1 - p:.1%}", "icon": "person_off", "accent": PERISHED},
    ])
    fig = go.Figure()
    fig.add_bar(y=[""], x=[p * 100], orientation="h", name="Survive", marker=dict(color=SURVIVED, cornerradius=6),
                text=[f"Survive {p:.0%}" if p > 0.12 else ""], textposition="inside", insidetextanchor="start", hovertemplate="Survival %{x:.1f}%<extra></extra>")
    fig.add_bar(y=[""], x=[(1 - p) * 100], orientation="h", name="Not survive", marker=dict(color=PERISHED, cornerradius=6),
                text=[f"Not survive {1 - p:.0%}" if p < 0.88 else ""], textposition="inside", insidetextanchor="end", hovertemplate="Non-survival %{x:.1f}%<extra></extra>")
    fig.add_vline(x=50, line=dict(color="rgba(255,255,255,0.6)", dash="dot", width=1))
    fig.update_layout(barmode="stack", height=110, showlegend=False, margin=dict(l=4, r=4, t=8, b=8), title=None)
    fig.update_xaxes(visible=False, range=[0, 100])
    fig.update_yaxes(visible=False)
    charts.show(fig)
    st.caption(f"Model: **{state.model_name(key)}** (refit on all 891 labelled passengers) · confidence: {res['confidence']} · "
               "the dotted line marks the 50% decision threshold.")
with r2:
    render_ship("passenger", height=340, key="pred_ship",
                beacon={"pclass": inp.pclass, "prob": p, "label": f"Your passenger · {p:.0%} est. survival"})

# ── explanation ─────────────────────────────────────────────────────────
section("Why this estimate?", "02", "psychology")
w1, w2 = st.columns([1.45, 1])
with w1:
    charts.show(charts.waterfall(base, contrib, p, "Feature contributions (SHAP, percentage points)"))
with w2:
    nar = xai.narrative(contrib)
    st.markdown("**Factors associated with a higher estimate**")
    st.markdown("\n".join(f"- {t}" for t in nar["higher"]) or "- none of note")
    st.markdown("**Factors associated with a lower estimate**")
    st.markdown("\n".join(f"- {t}" for t in nar["lower"]) or "- none of note")
    note(f"Method: {method}. These are <b>statistical associations</b> learned from 1912 passenger records — "
         "they describe the model's reasoning, not what caused anyone's survival.", "hub")

# ── what-if ─────────────────────────────────────────────────────────────
section("What-if lab", "03", "science")
learn("What-if analysis", "Change one thing and watch the estimate move. The curves vary a single input while everything else stays fixed "
      "(individual conditional expectation curves).")


@st.fragment
def what_if():
    c1, c2, c3, c4 = st.columns(4)
    w_cls = c1.select_slider("Class", [1, 2, 3], value=inp.pclass, key="wi_cls")
    w_sex = c2.radio("Sex", ["female", "male"], index=["female", "male"].index(inp.sex), horizontal=True, key="wi_sex")
    w_age = c3.slider("Age", 0, 80, int(inp.age if inp.age is not None else 28), key="wi_age")
    w_fam = c4.slider("Family aboard (SibSp+Parch)", 0, 10, inp.sibsp + inp.parch, key="wi_fam")
    sib = min(w_fam, 1 if w_fam else 0)
    par = w_fam - sib
    alt = PassengerInput(w_cls, w_sex, float(w_age), sib, par, inp.fare, inp.embarked, inp.cabin,
                         title=None, ticket=None)
    pa = float(pipe.predict_proba(alt.to_frame())[0, 1])
    m1, m2 = st.columns([1, 3])
    with m1:
        st.markdown(ring(pa, "what-if"), unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;color:var(--ink-2);margin-top:.4rem'>{(pa - p) * 100:+.1f} pts vs your passenger</div>",
                    unsafe_allow_html=True)
    with m2:
        t1, t2 = st.tabs(["Age curve", "Fare curve"])
        with t1:
            curve = what_if_curve(pipe, alt, "Age", list(range(1, 81)))
            charts.show(charts.line_ice(curve, "Age", "Estimate vs age (other inputs fixed)", current=w_age), key="ice_age")
        with t2:
            fares = np.round(np.geomspace(4, 300, 60), 2)
            curve = what_if_curve(pipe, alt, "Fare", fares)
            fig = charts.line_ice(curve, "Fare", "Estimate vs fare (other inputs fixed)", current=inp.fare, xlabel="Fare (£)")
            fig.update_xaxes(type="log")
            charts.show(fig, key="ice_fare")
    st.caption("Title is re-inferred from sex/age in the what-if lab, so crossing age 13 or 18 can change the title (e.g. Master → Mr).")


what_if()

# ── similar passengers ─────────────────────────────────────────────────
section("Most similar real passengers", "04", "group")


@st.cache_resource(show_spinner=False)
def neighbor_index(mtime: float):
    return SimilarPassengerIndex(df, k=5)


idx = neighbor_index(state.train_mtime())
sim = idx.query(res["frame"], k=5)
note("These are <b>real historical records</b> from Kaggle's train.csv, matched on class, sex, age, family, fare and port "
     "(standardised distance). Their actual outcomes are shown for context.", "history_edu")
show = sim[["similarity", "Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].copy()
show["Outcome"] = show["Survived"].map({1: "✔ Survived", 0: "✘ Did not survive"})
st.dataframe(show.drop(columns=["Survived"]), hide_index=True, width="stretch",
             column_config={"similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=1, format="%.2f"),
                            "Fare": st.column_config.NumberColumn("Fare", format="£%.2f")})
st.caption(f"{int(sim['Survived'].sum())} of these {len(sim)} similar passengers survived.")

with st.expander("Passenger galaxy — see this passenger among all 891 in 3D", expanded=False):
    @st.cache_data(show_spinner=False)
    def galaxy(mtime: float):
        pca = PCA(n_components=3, random_state=42).fit(idx.matrix)
        return pca, pca.transform(idx.matrix)

    pca, coords = galaxy(state.train_mtime())
    me = pca.transform(idx.encoder.transform(res["frame"]))[0]
    nn_idx = idx.nn.kneighbors(idx.encoder.transform(res["frame"]), n_neighbors=5)[1][0]
    hover = [f"{n} · class {c} · {s}" for n, c, s in zip(idx.df["Name"], idx.df["Pclass"], idx.df["Sex"])]
    charts.show(charts.scatter3d_galaxy(coords, idx.df["Survived"].to_numpy(), hover, "Passengers projected to 3D (PCA of standardised features)",
                                        highlight=me, neighbors=coords[nn_idx]))
    st.caption("Each point is a real passenger; nearby points have similar features. Lines connect your passenger to the 5 most similar ones. Drag to rotate.")
