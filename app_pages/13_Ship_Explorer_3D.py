"""3D Ship Explorer — the passenger data mapped onto an interactive 3D Titanic."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

from src import config, eda, model_manager
from src.train import get_cv, split_xy
from src.ui import charts, state
from src.ui.components import kpi_tiles, learn, note, page_header, section
from src.ui.ship import CLASS_LABELS, class_zones, render_ship
from src.ui.theme import AQUA, BRASS, GOLD, PERISHED, SURVIVED, VIOLET

page_header("Overview · 3D", "3D Ship Explorer",
            "Rotate the ship, hover the glowing class zones, colour 891 passenger lights by their real outcome or by the model's prediction, "
            "and scrub through the night of 14–15 April 1912.")
df = state.require_data()
reg = state.get_registry()
key = state.active_model()


@st.cache_data(show_spinner="Computing out-of-fold predictions for every passenger…")
def oof_predictions(model_key: str, mtime: float):
    """Each passenger is predicted by a model that did NOT see them (5-fold cross_val_predict)."""
    if config.BROWSER_DEMO:  # refitting needs the full app; these were computed the same way when the demo was built
        return np.asarray(model_manager.browser_oof()[model_key])
    X, y = split_xy(df)
    pipe = clone(state.get_pipeline(model_key, "dev"))
    prob = cross_val_predict(pipe, X, y, cv=get_cv(), method="predict_proba", n_jobs=-1)[:, 1]
    return prob


rows = []
if key:
    prob = oof_predictions(key, state.registry_mtime())
    for pid, c, s, p in zip(df["PassengerId"], df["Pclass"], df["Survived"], prob):
        rows.append([int(pid), int(c), int(s), int(p >= 0.5), round(float(p), 4)])
else:
    for pid, c, s in zip(df["PassengerId"], df["Pclass"], df["Survived"]):
        rows.append([int(pid), int(c), int(s), None, None])

learn("How to use the 3D view", "Drag to rotate · scroll to zoom · right-drag to pan. Use the on-ship controls to change the camera, the time of day, "
      "the colour mode of the passenger lights, show the lifeboats, or play the timeline of the sinking. Click a class zone to analyse it below.")
event = render_ship("explorer", height=640, key="explorer_ship", zones=class_zones(df), passengers=rows,
                    color_by="actual", stats={"total": len(df), "survived": int(df["Survived"].sum())})
note("Zones are approximate — the dataset does not record where passengers were on board. "
     + (f"Model colours use <b>out-of-fold</b> predictions from {state.model_name(key)}: each passenger is predicted by a model trained without them."
        if key else "Train a model to enable the 'Model prediction' and 'Correct?' colour modes."), "info")

if isinstance(event, dict) and event.get("zone"):
    st.session_state["explorer_zone"] = event["zone"]
zone = st.session_state.get("explorer_zone", "class1")
cls = {"class1": 1, "class2": 2, "class3": 3}.get(zone, 1)
sel = st.segmented_control("Analyse zone", [1, 2, 3], default=cls, format_func=lambda c: CLASS_LABELS[c], key=f"zone_sel_{zone}")
cls = sel or cls

sub = df[df["Pclass"] == cls]
e = eda.enrich(sub)
section(f"{CLASS_LABELS[cls]} — {len(sub)} passengers", "01", "groups")
tiles = [
    {"label": "Passengers", "count_to": len(sub), "icon": "groups", "accent": {1: GOLD, 2: VIOLET, 3: AQUA}[cls]},
    {"label": "Survival rate", "value": f"{sub['Survived'].mean():.1%}", "icon": "health_and_safety", "sub": f"overall {df['Survived'].mean():.1%}", "accent": SURVIVED},
    {"label": "Women · survival", "value": f"{sub[sub['Sex'] == 'female']['Survived'].mean():.0%}", "icon": "female", "accent": SURVIVED},
    {"label": "Men · survival", "value": f"{sub[sub['Sex'] == 'male']['Survived'].mean():.0%}", "icon": "male", "accent": PERISHED},
    {"label": "Median fare", "value": f"£{sub['Fare'].median():.2f}", "icon": "payments", "accent": BRASS},
]
if key:
    idx = df["Pclass"] == cls
    acc = float(((prob[idx.values] >= 0.5).astype(int) == df.loc[idx, "Survived"].values).mean())
    tiles.append({"label": "Model accuracy (OOF)", "value": f"{acc:.1%}", "icon": "verified", "sub": state.model_name(key), "accent": VIOLET})
kpi_tiles(tiles)
z1, z2 = st.columns(2)
with z1:
    g = eda.survival_by(e, "AgeGroup", order=__import__("src.feature_engineering", fromlist=["AGE_LABELS"]).AGE_LABELS + ["Unknown"])
    charts.show(charts.survival_rate_bar(g[g["passengers"] > 0], "AgeGroup", f"{CLASS_LABELS[cls]}: survival by age group", "Age group",
                                         overall=float(sub["Survived"].mean())), key=f"z_age_{cls}")
with z2:
    deck = e[e["Deck"] != "Unknown"]
    if len(deck):
        g = eda.survival_by(deck, "Deck").sort_values("Deck")
        charts.show(charts.outcome_stack(g, "Deck", f"{CLASS_LABELS[cls]}: recorded decks ({len(deck)} of {len(e)} known)", "Deck"), key=f"z_deck_{cls}")
    else:
        st.info("No cabin/deck information recorded for this class.")

section("Historical context", "02", "history_edu")
h1, h2, h3 = st.columns(3)
h1.markdown("""<div class="tml-card"><h4>Lifeboats</h4><p>20 lifeboats — 14 standard, 2 emergency cutters and 4 collapsibles — with a total capacity of
about 1,178 people, for about 2,224 people aboard.</p></div>""", unsafe_allow_html=True)
h2.markdown("""<div class="tml-card"><h4>The night</h4><p>23:40 collision with an iceberg · 00:05 order to uncover the lifeboats ·
00:45 first lifeboat lowered · 02:05 last lifeboat lowered · 02:18 lights fail · 02:20 the ship sinks.</p></div>""", unsafe_allow_html=True)
h3.markdown("""<div class="tml-card"><h4>The data</h4><p>Kaggle's 891 + 418 records cover 1,309 passengers. Crew are not included.
Times and lifeboat figures here are historical context, not from the dataset.</p></div>""", unsafe_allow_html=True)
st.caption("Approximate times from historical accounts. In memory of the more than 1,500 people who lost their lives.")
