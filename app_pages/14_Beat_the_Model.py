"""Beat the Model — guess the outcome of real holdout passengers, then compare with the model."""
import numpy as np
import pandas as pd
import streamlit as st

from src.ui import state
from src.ui.components import esc, learn, note, page_header, section
from src.ui.theme import PERISHED, SURVIVED

page_header("Predict · 10", "Beat the Model",
            "Ten real passengers from the holdout set — people the model never saw while training. Read each record, make your call, "
            "then see what the model predicted and what actually happened.")
df = state.require_data()
key = state.require_models()
ev = state.get_evaluation(key)
pipe = state.get_pipeline(key, "dev")
note("These are real people from the 1912 passenger list. The game is about understanding statistical patterns — and their limits — "
     "and is meant to be played with respect for those who were aboard.", "favorite")

ROUNDS = 10
hold = df.set_index("PassengerId").loc[ev["holdout_ids"]].reset_index()


def new_game():
    rng = np.random.default_rng()
    st.session_state["game"] = {"order": rng.choice(len(hold), ROUNDS, replace=False).tolist(), "i": 0, "you": 0, "model": 0,
                                "history": [], "revealed": False, "guess": None, "key": key}


g = st.session_state.get("game")
if g is None or g.get("key") != key:
    new_game()
    g = st.session_state["game"]


def score_html():
    return f"""<div class="tml-score"><div class="s"><div class="n">{g['you']}</div><div class="t">You</div></div>
    <div class="s"><div class="n">{min(g['i'] + (1 if g['revealed'] else 0), ROUNDS)}/{ROUNDS}</div><div class="t">Round</div></div>
    <div class="s"><div class="n">{g['model']}</div><div class="t">{esc(state.model_name(key))}</div></div></div>"""


st.markdown(score_html(), unsafe_allow_html=True)
st.write("")

if g["i"] >= ROUNDS:
    section("Final result", None, "emoji_events")
    if g["you"] > g["model"]:
        st.success(f"You beat the model {g['you']}–{g['model']}. Human intuition, or luck, wins this round.")
    elif g["you"] == g["model"]:
        st.info(f"A draw, {g['you']}–{g['model']}.")
    else:
        st.warning(f"The model wins {g['model']}–{g['you']}. It has learned the patterns from 712 passengers — but it is still wrong sometimes.")
    hist = pd.DataFrame(g["history"])
    st.dataframe(hist, hide_index=True, width="stretch")
    learn("What does this show?", "Most people quickly discover the main pattern ('women and children first', class matters). "
          "The model applies the same patterns consistently, with calibrated probabilities. Both get the surprising cases wrong — "
          "survival also depended on chance, location on the ship and decisions that are not in the data.")
    if st.button("Play again", type="primary", icon=":material/replay:"):
        new_game()
        st.rerun()
    st.stop()

row = hold.iloc[g["order"][g["i"]]]
fam = int(row["SibSp"] + row["Parch"])
age = "unknown" if pd.isna(row["Age"]) else f"{row['Age']:g}"
port = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}.get(row["Embarked"], "unknown")
st.markdown(f"""<div class="tml-quiz-card">
  <div class="tml-eyebrow">Passenger {g['i'] + 1} of {ROUNDS} · #{int(row['PassengerId'])}</div>
  <h3 style="margin:.4rem 0 .9rem">{esc(row['Name'])}</h3>
  <div class="tml-pass" style="display:block;min-height:0;animation:none"><div class="fields" style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem 1rem">
    <div class="f"><div class="k">Class</div><div class="v">{int(row['Pclass'])}</div></div>
    <div class="f"><div class="k">Sex</div><div class="v">{esc(row['Sex'])}</div></div>
    <div class="f"><div class="k">Age</div><div class="v">{age}</div></div>
    <div class="f"><div class="k">Family aboard</div><div class="v">{fam}</div></div>
    <div class="f"><div class="k">Fare</div><div class="v">£{row['Fare']:.2f}</div></div>
    <div class="f"><div class="k">Port</div><div class="v">{port}</div></div>
    <div class="f"><div class="k">Cabin</div><div class="v">{esc(row['Cabin']) if pd.notna(row['Cabin']) else '—'}</div></div>
    <div class="f"><div class="k">Ticket</div><div class="v">{esc(row['Ticket'])}</div></div>
  </div></div></div>""", unsafe_allow_html=True)
st.write("")

if not g["revealed"]:
    c1, c2 = st.columns(2)
    guess = None
    if c1.button("I think they survived", type="primary", width="stretch", icon=":material/health_and_safety:"):
        guess = 1
    if c2.button("I think they did not survive", width="stretch", icon=":material/person_off:"):
        guess = 0
    if guess is not None:
        X = row.drop(labels=["Survived"]).to_frame().T
        for c in ("Pclass", "SibSp", "Parch", "PassengerId"):
            X[c] = X[c].astype(int)
        for c in ("Age", "Fare"):
            X[c] = X[c].astype(float)
        prob = float(pipe.predict_proba(X)[0, 1])
        truth = int(row["Survived"])
        g["you"] += int(guess == truth)
        g["model"] += int(int(prob >= 0.5) == truth)
        g["history"].append({"Passenger": row["Name"], "Your guess": "Survived" if guess else "Did not survive",
                             "Model": f"{'Survived' if prob >= 0.5 else 'Did not survive'} ({prob:.0%})", "Actual": "Survived" if truth else "Did not survive"})
        g.update(revealed=True, guess=guess, prob=prob, truth=truth)
        st.rerun()
else:
    prob, truth, guess = g["prob"], g["truth"], g["guess"]
    col = SURVIVED if truth else PERISHED
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='tml-card'><h4>Your guess</h4><p style='font-size:1.2rem;color:var(--ink)'>{'Survived' if guess else 'Did not survive'} "
                f"{'✔' if guess == truth else '✘'}</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='tml-card'><h4>Model ({esc(state.model_name(key))})</h4><p style='font-size:1.2rem;color:var(--ink)'>{prob:.0%} survival → "
                f"{'Survived' if prob >= 0.5 else 'Did not survive'} {'✔' if int(prob >= 0.5) == truth else '✘'}</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='tml-card' style='border-color:{col}'><h4>What actually happened</h4><p style='font-size:1.2rem;color:{col}'>"
                f"{'Survived' if truth else 'Did not survive'}</p></div>", unsafe_allow_html=True)
    st.write("")
    if st.button("Next passenger →" if g["i"] + 1 < ROUNDS else "See final result", type="primary", width="stretch"):
        g["i"] += 1
        g["revealed"] = False
        st.rerun()
