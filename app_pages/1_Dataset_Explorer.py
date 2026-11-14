"""Dataset Explorer — source, validation, dictionary, filterable table, missing data, downloads."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, data_loader
from src.data_validation import COLUMN_DESCRIPTIONS, validate_dataframe
from src.eda import enrich, missing_table
from src.feature_engineering import TitanicFeatureEngineer
from src.ui import charts, state
from src.ui.components import kpi_tiles, learn, note, page_header, pill, section
from src.ui.theme import BRASS, PERISHED, SURVIVED, VIOLET

page_header("Data · 01", "Dataset Explorer",
            "The labelled Kaggle <code>train.csv</code> (891 passengers) and the unlabelled competition <code>test.csv</code> (418 passengers). "
            "Inspect, filter, validate and download.")
df = state.require_data()

# ── source & validation ───────────────────────────────────────────────
section("Source & validation", "01", "verified")
cols = st.columns(2)
for col, kind in zip(cols, ("train", "test")):
    stt = data_loader.dataset_status(kind)
    with col, st.container(border=True):
        badge = pill("Official Kaggle file · SHA-256 verified", "ok") if stt.is_official else (pill("Custom upload", "brass") if stt.exists else pill("Missing", "perished"))
        st.markdown(f"**{kind}.csv** &nbsp; {badge}", unsafe_allow_html=True)
        if stt.exists:
            st.caption(f"{stt.rows} rows · `{stt.path.relative_to(config.PROJECT_ROOT)}` · sha256 {stt.sha256[:16]}…")
        else:
            st.caption("Not found — upload it below or run `python scripts/download_data.py`.")

rep = validate_dataframe(df, "train")
kpi_tiles([
    {"label": "Rows", "count_to": rep.n_rows, "icon": "table_rows", "accent": BRASS},
    {"label": "Columns", "count_to": rep.n_cols, "icon": "view_column", "accent": VIOLET},
    {"label": "Missing cells", "count_to": int(df.isna().sum().sum()), "icon": "help", "sub": f"{100 * df.isna().sum().sum() / df.size:.1f}% of all cells", "accent": PERISHED},
    {"label": "Duplicate rows", "count_to": rep.duplicate_rows, "icon": "content_copy", "sub": f"{rep.duplicate_ids} duplicate IDs", "accent": SURVIVED},
    {"label": "Validation", "value": "Passed" if rep.is_valid else "Failed", "icon": "rule", "sub": f"{len(rep.errors)} errors · {len(rep.warnings)} warnings", "accent": BRASS},
])
with st.expander("Full validation report — required columns, types, value ranges, target, duplicates", expanded=False):
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    for i in rep.info:
        st.info(i)
    if rep.is_valid:
        st.success("All required columns present · numeric types valid · Pclass ∈ {1,2,3} · Sex ∈ {male, female} · Embarked ∈ {C,Q,S} · "
                    "Survived ∈ {0,1} with both classes · ages/fares in plausible ranges · no duplicate PassengerIds.")

with st.expander("Replace the dataset (upload CSV) · Kaggle download instructions"):
    u1, u2 = st.columns(2)
    for col, kind in ((u1, "train"), (u2, "test")):
        up = col.file_uploader(f"Upload {kind}.csv", type="csv", key=f"upload_{kind}")
        if up is not None and state.demo_mode():
            from src.data_validation import validate_dataframe as _v
            r = _v(pd.read_csv(up), kind)
            (col.success if r.is_valid else col.error)(("Valid" if r.is_valid else "Invalid: " + "; ".join(r.errors))
                                                     + " — replacing the dataset is disabled in the public demo.")
        elif up is not None and col.button(f"Validate & save {kind}.csv", key=f"save_{kind}"):
            ok, msgs = data_loader.save_uploaded(up.getvalue(), kind)
            (col.success if ok else col.error)("\n\n".join(msgs))
            if ok:
                state.refresh_caches()
    st.markdown("**Manual download**\n\n" + data_loader.MANUAL_INSTRUCTIONS)
    st.markdown("**Kaggle API**\n\n" + data_loader.KAGGLE_INSTRUCTIONS)
    st.caption(f"Kaggle credentials detected: {'yes' if data_loader.kaggle_credentials_available() else 'no'}")

# ── dictionary ────────────────────────────────────────────────────────
section("Column dictionary", "02", "menu_book")
st.dataframe(rep.summary, hide_index=True, width="stretch", column_config={
    "missing_pct": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%"),
    "description": st.column_config.TextColumn("description", width="large"),
})

# ── interactive table ─────────────────────────────────────────────────
section("Interactive passenger table", "03", "filter_alt")
learn("Filtering & sorting", "Use the filters to slice the passengers; click any column header in the table to sort. "
      "Everything here is the raw historical record — nothing is imputed.")
e = enrich(df)
f1, f2, f3, f4 = st.columns(4)
cls = f1.multiselect("Class", [1, 2, 3], default=[1, 2, 3], format_func=lambda c: f"{c}{'st' if c == 1 else 'nd' if c == 2 else 'rd'} class")
sex = f2.multiselect("Sex", ["female", "male"], default=["female", "male"])
port = f3.multiselect("Port", ["Southampton", "Cherbourg", "Queenstown", "Unknown"], default=["Southampton", "Cherbourg", "Queenstown", "Unknown"])
outcome = f4.multiselect("Outcome", ["Survived", "Did not survive"], default=["Survived", "Did not survive"])
g1, g2 = st.columns([2, 1])
age = g1.slider("Age range (passengers with unknown age are kept if 'include unknown' is on)", 0, 80, (0, 80))
name_q = g2.text_input("Search name / ticket / cabin", placeholder="e.g. Allison, PC 17599, C85")
inc_unknown = st.toggle("Include passengers with unknown age", value=True)

mask = e["Pclass"].isin(cls) & e["Sex"].isin(sex) & e["Port"].isin(port) & e["Outcome"].isin(outcome)
age_ok = e["Age"].between(age[0], age[1])
mask &= age_ok | (e["Age"].isna() & inc_unknown)
if name_q:
    q = name_q.lower()
    mask &= (e["Name"].str.lower().str.contains(q, regex=False) | e["Ticket"].astype(str).str.lower().str.contains(q, regex=False)
             | e["Cabin"].fillna("").str.lower().str.contains(q, regex=False))
view = df[mask.values]
st.caption(f"Showing **{len(view):,}** of {len(df):,} passengers · survival rate in selection: "
           f"**{view['Survived'].mean():.1%}**" if len(view) else "No passengers match these filters.")
st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
    "Survived": st.column_config.CheckboxColumn("Survived"),
    "Fare": st.column_config.NumberColumn("Fare (£)", format="£%.2f"),
    "Age": st.column_config.NumberColumn(format="%.1f"),
})
st.download_button("Download filtered rows (CSV)", view.to_csv(index=False), "titanic_filtered.csv", "text/csv", icon=":material/download:")

# ── missing values ────────────────────────────────────────────────────
section("Missing-value analysis", "04", "help_center")
mt = missing_table(df)
m1, m2 = st.columns([1, 1.3])
with m1:
    charts.show(charts.hbar(mt["column"], mt["missing_pct"], "Missing values (% of rows)", color=PERISHED, fmt=".1f", xlabel="% missing", height=280))
    st.dataframe(mt, hide_index=True, width="stretch")
with m2:
    sample = df.sort_values("PassengerId")
    z = sample.isna().astype(int).T.values
    fig = go.Figure(go.Heatmap(z=z, x=sample["PassengerId"], y=sample.columns, colorscale=[[0, "#12244a"], [1, PERISHED]], showscale=False,
                               ygap=2, hovertemplate="Passenger %{x}<br>%{y}: %{z}<extra>1 = missing</extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="PassengerId", showgrid=False)
    fig.update_layout(title="Missingness map (orange = missing)", height=420)
    charts.show(fig)
note("Cabin is missing for 77% of passengers — far more often for 2nd/3rd class — so its <b>absence is informative</b>. "
     "Instead of imputing a cabin, the pipeline keeps <code>CabinKnown</code> and <code>Deck = Unknown</code> as features.", "lightbulb")
cab = e.groupby("Class")["CabinKnown"].mean().rename("Share with a recorded cabin").to_frame().T
st.dataframe(cab.style.format("{:.0%}"), width="stretch")

# ── processed dataset ─────────────────────────────────────────────────
section("Processed dataset", "05", "download")
fe = TitanicFeatureEngineer(feature_set="engineered").fit(df)
processed = pd.concat([df[["PassengerId", "Survived"]].reset_index(drop=True), fe.transform(df).reset_index(drop=True)], axis=1)
note("For exploration only: here the feature engineer is fitted on all 891 rows. During training it is refitted inside every "
     "cross-validation fold, so validation rows never influence imputation values or ticket counts.", "info")
st.dataframe(processed.head(200), hide_index=True, width="stretch", height=320)
c1, c2 = st.columns(2)
c1.download_button("Download processed train set (CSV)", processed.to_csv(index=False), "titanic_processed.csv", "text/csv", icon=":material/download:", type="primary")
processed.to_csv(config.path("processed_dir") / "train_processed.csv", index=False)
c2.caption(f"Also saved to `data/processed/train_processed.csv` ({processed.shape[0]} × {processed.shape[1]}).")

if data_loader.config.raw_file("test").exists():
    with st.expander("Peek at the Kaggle competition test set (no labels)"):
        t = state.get_test()
        st.caption(f"{len(t)} passengers · Survived column is not provided by Kaggle.")
        st.dataframe(t, hide_index=True, width="stretch", height=300)
