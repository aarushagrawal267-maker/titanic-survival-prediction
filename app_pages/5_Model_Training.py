"""Model Training — pick an algorithm, configure, train with live progress, save to the registry."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.model_selection import cross_validate  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from src.evaluate import METRIC_LABELS, METRICS  # noqa: E402
from src.feature_engineering import BASE_FEATURES, ENGINEERED_FEATURES  # noqa: E402
from src.models import MODEL_ORDER, SPECS, coerce_knob_value  # noqa: E402
from src.preprocessing import build_pipeline, friendly, transformed_feature_names  # noqa: E402
from src.train import get_cv, recommended_setup, split_data, train_model  # noqa: E402
from src.ui import charts, state  # noqa: E402
from src.ui.components import kpi_tiles, learn, note, page_header, pill, section  # noqa: E402
from src.ui.theme import BRASS, PERISHED, SURVIVED, VIOLET  # noqa: E402

page_header("Modeling · 05", "Model Training",
            "Choose an algorithm, configure it by hand or let RandomizedSearchCV tune it, and watch stratified 5-fold cross-validation run. "
            "Training only happens when you press the button — results are saved to the model registry.")
df = state.require_data()

tab_train, tab_overfit, tab_switch = st.tabs(["Train a model", "Overfitting playground", "Feature switchboard"])

# ═════════════════════════════ TRAIN ═════════════════════════════
with tab_train:
    left, right = st.columns([1.05, 1])
    with left:
        key = st.selectbox("Algorithm", MODEL_ORDER, format_func=lambda k: SPECS[k].name, key="train_algo")
        spec = SPECS[key]
        rec = recommended_setup(key)
        mode = st.segmented_control("Hyper-parameters", ["Auto-tune", "Manual"], default="Auto-tune", key="train_mode") or "Auto-tune"
        params = {}
        if mode == "Manual":
            st.caption("Manual settings — the model is cross-validated with exactly these values.")
            kc = st.columns(2)
            for i, k in enumerate(spec.knobs):
                with kc[i % 2]:
                    wkey = f"knob_{key}_{k['param']}"
                    if k["type"] == "int":
                        v = st.slider(k["param"], k["min"], k["max"], k["default"], step=k.get("step", 1), help=k["help"], key=wkey)
                    elif k["type"] == "float":
                        v = st.slider(k["param"], float(k["min"]), float(k["max"]), float(k["default"]), step=0.05, help=k["help"], key=wkey)
                    elif k["type"] == "log_slider":
                        grid = np.round(np.geomspace(k["min"], k["max"], 25), 4).tolist()
                        dflt = min(grid, key=lambda g: abs(g - k["default"]))
                        v = st.select_slider(k["param"], options=grid, value=dflt, help=k["help"], key=wkey)
                    else:
                        v = st.selectbox(k["param"], k["options"], index=k["options"].index(k["default"]), help=k["help"], key=wkey)
                    params[k["param"]] = coerce_knob_value(k, v)
        else:
            n_iter = st.slider("Search candidates (RandomizedSearchCV n_iter)", 5, 60, 20, step=5, key="train_niter",
                               help="Each candidate is evaluated with 5-fold CV — more candidates = better search, longer runtime.")
            with st.expander("Search space"):
                st.json({k.replace("model__", ""): (str(v) if not isinstance(v, list) else v) for k, v in spec.search_space.items()})
        o1, o2 = st.columns(2)
        fset = o1.selectbox("Feature set", ["engineered", "base"], index=["engineered", "base"].index(rec["feature_set"]),
                            format_func={"engineered": "Engineered (17 features)", "base": "Base (7 features)"}.get, key="train_fset")
        scalers = ["standard", "robust", "none"]
        scaler = o2.selectbox("Scaler", scalers, index=scalers.index(rec["scaler"]),
                              format_func={"standard": "StandardScaler", "robust": "RobustScaler", "none": "None"}.get, key="train_scaler",
                              help="Defaults come from the scaling experiment.")
        do_shap = st.toggle("Compute SHAP explanations after training (+10–40 s)", value=True, key="train_shap")
        start = st.button(f"Start training · {spec.name}", type="primary", icon=":material/rocket_launch:", width="stretch")
    with right:
        with st.container(border=True):
            st.markdown(f"#### {spec.name}")
            st.markdown(f"<p>{spec.purpose}</p>", unsafe_allow_html=True)
            st.markdown("**Advantages**  \n" + "\n".join(f"- {p}" for p in spec.pros))
            st.markdown("**Limitations**  \n" + "\n".join(f"- {c}" for c in spec.cons))
            st.markdown(" ".join([pill("needs scaling" if spec.scale_sensitive else "scale-invariant"),
                                  pill("tree-based" if spec.tree_based else "non-tree")]), unsafe_allow_html=True)
        reg = state.get_registry()
        if key in reg:
            m = reg[key]
            note(f"A saved <b>{spec.name}</b> exists (CV ROC-AUC {m['cv']['roc_auc_mean']:.3f}, trained {m.get('saved_at', '')[:16].replace('T', ' ')} UTC). "
                 "Training again replaces it.", "history")
        learn("Stratified K-fold cross-validation", "The 712-passenger development set is split into 5 folds that each keep the 38/62 survival balance. "
              "The model trains on 4 folds and is scored on the 5th, five times. The mean tells you the expected performance; the spread tells you how stable it is. "
              "The 179-passenger holdout set is only used afterwards.")

    if start:
        with st.status(f"Training {spec.name}…", expanded=True) as status:
            bar = st.progress(0.0, text="Starting…")
            log = st.empty()
            lines = []

            def progress(f, msg):
                bar.progress(min(1.0, f), text=msg)
                lines.append(f"`{time.strftime('%H:%M:%S')}` {msg}")
                log.markdown("  \n".join(lines[-8:]))

            t0 = time.time()
            res = train_model(df, key, params=params if mode == "Manual" else None, tune=(mode == "Auto-tune"),
                              n_iter=st.session_state.get("train_niter", 20) if mode == "Auto-tune" else None,
                              feature_set=fset, scaler=scaler, compute_shap=do_shap, progress=progress, source="app",
                              save=not state.demo_mode())
            saved_txt = "shown below (public demo: not saved)" if state.demo_mode() else "saved"
            status.update(label=f"{spec.name} trained in {time.time() - t0:.0f}s — {saved_txt}", state="complete", expanded=False)
        if not state.demo_mode():
            state.refresh_caches()
        st.session_state["last_training"] = {"key": key, "meta": res["meta"], "search": res["evaluation"].get("search_table")}
        st.toast(f"{spec.name} {'trained (not saved in the public demo)' if state.demo_mode() else 'saved to the model registry'}", icon=":material/check_circle:")

    last = st.session_state.get("last_training")
    if last:
        meta = last["meta"]
        section(f"Results — {meta['name']}", "✓", "task_alt")
        cv, ho = meta["cv"], meta["holdout"]
        kpi_tiles([{"label": f"CV {METRIC_LABELS[m]}", "value": f"{cv[f'{m}_mean']:.3f}", "sub": f"± {cv[f'{m}_std']:.3f} across folds",
                    "icon": "repeat", "accent": SURVIVED} for m in METRICS])
        kpi_tiles([{"label": f"Holdout {METRIC_LABELS[m]}", "value": f"{ho[m]:.3f}", "icon": "verified", "accent": BRASS} for m in METRICS]
                  + [{"label": "Fit time", "value": f"{meta['train_time_s']:.2f}s", "sub": f"total incl. tuning {meta['total_time_s']:.0f}s", "icon": "timer", "accent": VIOLET}])
        r1, r2 = st.columns([1.2, 1])
        with r1:
            folds = cv["folds"]
            fig = go.Figure()
            for i, m in enumerate(METRICS):
                fig.add_scatter(x=[f"Fold {j + 1}" for j in range(len(folds[m]))], y=folds[m], mode="lines+markers", name=METRIC_LABELS[m],
                                line=dict(width=2, color=charts.CATEGORICAL[i]), marker=dict(size=8, line=dict(color="#050b1a", width=1.5)))
            fig.update_layout(title="Individual fold scores", height=360)
            fig.update_yaxes(title="Score")
            charts.show(fig)
        with r2:
            st.markdown("**Best / used hyper-parameters**")
            st.json({k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in meta["best_params"].items()})
            st.caption(f"Feature set: **{meta['feature_set']}** · scaler: **{meta['scaler']}** · dev rows {meta['n_dev']} · holdout rows {meta['n_holdout']}")
        if last.get("search"):
            st.markdown("**Top search candidates** (RandomizedSearchCV, 5-fold CV on the development set)")
            stab = pd.DataFrame(last["search"])
            stab["params"] = stab["params"].astype(str)
            st.dataframe(stab.round(4), hide_index=True, width="stretch")
        note("CV scores of a tuned model are slightly optimistic (the same folds were used to pick the hyper-parameters). "
             "The holdout scores are the unbiased estimate.", "info")

    if state.demo_mode():
        note("Public demo: your training runs are shown to you but not saved, so other visitors keep using the reference models. "
             "Clone the repository to train and save your own.", "public")
    else:
        with st.expander("Train or re-train the entire model zoo"):
            st.caption("Runs all experiments and tunes all algorithms — the same as `python scripts/train_all.py`.")
            if st.button("Train all models", key="train_all_btn"):
                state.train_all_models(df)
                st.rerun()

# ═════════════════════════════ OVERFITTING ═════════════════════════════
with tab_overfit:
    learn("Overfitting", "A model that memorises the training passengers scores brilliantly on them but poorly on new ones. "
          "Slide the decision-tree depth: training accuracy keeps climbing, while validation accuracy peaks and then falls — that gap is overfitting.")

    @st.cache_data(show_spinner="Cross-validating trees of every depth…")
    def depth_curve(mtime: float):
        X_dev, _, y_dev, _ = split_data(df)
        rows = []
        for d in range(1, 21):
            pipe = build_pipeline(DecisionTreeClassifier(max_depth=d, random_state=42), scaler="none")
            r = cross_validate(pipe, X_dev, y_dev, cv=get_cv(), scoring="accuracy", return_train_score=True, n_jobs=-1)
            rows.append({"depth": d, "train": r["train_score"].mean(), "val": r["test_score"].mean(), "val_std": r["test_score"].std()})
        return pd.DataFrame(rows)

    curve = depth_curve(state.train_mtime())
    depth = st.slider("Decision-tree max_depth", 1, 20, 3, key="of_depth")
    best_d = int(curve.loc[curve["val"].idxmax(), "depth"])
    o1, o2 = st.columns([1.3, 1])
    with o1:
        fig = go.Figure()
        fig.add_scatter(x=curve["depth"], y=curve["train"], name="Training accuracy", mode="lines+markers", line=dict(color=BRASS, width=2.5))
        fig.add_scatter(x=curve["depth"], y=curve["val"], name="Validation accuracy (CV)", mode="lines+markers", line=dict(color=SURVIVED, width=2.5),
                        error_y=dict(type="data", array=curve["val_std"], color="rgba(58,142,230,0.4)", thickness=1))
        fig.add_vline(x=depth, line=dict(color="#ffffff", dash="dot", width=1), annotation_text=f"depth {depth}", annotation_font_color="#ffffff")
        fig.add_vrect(x0=best_d - 0.5, x1=best_d + 0.5, fillcolor="rgba(31,168,131,0.12)", line_width=0, annotation_text="sweet spot", annotation_position="bottom left")
        fig.update_layout(title="Train vs validation accuracy by tree depth", height=400)
        fig.update_xaxes(title="max_depth", dtick=1)
        fig.update_yaxes(title="Accuracy", tickformat=".0%")
        charts.show(fig)
    with o2:
        row = curve[curve["depth"] == depth].iloc[0]
        gap = row["train"] - row["val"]
        kpi_tiles([
            {"label": "Training accuracy", "value": f"{row['train']:.1%}", "icon": "school", "accent": BRASS},
            {"label": "Validation accuracy", "value": f"{row['val']:.1%}", "icon": "verified", "accent": SURVIVED},
            {"label": "Overfitting gap", "value": f"{gap * 100:.1f} pts", "icon": "warning" if gap > 0.06 else "check_circle",
             "sub": "large gap = memorising" if gap > 0.06 else "healthy", "accent": PERISHED if gap > 0.06 else SURVIVED},
        ])
        st.caption(f"Best validation accuracy at depth **{best_d}**.")
    if depth <= 4:
        X_dev, _, y_dev, _ = split_data(df)
        pipe = build_pipeline(DecisionTreeClassifier(max_depth=depth, random_state=42), scaler="none").fit(X_dev, y_dev)
        names = [n.split("__", 1)[1] for n in transformed_feature_names(pipe)]
        fig, ax = plt.subplots(figsize=(min(26, 5 + 4.2 * depth), 3 + 1.6 * depth), dpi=110)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        art = plot_tree(pipe.named_steps["model"], feature_names=names, class_names=["Died", "Survived"], filled=False, rounded=True, impurity=False,
                        proportion=True, precision=2, fontsize=9, ax=ax)
        for a in art:
            a.set_color("#eaf0fa")
            box = a.get_bbox_patch()
            if box is not None:
                txt = a.get_text()
                surv = "class = Survived" in txt
                box.set_facecolor("#16325f" if surv else "#3d2418")
                box.set_edgecolor(SURVIVED if surv else PERISHED)
            if hasattr(a, "arrow_patch") and a.arrow_patch is not None:
                a.arrow_patch.set_color("#7d8ca8")
        st.pyplot(fig, transparent=True, width="stretch")
        plt.close(fig)
        st.caption("Each box: split rule, share of passengers reaching it, class proportions [died, survived] and the majority class. "
                   "Blue = survived majority, orange = did-not-survive majority.")
    else:
        st.info("The tree is drawn for depth ≤ 4 — deeper trees have too many leaves to read, which is part of the point.")

# ═════════════════════════════ SWITCHBOARD ═════════════════════════════
with tab_switch:
    learn("Feature switchboard", "Turn features off and see how cross-validated performance changes. This is a quick, interactive version of the "
          "ablation study on the Feature Engineering page.")
    all_feats = BASE_FEATURES + ENGINEERED_FEATURES
    sw_model = st.selectbox("Model", ["logistic_regression", "random_forest", "gradient_boosting"], format_func=lambda k: SPECS[k].name, key="sw_model")
    chosen = st.pills("Active features (click to toggle)", all_feats, default=all_feats, selection_mode="multi", format_func=friendly, key="sw_feats")
    chosen = chosen or []
    if len(chosen) == 0:
        st.warning("Select at least one feature.")
    else:
        @st.cache_data(show_spinner="Running 5-fold CV…")
        def quick_cv(model_key: str, drop: tuple, mtime: float):
            X_dev, _, y_dev, _ = split_data(df)
            pipe = build_pipeline(SPECS[model_key].make(), scaler="standard" if SPECS[model_key].scale_sensitive else "none", drop_features=drop)
            r = cross_validate(pipe, X_dev, y_dev, cv=get_cv(), scoring=["roc_auc", "accuracy"], n_jobs=-1)
            return float(r["test_roc_auc"].mean()), float(r["test_accuracy"].mean()), float(r["test_roc_auc"].std())

        drop = tuple(sorted(set(all_feats) - set(chosen)))
        auc, acc, std = quick_cv(sw_model, drop, state.train_mtime())
        ref_auc, ref_acc, _ = quick_cv(sw_model, tuple(), state.train_mtime())
        kpi_tiles([
            {"label": "CV ROC-AUC", "value": f"{auc:.3f}", "sub": f"{(auc - ref_auc) * 100:+.1f} pts vs all features (± {std:.3f})", "icon": "insights",
             "accent": SURVIVED if auc >= ref_auc - 0.002 else PERISHED},
            {"label": "CV accuracy", "value": f"{acc:.1%}", "sub": f"{(acc - ref_acc) * 100:+.1f} pts vs all features", "icon": "check_circle",
             "accent": SURVIVED if acc >= ref_acc - 0.002 else PERISHED},
            {"label": "Features used", "count_to": len(chosen), "sub": f"{len(drop)} switched off", "icon": "toggle_on", "accent": BRASS},
        ])
        st.caption("Default hyper-parameters, development set only. Try switching off Sex and Title together to see how much the models rely on them.")
