"""Cached data/model access shared by every page, plus onboarding guards."""
from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from src import config, data_loader, model_manager
from src.models import SPECS


def _mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def train_mtime() -> float:
    return _mtime(config.raw_file("train"))


def registry_mtime() -> float:
    return _mtime(model_manager.models_dir() / "registry.json")


@st.cache_data(show_spinner=False)
def _load_raw(kind: str, mtime: float) -> pd.DataFrame:
    return data_loader.load_raw(kind)


def get_train() -> pd.DataFrame:
    return _load_raw("train", train_mtime()).copy()


def get_test() -> pd.DataFrame:
    return _load_raw("test", _mtime(config.raw_file("test"))).copy()


def data_ready() -> bool:
    return config.raw_file("train").exists()


@st.cache_data(show_spinner=False)
def _registry(mtime: float) -> dict:
    return model_manager.load_registry()


def get_registry() -> dict:
    return _registry(registry_mtime())


@st.cache_resource(show_spinner=False, max_entries=20)
def _pipeline(key: str, which: str, mtime: float):
    return model_manager.load_pipeline(key, which)


def get_pipeline(key: str, which: str = "full"):
    return _pipeline(key, which, registry_mtime())


@st.cache_data(show_spinner=False, max_entries=20)
def _evaluation(key: str, mtime: float) -> dict:
    return model_manager.load_evaluation(key)


def get_evaluation(key: str) -> dict:
    return _evaluation(key, registry_mtime())


def best_key() -> str | None:
    return model_manager.best_model_key() if get_registry() else None


def active_model() -> str | None:
    """Model chosen in the sidebar (defaults to the best model)."""
    reg = get_registry()
    if not reg:
        return None
    key = st.session_state.get("active_model")
    return key if key in reg else best_key()


def model_name(key: str | None) -> str:
    if not key:
        return "—"
    return SPECS[key].name if key in SPECS else key


def refresh_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


# ── deployment helpers ──────────────────────────────────────────
def demo_mode() -> bool:
    """Public-demo mode: visitors can train and experiment, but nothing shared is overwritten.

    Enable with the environment variable ``TITANIC_DEMO_MODE=true`` or the Streamlit secret
    ``TITANIC_DEMO_MODE = "true"`` (e.g. in Streamlit Community Cloud → Settings → Secrets).
    """
    val = os.environ.get("TITANIC_DEMO_MODE")
    if val is None:
        try:
            val = st.secrets.get("TITANIC_DEMO_MODE")
        except Exception:  # no secrets file
            val = None
    return str(val).strip().lower() in ("1", "true", "yes", "on")


@st.cache_resource(show_spinner=False, ttl=600)
def _download_dataset_once() -> tuple[str, ...]:
    return tuple(data_loader.ensure_dataset())


def ensure_dataset_present() -> None:
    """Fetch the Kaggle CSVs on first launch.

    They are not stored in the repository (Kaggle's competition rules do not allow redistributing them), so a fresh
    server downloads them once: Kaggle API if configured, otherwise a public mirror whose files must match Kaggle's
    SHA-256 checksums. At most one attempt every 10 minutes; if it fails, the pages show the manual options.
    """
    if config.raw_file("train").exists() and config.raw_file("test").exists():
        return
    with st.spinner("Downloading the Kaggle Titanic dataset (first launch only)…"):
        _download_dataset_once()


@st.cache_data(show_spinner=False)
def _stale(mtime: float) -> list:
    return model_manager.stale_models()


def ensure_models_ready() -> None:
    """Retrain models that were saved with different library versions (runs once per server).

    Uses a lock file so concurrent visitors don't all retrain at the same time.
    """
    if not data_ready():
        return
    stale = _stale(registry_mtime())
    if not stale:
        return
    lock = model_manager.models_dir() / ".retrain.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 1200:
        st.info("The models are being re-trained for this server's library versions — please refresh in a minute.", icon=":material/hourglass_top:")
        st.stop()
    from src.train import train_model

    lock.write_text(str(time.time()))
    try:
        df = get_train()
        with st.status("Preparing the models for this server (one-time, ~1–3 min)…", expanded=True) as status:
            bar = st.progress(0.0)
            for i, (key, reason) in enumerate(stale):
                st.write(f"**{model_name(key)}** — {reason}")
                train_model(df, key, tune=True, n_iter=10, source="auto-retrain",
                            progress=lambda f, m, i=i: bar.progress(min(1.0, (i + f) / len(stale)), text=f"{model_name(key)}: {m}"))
            status.update(label="Models ready", state="complete")
    finally:
        lock.unlink(missing_ok=True)
    refresh_caches()
    st.rerun()


# ── onboarding guards ───────────────────────────────────────────
def require_data() -> pd.DataFrame:
    if data_ready():
        return get_train()
    from src.ui.components import note, section

    section("Dataset needed", icon_name="dataset")
    note("The Kaggle Titanic <code>train.csv</code> was not found in <code>data/raw/</code>. Choose one of the options below.", "warning")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("**Automatic download**  \nTries the Kaggle API (if configured), then a public mirror of the identical files verified by SHA-256 checksum.")
        if st.button("Download dataset", type="primary", key="dl_data"):
            with st.spinner("Downloading…"):
                log = data_loader.ensure_dataset()
            for line in log:
                st.write("•", line)
            refresh_caches()
            st.rerun()
    with c2, st.container(border=True):
        up = st.file_uploader("…or upload train.csv", type="csv", key="up_train")
        if up is not None:
            ok, msgs = data_loader.save_uploaded(up.getvalue(), "train")
            (st.success if ok else st.error)("\n".join(msgs))
            if ok:
                refresh_caches()
                st.rerun()
    with st.expander("Manual download instructions"):
        st.markdown(data_loader.MANUAL_INSTRUCTIONS)
    st.stop()


def train_all_models(df: pd.DataFrame, quick: bool = False) -> None:
    from src.models import MODEL_ORDER
    from src.train import run_experiments, train_model

    with st.status("Training the model zoo…", expanded=True) as status:
        bar = st.progress(0.0)
        if not quick:
            st.write("Running preprocessing & feature experiments…")
            run_experiments(df, log=lambda m: st.write("▸", m))
        n = len(MODEL_ORDER)
        for i, key in enumerate(MODEL_ORDER):
            t = time.time()
            st.write(f"**{SPECS[key].name}** — tuning with RandomizedSearchCV…")
            res = train_model(df, key, tune=True, n_iter=8 if quick else None,
                              progress=lambda f, m, i=i: bar.progress(min(1.0, (i + f) / n), text=f"{SPECS[key].name}: {m}"))
            st.write(f"✔ CV ROC-AUC {res['meta']['cv']['roc_auc_mean']:.3f} · holdout accuracy {res['meta']['holdout']['accuracy']:.3f} · {time.time() - t:.0f}s")
        status.update(label="All models trained", state="complete")
    refresh_caches()


def require_models() -> str:
    """Return the active model key, or show a one-click training panel and stop."""
    key = active_model()
    if key:
        return key
    from src.ui.components import note, section

    section("No trained models yet", icon_name="model_training")
    note("Models are trained locally (they are not stored in the repository). Train the full zoo once — "
         "it runs the experiments and tunes all algorithms; it takes a few minutes on a laptop.", "info")
    c1, c2 = st.columns(2)
    full = c1.button("Train all models (full)", type="primary", key="train_all_full")
    quick = c2.button("Quick train (≈1 min)", key="train_all_quick")
    if full or quick:
        train_all_models(require_data(), quick=quick)
        st.rerun()
    st.caption("Or from a terminal: `python scripts/train_all.py`")
    st.stop()
