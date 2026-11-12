"""Thin wrapper around the Three.js ``ship3d`` component with a graceful fallback."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import TARGET

try:
    from components.ship3d import ship3d as _ship3d

    SHIP3D_AVAILABLE = True
except Exception:  # pragma: no cover - component missing/broken
    _ship3d = None
    SHIP3D_AVAILABLE = False

CLASS_LABELS = {1: "First class", 2: "Second class", 3: "Third class"}


def class_zones(df: pd.DataFrame) -> list[dict]:
    zones = []
    for c in (1, 2, 3):
        sub = df[df["Pclass"] == c]
        n = int(len(sub))
        s = int(sub[TARGET].sum()) if TARGET in sub else 0
        zones.append({"id": f"class{c}", "label": CLASS_LABELS[c], "count": n, "survived": s, "rate": (s / n) if n else 0.0})
    return zones


def render_ship(mode: str = "hero", height: int = 520, key: str | None = None, **kwargs):
    if SHIP3D_AVAILABLE:
        try:
            return _ship3d(mode=mode, height=height, key=key, **kwargs)
        except Exception as exc:  # pragma: no cover
            st.caption(f"3D view unavailable: {exc}")
    st.markdown(
        f"""<div class="tml-card" style="height:{min(height, 260)}px;display:grid;place-items:center;text-align:center">
        <div><span class="msr" style="font-size:3rem;color:var(--brass)">directions_boat</span>
        <p>The interactive 3D ship could not be loaded in this browser/session.</p></div></div>""",
        unsafe_allow_html=True,
    )
    return None
