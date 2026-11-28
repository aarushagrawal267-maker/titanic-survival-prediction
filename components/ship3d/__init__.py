"""ship3d — interactive 3D Titanic scene as a bidirectional Streamlit component.

The frontend is plain HTML + ES modules (Three.js vendored under
``frontend/vendor/three``), so there is no build step and nothing is fetched
from the network at runtime.

Example::

    from components.ship3d import ship3d

    event = ship3d(mode="explorer", zones=zones, passengers=rows, key="explorer")
    if event and event.get("event") == "zone_click":
        st.write("Clicked", event["zone"])
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

import streamlit.components.v1 as components

__all__ = ["ship3d"]

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
# The in-browser build (browser_demo/) serves the frontend as static files next to the page, because its
# ES-module imports cannot be resolved from Pyodide's virtual file system.
_URL = os.environ.get("SHIP3D_COMPONENT_URL")
_component = (components.declare_component("ship3d", url=_URL) if _URL
              else components.declare_component("ship3d", path=str(_FRONTEND_DIR)))

_MODES = {"hero", "explorer", "passenger"}
_COLOR_BY = {"class", "actual", "predicted", "correct"}
_TIMES = {"night", "dusk", "day"}
_CAMERAS = {"side", "bow", "stern", "top", "iceberg"}
_QUALITY = {"auto", "high", "low"}


def _scalar(v: Any) -> Any:
    """Convert numpy / pandas scalars to plain JSON-safe Python values (NaN -> None)."""
    if v is None:
        return None
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict, str, bytes)):
        try:
            v = v.item()
        except (ValueError, TypeError):
            pass
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, (int, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _clean_dict(d: Any) -> dict | None:
    if not isinstance(d, dict):
        return None
    return {str(k): _scalar(v) for k, v in d.items()}


def _clean_zones(zones: Iterable[dict] | None) -> list[dict]:
    if zones is None:
        return []
    out = []
    for z in zones:
        cz = _clean_dict(z)
        if cz:
            out.append(cz)
    return out


def _clean_passengers(passengers: Any) -> list[list]:
    """Accept a list of rows or a DataFrame with the 5 compact columns."""
    if passengers is None:
        return []
    if hasattr(passengers, "itertuples"):  # pandas DataFrame
        rows = passengers.itertuples(index=False, name=None)
    else:
        rows = passengers
    out = []
    for row in rows:
        try:
            r = list(row)[:5]
        except TypeError:
            continue
        r += [None] * (5 - len(r))
        out.append([_scalar(v) for v in r])
    return out


def ship3d(
    mode: str = "hero",
    height: int = 560,
    zones: list[dict] | None = None,
    passengers: list | None = None,
    color_by: str = "actual",
    beacon: dict | None = None,
    stats: dict | None = None,
    time_of_day: str = "night",
    auto_rotate: bool = True,
    camera: str = "side",
    quality: str = "auto",
    show_hud: bool = True,
    key: str | None = None,
) -> dict | None:
    """Render the interactive 3D Titanic.

    Args:
        mode: ``"hero"`` (cinematic banner), ``"explorer"`` (class zones,
            passenger lights, lifeboats, timeline) or ``"passenger"``
            (beacon for one passenger).
        height: Frame height in pixels.
        zones: ``[{"id": "class1"|"class2"|"class3", "label", "count",
            "survived", "rate"}]``. ``rate`` (0-1) colours the zone; it is
            derived from survived/count when missing.
        passengers: rows ``[passenger_id, pclass(1-3), actual(0/1|None),
            predicted(0/1|None), prob(0-1|None)]`` (or a DataFrame with those
            five columns in that order).
        color_by: initial colouring of passenger lights:
            ``"class" | "actual" | "predicted" | "correct"``.
        beacon: passenger mode: ``{"pclass": 1, "prob": 0.57, "label": "Your passenger"}``.
        stats: optional numbers for the HUD caption, e.g.
            ``{"total": 891, "survived": 342, "accuracy": 0.82}``.
        time_of_day: initial ``"night" | "dusk" | "day"``.
        auto_rotate: initial auto-rotate state.
        camera: initial camera preset ``"side" | "bow" | "stern" | "top" | "iceberg"``
            (in passenger mode ``"side"`` is framed as a 3/4 view).
        quality: ``"auto" | "high" | "low"`` (high = bloom + 2x pixel ratio).
        show_hud: show the on-canvas controls.
        key: Streamlit widget key (use a distinct key per instance).

    Returns:
        The most recent click event, or ``None`` before any click, e.g.
        ``{"event": "zone_click", "zone": "class1", "nonce": 1731400000000}`` or
        ``{"event": "passenger_click", "passenger_id": 12, "pclass": 3, "nonce": ...}``.
        Like every Streamlit component value it persists across reruns until
        the next click; compare ``nonce`` to detect a new click.

    HUD-only changes (colour-by, time of day, camera, timeline) happen inside
    the browser and do not trigger a Streamlit rerun. Changing an argument
    from Python updates the existing scene in place (no reload / flicker).
    """
    mode = mode if mode in _MODES else "hero"
    try:
        height = max(160, int(height))
    except (TypeError, ValueError):
        height = 560

    return _component(
        mode=mode,
        height=height,
        zones=_clean_zones(zones),
        passengers=_clean_passengers(passengers),
        color_by=color_by if color_by in _COLOR_BY else "actual",
        beacon=_clean_dict(beacon),
        stats=_clean_dict(stats),
        time_of_day=time_of_day if time_of_day in _TIMES else "night",
        auto_rotate=bool(auto_rotate),
        camera=camera if camera in _CAMERAS else "side",
        quality=quality if quality in _QUALITY else "auto",
        show_hud=bool(show_hud),
        key=key,
        default=None,
    )
