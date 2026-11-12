# ship3d — 3D Titanic Streamlit component

A bidirectional Streamlit component (classic `declare_component(path=...)` API)
that renders a procedural RMS Titanic on an animated night ocean with Three.js.
Plain HTML + ES modules: **no npm build step, no React, no network at runtime**.

```python
from components.ship3d import ship3d

event = ship3d(mode="explorer", zones=zones, passengers=rows, stats=stats, key="explorer")
if event and event.get("event") == "zone_click":
    ...  # event = {"event": "zone_click", "zone": "class1", "nonce": 1731400000000}
```

Modes: `hero` (cinematic banner), `explorer` (class zones, passenger lights,
lifeboats, timeline of 14–15 April 1912), `passenger` (beacon for one passenger).
See the docstring of `ship3d()` for all arguments.

Return value: the latest click event, or `None`.
- `{"event": "zone_click", "zone": "class1"|"class2"|"class3", "nonce": <ms>}` (click a zone volume or its floating tag)
- `{"event": "passenger_click", "passenger_id": 12, "pclass": 3, "nonce": <ms>}` (click a passenger light)

Like any component value it persists across reruns until the next click; use
`nonce` to detect new clicks. HUD changes (colour-by, time of day, camera,
timeline, toggles) stay in the browser and do not rerun the script. Changing
arguments from Python updates the existing scene in place — the WebGL renderer
is created once per component instance (keep `key` stable).

## Layout

| File | Purpose |
| --- | --- |
| `__init__.py` | Python wrapper `ship3d(...)`, input sanitising (numpy/NaN/DataFrame) |
| `frontend/index.html` | Import map (vendored three), HUD markup, no-WebGL fallback |
| `frontend/main.js` | Streamlit protocol, bootstrap, arg diffing, HUD + pointer wiring |
| `frontend/scene.js` | Renderer, camera presets, OrbitControls, sky/stars/aurora, lights, time-of-day presets, bloom, quality, resize & pause |
| `frontend/ship.js` | Procedural ship: lofted hull, superstructure, funnels, masts, rigging, lifeboats, windows, smoke |
| `frontend/ocean.js` | Gerstner-wave ocean shader (procedural normal map, wake, foam, fog) and iceberg |
| `frontend/overlays.js` | Class zones, passenger lights, beacon, timeline + lowered lifeboats, data colours |
| `frontend/style.css` | Frosted-glass HUD in the Midnight Atlantic theme |
| `frontend/vendor/three/` | three.js r180 module build + used addons (MIT, see `LICENSE`) |
| `demo_app.py` | Dev demo with synthetic data: `streamlit run components/ship3d/demo_app.py` (`?mode=hero|explorer|passenger&quality=high|low`) |

## Notes

- Units are metres; ship-local frame: +X bow, +Y up (waterline y=0), +Z port.
- Class zones are approximate (the dataset has no cabin locations) — the HUD says so.
- Data colours: survived `#3a8ee6`, did not survive `#e8693a`, classes `#c08a1c` / `#8a7ae6` / `#1fa883`;
  survival-rate scale orange → slate `#4a5878` → blue. Legends always name the colours.
- Neutral tone mapping is used so data colours stay faithful; bloom only on `quality="high"`.
- `quality="auto"` picks low on small/low-core devices and drops to low if frames are slow.
- The render loop pauses when the tab is hidden or the frame is scrolled off-screen.
- Debug hook in the iframe: `window.__ship3d` (`setTimeline(t)`, `setColorBy(v)`, `setTod(v)`, `setCamera(v)`).
- Updating three: `npm pack three@<ver>`, copy `build/three.module.min.js`, `build/three.core.min.js`,
  `LICENSE`, and the addon files listed under `vendor/three/addons/` (keep the folder layout).
