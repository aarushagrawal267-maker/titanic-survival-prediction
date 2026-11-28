"""Build the static, in-browser version of Titanic ML Lab (Streamlit running on Pyodide via stlite).

    python browser_demo/build.py            # writes browser_demo/dist/  (index.html, app.zip, README.md)

The output is a plain static site; it is published as a free *static* Hugging Face Space.

What it contains, and why:
* Only models listed in ``verified_models.json`` with a max probability difference below ``TOLERANCE``. That file
  records, for every committed model, the largest |difference| between its survival probabilities in the browser
  runtime and with the pinned desktop libraries, over all 1,309 Kaggle passengers (``verify_models.py``).
  XGBoost does not reproduce in the browser's older XGBoost build, so it is left out.
* No Kaggle CSVs: Kaggle's rules do not allow redistributing them. The app downloads them in the browser from the
  checksum-verified mirror, exactly as the full app does on a fresh install.
* ``TITANIC_BROWSER_DEMO=1`` turns on the browser-demo behaviour (banner, no training page, no live SHAP), and the
  3D ship component's frontend is served as static files from ``ship3d/``.
* The 3D Ship Explorer's out-of-fold predictions need refitting, which the browser cannot do with the committed
  pickles, so they are computed here with the pinned desktop libraries and shipped as ``models/browser_oof.json``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = HERE / "dist"
TOLERANCE = 1e-9

# Pyodide's own builds (scikit-learn 1.8.0, NumPy 2.4, pandas 3.0, SciPy 1.18) plus pure-Python wheels from PyPI.
REQUIREMENTS = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyyaml", "joblib", "pyodide-http",
                "plotly==7.2.0"]
ENV = {"TITANIC_BROWSER_DEMO": "1", "TITANIC_DEMO_MODE": "true"}  # index.html adds SHIP3D_COMPONENT_URL (absolute URL of ship3d/)
SKIP_PREFIXES = ("tests/", "notebooks/", "scripts/", "docs/", ".github/", "browser_demo/", "data/raw/")
SKIP_FILES = {"Dockerfile", ".dockerignore", ".gitignore", "requirements.txt", "requirements-dev.txt", "PROJECT_REPORT.md"}

SPACE_README = """---
title: Titanic ML Lab
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Titanic survival ML lab running entirely in your browser
---

Browser-only demo of [Titanic ML Lab](https://github.com/aarushagrawal267-maker/titanic-survival-prediction):
the Streamlit app runs on Pyodide (WebAssembly) via stlite. Built with `browser_demo/build.py`.
"""


def out_of_fold(keys) -> dict[str, list[float]]:
    """The 3D Ship Explorer's out-of-fold predictions, computed exactly as the page does in the full app."""
    import sys

    sys.path.insert(0, str(ROOT))
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    from src import data_loader, model_manager
    from src.train import get_cv, split_xy

    data_loader.ensure_dataset()
    X, y = split_xy(data_loader.load_train())
    return {k: [round(float(p), 6) for p in cross_val_predict(clone(model_manager.load_pipeline(k, "dev")), X, y,
                                                               cv=get_cv(), method="predict_proba")[:, 1]]
            for k in sorted(keys)}


def main() -> None:
    verified = json.loads((HERE / "verified_models.json").read_text())
    ok = {k: d for k, d in verified["max_abs_diff"].items() if d is not None and d < TOLERANCE}
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], check=True, capture_output=True, text=True).stdout.split()

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    with zipfile.ZipFile(DIST / "app.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in tracked:
            if rel.startswith(SKIP_PREFIXES) or rel in SKIP_FILES or rel.endswith(".md"):
                continue
            if rel.startswith("models/") and rel.count("/") >= 2 and rel.split("/")[1] not in ok:
                continue  # model directory that did not reproduce in the browser
            if rel == "models/registry.json":
                reg = json.loads((ROOT / rel).read_text())
                zf.writestr(rel, json.dumps({k: v for k, v in reg.items() if k in ok}, indent=2))
                continue
            zf.write(ROOT / rel, rel)
        zf.writestr("data/raw/.gitkeep", "")
        zf.writestr("models/browser_verified.json", json.dumps({**verified, "max_abs_diff": ok}, indent=2))
        zf.writestr("models/browser_oof.json", json.dumps(out_of_fold(ok)))

    html = (HERE / "index.html").read_text()
    html = html.replace("__REQUIREMENTS__", json.dumps(REQUIREMENTS)).replace("__ENV__", json.dumps(ENV))
    (DIST / "index.html").write_text(html)
    (DIST / "README.md").write_text(SPACE_README)
    # 3D ship component frontend, served as static files (see components/ship3d/__init__.py)
    shutil.copytree(ROOT / "components" / "ship3d" / "frontend", DIST / "ship3d")
    print(f"Built {DIST} with models: {', '.join(sorted(ok))}")
    print(f"Left out: {', '.join(sorted(set(verified['max_abs_diff']) - set(ok))) or 'none'}")


if __name__ == "__main__":
    main()
