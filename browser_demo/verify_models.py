"""Dump every committed pipeline's survival probabilities for all 1,309 Kaggle passengers (train + test).

Run it twice — once with the pinned desktop libraries and once inside the browser runtime (Pyodide, e.g. by loading
browser_demo/verify.html in a browser) — and compare the two JSON files.
The largest absolute difference per model goes into browser_demo/verified_models.json; build.py ships only models
whose predictions match.

    python browser_demo/verify_models.py <project root> <out.json>
"""
import json
import os
import sys
import warnings

root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
os.chdir(root)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

X = pd.concat([pd.read_csv("data/raw/train.csv").drop(columns=["Survived"]), pd.read_csv("data/raw/test.csv")],
              ignore_index=True)
res = {"sklearn": sklearn.__version__, "n": len(X), "pred": {}}
warnings.simplefilter("ignore")
for key in sorted(os.listdir("models")):
    if not os.path.isdir(os.path.join("models", key)):
        continue
    for which in ("pipeline", "pipeline_full"):
        try:
            pipe = joblib.load(f"models/{key}/{which}.joblib")
            res["pred"][f"{key}/{which}"] = pipe.predict_proba(X)[:, 1].tolist()
        except Exception as exc:  # recorded, so the model is left out of the browser build
            res["pred"][f"{key}/{which}"] = f"ERROR {type(exc).__name__}: {exc}"[:300]
with open(out, "w") as fh:
    json.dump(res, fh)
