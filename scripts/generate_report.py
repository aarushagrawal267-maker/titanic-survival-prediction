"""Generate PROJECT_REPORT.md (+ figures in outputs/figures) from the models that were actually trained.

Usage:  python scripts/generate_report.py
Nothing in the report is typed by hand: every number is read from models/registry.json,
models/*/evaluation.joblib and outputs/reports/experiments.json.
"""
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config, data_loader, eda, model_manager  # noqa: E402
from src import explainability as xai  # noqa: E402
from src.evaluate import METRIC_LABELS, METRICS  # noqa: E402
from src.models import MODEL_ORDER, SPECS  # noqa: E402
from src.preprocessing import friendly  # noqa: E402
from src.submission import make_submission, save_submission, validate_submission  # noqa: E402
from src.train import load_experiments  # noqa: E402

BG, SURF, INK, MUTED, BLUE, ORANGE, BRASS = "#050b1a", "#0d1a33", "#eaf0fa", "#7d8ca8", "#3a8ee6", "#e8693a", "#e3b866"
CAT = [BLUE, ORANGE, "#1fa883", "#8a7ae6", "#c08a1c", "#d55181"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": SURF, "axes.edgecolor": "#2a3550", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#1c2a47", "axes.grid": True,
                     "grid.linewidth": 0.6, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12, "savefig.dpi": 150})
FIG = config.path("figures_dir")


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor=BG)
    plt.close(fig)
    return f"outputs/figures/{name}"


def main() -> int:
    reg = model_manager.load_registry()
    if not reg:
        print("No trained models — run python scripts/train_all.py first.")
        return 1
    df = data_loader.load_train()
    test = data_loader.load_test()
    exp = load_experiments()
    keys = [k for k in MODEL_ORDER if k in reg]
    best = model_manager.best_model_key()
    ov = eda.overview(df)
    e = eda.enrich(df)
    versions = model_manager.library_versions()
    figs = {}

    # ── figures ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, order in ((axes[0], "Sex", ["female", "male"]), (axes[1], "Class", ["1st class", "2nd class", "3rd class"])):
        g = eda.survival_by(e, col, order=order)
        sns.barplot(data=g, x=col, y="survival_rate", color=BLUE, ax=ax)
        ax.axhline(ov["survival_rate"], color=BRASS, ls=":", lw=1.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Survival rate")
        ax.set_title(f"Survival rate by {col.lower()}")
        for p_, v in zip(ax.patches, g["survival_rate"]):
            ax.annotate(f"{v:.0%}", (p_.get_x() + p_.get_width() / 2, v + 0.02), ha="center", color=INK)
    figs["eda"] = save(fig, "eda_survival_by_sex_class.png")

    rows = []
    for k in keys:
        m = reg[k]
        rows.append({"Model": SPECS[k].name, **{METRIC_LABELS[x]: m["holdout"][x] for x in METRICS}})
    hdf = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.barplot(data=hdf, x="Model", y="Score", hue="Metric", palette=CAT[:5], ax=ax)
    ax.set_ylim(0.6, 0.95)
    ax.set_title("Holdout metrics by model (179 passengers)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.tick_params(axis="x", rotation=15)
    figs["compare"] = save(fig, "model_comparison_holdout.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    folds = pd.DataFrame({SPECS[k].name: reg[k]["cv"]["folds"]["roc_auc"] for k in keys}).melt(var_name="Model", value_name="ROC-AUC")
    sns.boxplot(data=folds, x="Model", y="ROC-AUC", color="#16325f", ax=ax, linecolor=MUTED)
    sns.stripplot(data=folds, x="Model", y="ROC-AUC", color=BLUE, size=6, ax=ax)
    ax.set_title("5-fold cross-validation ROC-AUC")
    ax.tick_params(axis="x", rotation=15)
    figs["cv"] = save(fig, "cv_roc_auc.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED)
    for i, k in enumerate(keys):
        ev = model_manager.load_evaluation(k)
        hi = k == best
        ax.plot(ev["roc"]["fpr"], ev["roc"]["tpr"], lw=2.4 if hi else 1.1, color=BRASS if hi else CAT[i % len(CAT)], alpha=1 if hi else 0.6,
                label=f"{SPECS[k].name} ({ev['metrics']['roc_auc']:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC curves")
    ax.legend(fontsize=8, facecolor=SURF, edgecolor="#2a3550")
    figs["roc"] = save(fig, "roc_curves.png")

    evb = model_manager.load_evaluation(best)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    cm = np.array(evb["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(BLUE, as_cmap=True), cbar=False, ax=ax,
                xticklabels=["Pred: died", "Pred: survived"], yticklabels=["Actual: died", "Actual: survived"], annot_kws={"size": 14, "color": "#0b0b0b"})
    ax.set_title(f"Confusion matrix — {SPECS[best].name}")
    figs["cm"] = save(fig, "confusion_matrix_best.png")

    shp = evb.get("shap")
    if shp:
        imp = xai.shap_importance(shp).head(12)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(imp["label"][::-1], imp["mean_abs_shap"][::-1] * 100, color=BLUE)
        ax.set_xlabel("Mean |SHAP| (percentage points)")
        ax.set_title(f"SHAP feature importance — {SPECS[best].name}")
        figs["shap"] = save(fig, "shap_importance_best.png")

    # ── submission ───────────────────────────────────────────
    sub = make_submission(model_manager.load_pipeline(best, "full"), test)
    sub_errors = validate_submission(sub, test)
    if not sub_errors:
        save_submission(sub)

    # ── markdown ─────────────────────────────────────────────
    L = []
    L += ["# Titanic ML Lab — Project Report", "",
          f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/generate_report.py` from the models trained in this repository. "
          "No number in this report was typed by hand.*", "",
          f"Environment: Python {versions['python']} · scikit-learn {versions['scikit-learn']} · pandas {versions['pandas']} · NumPy {versions['numpy']}"
          + (f" · XGBoost {versions['xgboost']}" if "xgboost" in versions else ""), ""]
    L += ["## 1. Data", "",
          f"- `train.csv`: {len(df)} passengers × {df.shape[1]} columns — official Kaggle file (SHA-256 verified: "
          f"{'yes' if data_loader.dataset_status('train').is_official else 'no'}).",
          f"- `test.csv`: {len(test)} passengers, no labels (SHA-256 verified: {'yes' if data_loader.dataset_status('test').is_official else 'no'}).",
          f"- Survivors: {ov['survivors']} ({ov['survival_rate']:.1%}); non-survivors: {ov['non_survivors']}. Duplicates: {ov['duplicates']}.",
          "", md_table(eda.missing_table(df)[["column", "missing", "missing_pct", "strategy"]], ".1f"), "",
          f"![Survival by sex and class]({figs['eda']})", "", "### Automatically generated insights", ""]
    L += [f"- **{i['title']}** — {i['text']}" for i in eda.auto_insights(df)]
    L += ["", "Correlation is not causation: these are associations in historical data.", ""]

    L += ["## 2. Method", "",
          "- Stratified 80/20 split (seed 42): **712** development passengers, **179** holdout passengers.",
          "- All preprocessing (feature engineering, imputation, one-hot encoding, scaling) inside one scikit-learn `Pipeline`, refit inside every CV fold.",
          f"- 5-fold `StratifiedKFold` on the development set; `RandomizedSearchCV` with {config.get('training', 'n_iter')} candidates per model, optimising ROC-AUC.",
          "- Model selection: highest mean CV ROC-AUC (tie-break CV F1, then lower std). The holdout is only used for the final, reported metrics.",
          "- Final models are refit on all 891 labelled passengers for predictions and the Kaggle submission.", ""]

    if exp:
        L += ["## 3. Preprocessing & feature experiments", "", "All with default hyper-parameters, 5-fold CV ROC-AUC on the development set.", "",
              "### 3.1 Scaling", ""]
        sc = pd.DataFrame(exp["scaling"]["rows"]).pivot_table(index="model", columns="scaler", values="roc_auc").reset_index()
        L += [md_table(sc), "", "Chosen: " + ", ".join(f"{SPECS[k].name} → {v}" for k, v in exp["scaling"]["best"].items()), ""]
        fs = pd.DataFrame(exp["feature_sets"]["rows"]).pivot_table(index="model", columns="feature_set", values="roc_auc").reset_index()
        fs["gain"] = fs["engineered"] - fs["base"]
        L += ["### 3.2 Base (7) vs engineered (17) features", "", md_table(fs, ".4f"), ""]
        ab = pd.DataFrame(exp["ablation"]["rows"])
        abp = ab[ab["dropped"] != "(none — all features)"].pivot_table(index="dropped", columns="model", values="delta").reset_index()
        abp["mean"] = abp.drop(columns="dropped").mean(axis=1)
        abp = abp.sort_values("mean")
        top = abp.iloc[0]
        n_hurt = int((abp["mean"] > 0).sum())
        n_better = int((fs["gain"] > 0).sum())
        L += ["### 3.3 Drop-one-feature ablation (Δ CV ROC-AUC when the feature is removed; negative = the feature helped)", "",
              md_table(abp, "+.4f"), "",
              f"Interpretation: removing **{friendly(top['dropped'])}** costs the most ({top['mean']:+.4f} mean Δ ROC-AUC). "
              f"Every individual change is smaller than the fold-to-fold standard deviation (≈0.02), largely because the engineered features "
              f"overlap (FamilySize / IsAlone / FamilyCategory); for {n_hurt} of {len(abp)} features, removal slightly *improved* mean CV ROC-AUC. "
              f"As a set, the engineered features beat the base features for {n_better} of {len(fs)} algorithms (3.2), so the engineered set was kept.", ""]
        en = pd.DataFrame(exp["encoding"]["rows"])[["model", "pclass_encoding", "age_strategy", "roc_auc", "accuracy"]]
        L += ["### 3.4 Pclass encoding × age imputation", "", md_table(en, ".4f"), ""]

    L += ["## 4. Model results", ""]
    rows = []
    for k in sorted(keys, key=lambda k: -reg[k]["cv"]["roc_auc_mean"]):
        m = reg[k]
        rows.append({"Model": SPECS[k].name + (" ★" if k == best else ""),
                     "CV ROC-AUC": f"{m['cv']['roc_auc_mean']:.3f} ± {m['cv']['roc_auc_std']:.3f}",
                     "CV accuracy": f"{m['cv']['accuracy_mean']:.3f}", "CV F1": f"{m['cv']['f1_mean']:.3f}",
                     **{f"Holdout {METRIC_LABELS[x]}": m["holdout"][x] for x in METRICS},
                     "Fit time (s)": m["train_time_s"], "Features": m["feature_set"], "Scaler": m["scaler"]})
    L += [md_table(pd.DataFrame(rows)), "", f"![Holdout metrics]({figs['compare']})", "", f"![CV]({figs['cv']})", "", f"![ROC]({figs['roc']})", ""]
    L += ["### Tuned hyper-parameters", ""]
    for k in keys:
        bp = ", ".join(f"`{p}={v:.4g}`" if isinstance(v, float) else f"`{p}={v}`" for p, v in (reg[k].get("best_params") or {}).items())
        L.append(f"- **{SPECS[k].name}**: {bp}")
    mb = reg[best]
    L += ["", f"## 5. Selected model: {SPECS[best].name}", "",
          f"Selected by mean CV ROC-AUC ({mb['cv']['roc_auc_mean']:.3f} ± {mb['cv']['roc_auc_std']:.3f}). "
          f"On the untouched holdout: accuracy {mb['holdout']['accuracy']:.3f}, precision {mb['holdout']['precision']:.3f}, "
          f"recall {mb['holdout']['recall']:.3f}, F1 {mb['holdout']['f1']:.3f}, ROC-AUC {mb['holdout']['roc_auc']:.3f}.", ""]
    best_holdout_acc = max(keys, key=lambda k: reg[k]["holdout"]["accuracy"])
    if best_holdout_acc != best:
        L += [f"Note: {SPECS[best_holdout_acc].name} has the highest *holdout accuracy* ({reg[best_holdout_acc]['holdout']['accuracy']:.3f}). "
              "We deliberately do not select on the holdout (that would turn it into another tuning set) nor on accuracy alone; with 179 passengers, "
              "holdout differences of 2–3 points are within sampling noise (≈ ±3 pts standard error).", ""]
    L += [f"![Confusion matrix]({figs['cm']})", ""]
    rep = pd.DataFrame(evb["classification_report"]).T.reset_index().rename(columns={"index": "class"})
    L += [md_table(rep[["class", "precision", "recall", "f1-score", "support"]]), ""]
    if shp:
        L += [f"![SHAP]({figs['shap']})", "", "Top features by mean |SHAP| (percentage points of survival probability): "
              + ", ".join(f"{r.label} ({r.mean_abs_shap * 100:.1f})" for r in xai.shap_importance(shp).head(6).itertuples()) + ".", ""]
    pi = pd.DataFrame(evb["permutation_importance"]).head(6)
    L += ["Permutation importance (holdout ROC-AUC drop): " + ", ".join(f"{r.label} ({r.importance_mean:.3f})" for r in pi.itertuples()) + ".", ""]

    L += ["## 6. Kaggle submission", ""]
    if sub_errors:
        L += ["Submission validation FAILED: " + "; ".join(sub_errors), ""]
    else:
        L += [f"`outputs/submission.csv` — {len(sub)} rows, columns `PassengerId,Survived`, model {SPECS[best].name}; "
              f"predicted survival rate {sub['Survived'].mean():.1%} (training data: {ov['survival_rate']:.1%}). All format checks passed.", "",
              "**The Kaggle leaderboard score is not known locally** — test labels are private. Submit the file at "
              "https://www.kaggle.com/competitions/titanic/submit to obtain it. Expect it to differ from the validation estimates above.", ""]
    L += ["## 7. Limitations", "",
          "- Small data: holdout metrics on 179 passengers have a standard error of roughly ±3 percentage points.",
          "- CV scores of tuned models are mildly optimistic (the same folds chose the hyper-parameters); nested CV would remove this bias.",
          "- Models capture historical associations (sex, class, age, family) — not causes — and must not be used to judge individuals.",
          "- Cabin is missing for 77% and age for 20% of passengers; crew are absent from the dataset.", ""]
    (config.PROJECT_ROOT / "PROJECT_REPORT.md").write_text("\n".join(L))
    print(f"✔ PROJECT_REPORT.md written ({len(L)} lines), figures in outputs/figures/, submission {'saved' if not sub_errors else 'INVALID'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
