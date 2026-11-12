"""Run all experiments, then tune, evaluate and save every model.

Usage:
    python scripts/train_all.py                 # experiments + all models (a few minutes)
    python scripts/train_all.py --quick         # fewer search iterations, no experiments
    python scripts/train_all.py --models random_forest svm
"""
import argparse
import time

import _bootstrap  # noqa: F401
from src import model_manager
from src.data_loader import ensure_dataset, load_train
from src.models import MODEL_ORDER, SPECS
from src.train import run_experiments, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=MODEL_ORDER, help=f"subset of: {', '.join(MODEL_ORDER)}")
    ap.add_argument("--quick", action="store_true", help="8 search iterations and skip experiments")
    ap.add_argument("--n-iter", type=int, default=None)
    ap.add_argument("--skip-experiments", action="store_true")
    args = ap.parse_args()

    for line in ensure_dataset():
        print("•", line)
    df = load_train()
    t0 = time.time()
    if not (args.quick or args.skip_experiments):
        run_experiments(df, log=lambda m: print("▸", m))
    n_iter = args.n_iter or (8 if args.quick else None)
    for key in args.models:
        if key not in SPECS:
            print(f"  skipping unknown/unavailable model: {key}")
            continue
        t = time.time()
        print(f"\n━━ {SPECS[key].name} ━━")
        res = train_model(df, key, tune=True, n_iter=n_iter, source="cli",
                          progress=lambda f, m: print(f"   [{f:4.0%}] {m}"))
        cv, ho = res["meta"]["cv"], res["meta"]["holdout"]
        print(f"   CV ROC-AUC {cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f} | holdout acc {ho['accuracy']:.3f} "
              f"F1 {ho['f1']:.3f} AUC {ho['roc_auc']:.3f} | {time.time() - t:.0f}s")
    best = model_manager.best_model_key()
    print(f"\nBest model by CV ROC-AUC: {SPECS[best].name if best else '—'}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
