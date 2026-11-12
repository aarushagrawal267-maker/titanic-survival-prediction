"""Create outputs/submission.csv (PassengerId,Survived) from a trained model.

Usage:
    python scripts/make_submission.py              # best model by CV ROC-AUC
    python scripts/make_submission.py --model svm
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from src import model_manager
from src.data_loader import load_test
from src.submission import make_submission, save_submission, validate_submission


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    key = args.model or model_manager.best_model_key()
    if not key:
        print("No trained models. Run: python scripts/train_all.py")
        return 1
    test = load_test()
    sub = make_submission(model_manager.load_pipeline(key, "full"), test)
    errors = validate_submission(sub, test)
    if errors:
        print("Submission INVALID:\n  " + "\n  ".join(errors))
        return 1
    path = save_submission(sub)
    print(f"✔ {path} — {len(sub)} rows, model '{key}', predicted survival rate {sub['Survived'].mean():.1%}")
    print("Note: Kaggle test labels are not public; the real score is only available by submitting on kaggle.com.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
