"""Acquire the Kaggle Titanic dataset into data/raw.

Usage:
    python scripts/download_data.py                 # local → Kaggle API → verified mirror
    python scripts/download_data.py --source kaggle # Kaggle API only
    python scripts/download_data.py --source mirror # verified public mirror only
"""
import argparse

import _bootstrap  # noqa: F401
from src import data_loader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "kaggle", "mirror"], default="auto")
    args = ap.parse_args()
    for line in data_loader.ensure_dataset(prefer=args.source):
        print("•", line)
    for kind in ("train", "test"):
        st = data_loader.dataset_status(kind)
        flag = "official Kaggle file ✔" if st.is_official else ("custom file" if st.exists else "MISSING")
        print(f"  {kind}.csv: {st.rows if st.exists else '-'} rows — {flag}")


if __name__ == "__main__":
    main()
