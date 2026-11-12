"""Validate data/raw/train.csv and data/raw/test.csv and print a report.

Exit code is 1 if either file is missing or invalid.
"""
import sys

import _bootstrap  # noqa: F401
from src import data_loader
from src.data_validation import validate_dataframe


def main() -> int:
    ok = True
    for kind in ("train", "test"):
        try:
            df = data_loader.load_raw(kind)
        except FileNotFoundError as exc:
            print(f"✘ {exc}")
            ok = False
            continue
        rep = validate_dataframe(df, kind=kind)
        status = data_loader.dataset_status(kind)
        print(f"\n══ {kind}.csv — {rep.n_rows} rows × {rep.n_cols} columns — "
              f"{'VALID' if rep.is_valid else 'INVALID'}{' (official Kaggle checksum)' if status.is_official else ''}")
        for label, items in (("ERROR", rep.errors), ("WARN", rep.warnings), ("INFO", rep.info)):
            for item in items:
                print(f"  [{label}] {item}")
        print(f"  duplicates: {rep.duplicate_rows} rows, {rep.duplicate_ids} ids")
        print(rep.summary[["column", "dtype", "missing", "missing_pct", "unique"]].to_string(index=False))
        ok &= rep.is_valid
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
