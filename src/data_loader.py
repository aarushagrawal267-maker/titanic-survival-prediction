"""Dataset acquisition.

Order of preference (never fabricates real observations):

1. CSV files already present in ``data/raw``.
2. The official Kaggle API, when credentials are configured.
3. A public mirror of the identical Kaggle files (verified by SHA-256).
4. Manual upload through the app (``save_uploaded``) or manual download
   following ``MANUAL_INSTRUCTIONS``.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config

MANUAL_INSTRUCTIONS = """\
1. Sign in at https://www.kaggle.com and open https://www.kaggle.com/competitions/titanic/data
2. Accept the competition rules, then click **Download All**.
3. Unzip the archive and copy `train.csv` and `test.csv` into the project's `data/raw/` folder.
4. Re-run `python scripts/validate_data.py` (or reload the app)."""

KAGGLE_INSTRUCTIONS = """\
1. Kaggle → Account → *Create New API Token* downloads `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` (Windows: `C:\\Users\\<you>\\.kaggle\\kaggle.json`)
   or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
3. `pip install kaggle`, then run `python scripts/download_data.py --source kaggle`."""


@dataclass
class DatasetStatus:
    kind: str
    path: Path
    exists: bool
    rows: int | None = None
    sha256: str | None = None
    is_official: bool | None = None
    notes: list[str] = field(default_factory=list)


def sha256_of(path_or_bytes: Path | bytes) -> str:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return hashlib.sha256(data).hexdigest()


def dataset_status(kind: str) -> DatasetStatus:
    p = config.raw_file(kind)
    st = DatasetStatus(kind=kind, path=p, exists=p.exists())
    if st.exists:
        try:
            st.rows = int(len(pd.read_csv(p)))
            st.sha256 = sha256_of(p)
            st.is_official = st.sha256 == config.get("data", "sha256", {}).get(kind)
            if not st.is_official:
                st.notes.append("File differs from the official Kaggle file (custom or modified upload).")
        except Exception as exc:  # pragma: no cover - defensive
            st.notes.append(f"Could not read file: {exc}")
    return st


def load_raw(kind: str = "train") -> pd.DataFrame:
    """Load ``train`` or ``test`` from ``data/raw``. Raises FileNotFoundError with guidance."""
    p = config.raw_file(kind)
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} not found in {p.parent}. Run `python scripts/download_data.py` "
            "or upload the file on the Dataset Explorer page."
        )
    return pd.read_csv(p)


def load_train() -> pd.DataFrame:
    return load_raw("train")


def load_test() -> pd.DataFrame:
    return load_raw("test")


def kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download_from_kaggle() -> tuple[bool, str]:
    """Download via the Kaggle CLI. Returns (ok, message)."""
    if not kaggle_credentials_available():
        return False, "Kaggle credentials not found."
    if shutil.which("kaggle") is None:
        return False, "The `kaggle` package is not installed (pip install kaggle)."
    comp = config.get("data", "kaggle_competition", "titanic")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["kaggle", "competitions", "download", "-c", comp, "-p", tmp],
                check=True, capture_output=True, text=True, timeout=180,
            )
        except subprocess.CalledProcessError as exc:
            return False, f"Kaggle download failed: {exc.stderr.strip() or exc}"
        except Exception as exc:
            return False, f"Kaggle download failed: {exc}"
        zips = list(Path(tmp).glob("*.zip"))
        if not zips:
            return False, "Kaggle returned no archive."
        with zipfile.ZipFile(zips[0]) as zf:
            for kind in ("train", "test"):
                name = config.get("paths", f"{kind}_file")
                with zf.open(name) as src, open(config.raw_file(kind), "wb") as dst:
                    dst.write(src.read())
    return True, "Downloaded train.csv and test.csv from Kaggle."


def download_from_mirror(kind: str, timeout: int = 30) -> tuple[bool, str]:
    """Download one file from the public mirror and verify its SHA-256 against Kaggle's."""
    url = config.get("data", "mirrors", {}).get(kind)
    expected = config.get("data", "sha256", {}).get(kind)
    if not url:
        return False, "No mirror configured."
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            data = resp.read()
    except Exception as exc:
        return False, f"Mirror download failed: {exc}"
    digest = sha256_of(data)
    if expected and digest != expected:
        return False, "Mirror file does not match the official Kaggle checksum — not saved."
    config.raw_file(kind).write_bytes(data)
    return True, f"Downloaded {kind}.csv from mirror (checksum verified)."


def ensure_dataset(prefer: str = "auto") -> list[str]:
    """Make sure train/test exist locally. Returns a log of what happened."""
    log: list[str] = []
    missing = [k for k in ("train", "test") if not config.raw_file(k).exists()]
    if not missing:
        return ["Dataset already present locally."]
    if prefer in ("auto", "kaggle"):
        ok, msg = download_from_kaggle()
        log.append(msg)
        missing = [k for k in ("train", "test") if not config.raw_file(k).exists()]
        if ok and not missing:
            return log
    if prefer in ("auto", "mirror"):
        for kind in missing:
            ok, msg = download_from_mirror(kind)
            log.append(msg)
    if any(not config.raw_file(k).exists() for k in ("train", "test")):
        log.append("Dataset still missing. Manual steps:\n" + MANUAL_INSTRUCTIONS)
    return log


def save_uploaded(content: bytes, kind: str) -> tuple[bool, list[str]]:
    """Validate an uploaded CSV and store it as the raw train/test file."""
    from src.data_validation import validate_dataframe

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return False, [f"Could not parse CSV: {exc}"]
    report = validate_dataframe(df, kind=kind)
    if not report.is_valid:
        return False, report.errors
    config.raw_file(kind).write_bytes(content)
    return True, [f"Saved {kind}.csv ({len(df)} rows)."] + report.warnings
