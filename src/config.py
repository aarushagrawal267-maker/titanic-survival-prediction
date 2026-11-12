"""Configuration loader.

Reads ``config.yaml`` once and exposes typed helpers. All paths are resolved
relative to the project root so the app works from any working directory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    cfg = load_config().get(section, {})
    if key is None:
        return cfg
    return cfg.get(key, default)


def path(key: str) -> Path:
    """Absolute path for a ``paths`` entry, creating directories on demand."""
    p = PROJECT_ROOT / get("paths", key)
    if key.endswith("_dir"):
        p.mkdir(parents=True, exist_ok=True)
    return p


RANDOM_STATE: int = int(get("project", "random_state", 42))
TARGET: str = get("data", "target", "Survived")
ID_COL: str = get("data", "id_column", "PassengerId")


def raw_file(kind: str) -> Path:
    """Path of the raw ``train`` or ``test`` CSV."""
    return path("raw_dir") / get("paths", f"{kind}_file")
