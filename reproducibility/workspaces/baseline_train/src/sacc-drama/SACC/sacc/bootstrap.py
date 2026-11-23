"""Bootstrap helpers for importing the local DRAMA benchmark repo."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_relative_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    return candidate


def drama_root() -> Path:
    configured = os.environ.get("DRAMA_ROOT")
    if configured:
        return _repo_relative_path(configured)
    return repo_root().parent / "DRAMA"


def results_root() -> Path:
    return _repo_relative_path(os.environ.get("SACC_RESULTS_DIR", "results"))


def ensure_numpy_compat_aliases() -> None:
    alias_pairs = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for alias, target in alias_pairs.items():
        if alias not in np.__dict__:
            setattr(np, alias, target)


def ensure_drama_on_path() -> Path:
    ensure_numpy_compat_aliases()
    root = repo_root()
    drama = drama_root()
    if not drama.exists():
        raise FileNotFoundError(
            f"DRAMA repo not found at {drama}; set DRAMA_ROOT or place DRAMA beside SACC."
        )
    for candidate in (str(drama), str(root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return drama
