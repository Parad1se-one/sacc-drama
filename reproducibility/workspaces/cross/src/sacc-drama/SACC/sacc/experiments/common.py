"""Shared helpers for SACC DRAMA experiments."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch

from sacc.bootstrap import results_root

DEFAULT_MAX_STEPS = {
    "easy": 300,
    "hard": 500,
    "large": 800,
    "linear6": 700,
    "cross5": 800,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_max_steps(difficulty: str, explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    try:
        return int(DEFAULT_MAX_STEPS[difficulty.lower()])
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty: {difficulty}") from exc


def make_run_dir(exp_name: str, difficulty: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = results_root() / "drama" / difficulty.lower() / f"{exp_name}_{timestamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
