#!/usr/bin/env python3
"""Read SACC JSON logs and print/plot training completion traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _latest_eval(eval_dir: Path) -> tuple[Path | None, dict | None]:
    def _episode_key(path: Path) -> int:
        match = re.search(r"eval_ep(\\d+)", path.stem)
        return int(match.group(1)) if match else -1

    files = sorted(eval_dir.glob("eval_ep*.json"), key=_episode_key)
    if not files:
        return None, None
    latest = files[-1]
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor SACC training traces")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--plot-out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics = _load_jsonl(run_dir / "training_metrics.jsonl")
    if not metrics:
        raise SystemExit(f"No training_metrics.jsonl found under {run_dir}")

    latest = metrics[-1]
    best = max(metrics, key=lambda row: row["completion_ratio"])
    print(json.dumps({"latest": latest, "best_train": best}, indent=2, ensure_ascii=False))

    latest_eval_path, latest_eval = _latest_eval(run_dir / "eval")
    if latest_eval is not None:
        print(json.dumps({"latest_eval_file": str(latest_eval_path), "latest_eval": latest_eval["summary"]}, indent=2, ensure_ascii=False))

    if args.plot_out:
        episodes = [row["episode"] for row in metrics]
        completion = [row["completion_ratio"] for row in metrics]
        rewards = [row["reward"] for row in metrics]
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, completion, label="train completion")
        plt.plot(episodes, rewards, label="train reward", alpha=0.5)
        plt.xlabel("Episode")
        plt.legend()
        plt.tight_layout()
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.plot_out, dpi=180)
        print(json.dumps({"plot_out": args.plot_out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
