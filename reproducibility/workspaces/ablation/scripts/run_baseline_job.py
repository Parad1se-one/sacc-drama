#!/usr/bin/env python3
"""Run one frozen missing-baseline job in an isolated result workspace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--records-dir", type=Path, default=Path("experiments/baseline_run_records"))
    parser.add_argument("--results-root", type=Path, default=Path("results/rebuttal"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [record for record in plan["records"] if record["run_id"] == args.run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one plan record for {args.run_id!r}, found {len(matches)}")
    record = dict(matches[0])

    current_source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src/sacc-drama"], cwd=repo_root, text=True
    ).strip()
    if current_source_tree != record["source_tree"]:
        raise RuntimeError(
            f"Plan requires source tree {record['source_tree']}, current tree is {current_source_tree}"
        )
    spec_path = repo_root / record["spec_path"]
    if _sha256(spec_path) != record["spec_sha256"]:
        raise RuntimeError(f"Baseline specification changed after plan generation: {spec_path}")

    records_dir = args.records_dir if args.records_dir.is_absolute() else repo_root / args.records_dir
    status_path = records_dir / f"{record['run_id']}.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
        raise RuntimeError(f"Run is already complete: {status_path}")

    results_root = args.results_root if args.results_root.is_absolute() else repo_root / args.results_root
    workspace = results_root / record["experiment_name"] / "workspaces" / record["run_id"]
    workspace.mkdir(parents=True, exist_ok=True)
    drama_root = repo_root / "src" / "sacc-drama" / "DRAMA"
    command = [sys.executable, str(drama_root / "train_baseline.py"), *record["argv"]]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(drama_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    status = {
        **record,
        "status": "dry_run" if args.dry_run else "running",
        "started_at": _utc_now(),
        "python": sys.executable,
        "command": command,
        "cwd": str(workspace),
    }
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return
    _write_json(status_path, status)

    stdout_path = records_dir / "logs" / f"{record['run_id']}.stdout.log"
    stderr_path = records_dir / "logs" / f"{record['run_id']}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        status.update({"finished_at": _utc_now(), "returncode": completed.returncode})
        if completed.returncode != 0:
            status["status"] = "failed"
            _write_json(status_path, status)
            raise SystemExit(completed.returncode)

        pattern = f"{record['run_id']}_*_best_by_completion.pth"
        checkpoints = list((workspace / "checkpoints" / record["algo"]).glob(pattern))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one selected checkpoint matching {pattern}, found {len(checkpoints)}")
        checkpoint = checkpoints[0]
        status.update(
            {
                "status": "completed",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        _write_json(status_path, status)
        print(json.dumps({"run_id": record["run_id"], "checkpoint": str(checkpoint)}))
    except BaseException:
        if status.get("status") == "running":
            status.update({"status": "failed", "finished_at": _utc_now()})
            _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
