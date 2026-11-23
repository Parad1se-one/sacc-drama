#!/usr/bin/env python3
"""Run one frozen final-evaluation record with the common evaluator."""

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


def _validate_evaluation_payload(payload: dict, record: dict) -> None:
    if payload["start_set_id"] != record["start_set_id"]:
        raise RuntimeError("Evaluator returned an unexpected start-set identifier")
    if payload["algo"] != record["method"]:
        raise RuntimeError("Evaluator returned an unexpected method label")
    if len(payload["episodes"]) != int(record["episodes"]):
        raise RuntimeError("Evaluator returned an unexpected episode count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--records-dir", type=Path, default=Path("experiments/final_eval_run_records"))
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
    checkpoint_path = Path(record["checkpoint_path"])
    start_set_path = Path(record["start_set_path"])
    if _sha256(checkpoint_path) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after plan generation: {checkpoint_path}")
    if _sha256(start_set_path) != record["start_set_file_sha256"]:
        raise RuntimeError(f"Start set changed after plan generation: {start_set_path}")

    records_dir = args.records_dir if args.records_dir.is_absolute() else repo_root / args.records_dir
    status_path = records_dir / f"{record['run_id']}.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
        raise RuntimeError(f"Run is already complete: {status_path}")
    results_root = args.results_root if args.results_root.is_absolute() else repo_root / args.results_root
    result_path = results_root / record["experiment_name"] / "evaluations" / f"{record['run_id']}.json"

    sacc_root = repo_root / "src" / "sacc-drama" / "SACC"
    drama_root = repo_root / "src" / "sacc-drama" / "DRAMA"
    if record["method"] in {"qmix", "vdn"}:
        command = [
            sys.executable,
            "-m",
            "utils.evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(record["episodes"]),
            "--seed",
            str(record["evaluation_seed"]),
            "--difficulty",
            record["difficulty"],
            "--max-steps",
            str(record["max_steps"]),
            "--start-set",
            str(start_set_path),
            "--algo",
            record["method"],
            "--json-out",
            str(result_path),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "sacc.utils.eval",
            "--checkpoint",
            str(checkpoint_path),
            "--episodes",
            str(record["episodes"]),
            "--seed",
            str(record["evaluation_seed"]),
            "--difficulty",
            record["difficulty"],
            "--max-steps",
            str(record["max_steps"]),
            "--device",
            "cuda:0",
            "--policy-mode",
            record["policy_mode"],
            "--temperature",
            str(record.get("temperature", 0.35)),
            "--start-set",
            str(start_set_path),
            "--algo",
            record["method"],
            "--json-out",
            str(result_path),
        ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(drama_root), str(sacc_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    status = {
        **record,
        "status": "dry_run" if args.dry_run else "running",
        "started_at": _utc_now(),
        "python": sys.executable,
        "command": command,
        "cwd": str(sacc_root),
        "result_path": str(result_path),
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
                cwd=sacc_root,
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
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_evaluation_payload(payload, record)
        status.update(
            {
                "status": "completed",
                "result_sha256": _sha256(result_path),
                "actor_observation_mode": payload["actor_observation_mode"],
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        _write_json(status_path, status)
        print(json.dumps({"run_id": record["run_id"], "result": str(result_path)}))
    except BaseException:
        if status.get("status") == "running":
            status.update({"status": "failed", "finished_at": _utc_now()})
            _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
