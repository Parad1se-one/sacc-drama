#!/usr/bin/env python3
"""Run exactly one planned SACC ablation job and preserve its provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT_TOKEN = "{repo_root}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_record(plan: dict, *, index: int | None, run_id: str | None) -> dict:
    if (index is None) == (run_id is None):
        raise ValueError("Provide exactly one of --index or --run-id")
    if index is not None:
        try:
            return dict(plan["records"][index])
        except IndexError as exc:
            raise ValueError(f"Plan has no run index {index}") from exc
    matches = [record for record in plan["records"] if record["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one plan record for {run_id!r}, found {len(matches)}")
    return dict(matches[0])


def _resolve_argv(argv: list[str], repo_root: Path) -> list[str]:
    root = str(repo_root)
    return [value.replace(REPO_ROOT_TOKEN, root) for value in argv]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results-root", type=Path, default=Path("results/rebuttal"))
    parser.add_argument("--records-dir", type=Path, default=Path("experiments/run_records"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    record = _select_record(plan, index=args.index, run_id=args.run_id)

    current_source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src/sacc-drama"], cwd=repo_root, text=True
    ).strip()
    if current_source_tree != record["source_tree"]:
        raise RuntimeError(
            f"Plan requires source tree {record['source_tree']}, current tree is {current_source_tree}"
        )
    spec_path = repo_root / record["spec_path"]
    if _sha256(spec_path) != record["spec_sha256"]:
        raise RuntimeError(f"Experiment specification changed after plan generation: {spec_path}")
    for key in ("selection_start_set", "final_random_start_set"):
        manifest = record[key]
        manifest_path = repo_root / manifest["path"]
        if _sha256(manifest_path) != manifest["file_sha256"]:
            raise RuntimeError(f"Start-set file changed after plan generation: {manifest_path}")

    records_dir = args.records_dir if args.records_dir.is_absolute() else repo_root / args.records_dir
    status_path = records_dir / f"{record['run_id']}.json"
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            raise RuntimeError(f"Run is already complete: {status_path}")

    results_root = args.results_root if args.results_root.is_absolute() else repo_root / args.results_root
    sacc_root = repo_root / "src" / "sacc-drama" / "SACC"
    drama_root = repo_root / "src" / "sacc-drama" / "DRAMA"
    command = [sys.executable, *_resolve_argv(record["argv"], repo_root)]
    environment = os.environ.copy()
    environment["DRAMA_ROOT"] = str(drama_root)
    environment["SACC_RESULTS_DIR"] = str(results_root / record["experiment_name"])
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
        "results_root": environment["SACC_RESULTS_DIR"],
    }
    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return

    _write_json(status_path, status)

    log_dir = records_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{record['run_id']}.stdout.log"
    stderr_path = log_dir / f"{record['run_id']}.stderr.log"
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
        status["finished_at"] = _utc_now()
        status["returncode"] = completed.returncode
        status["stdout_log"] = str(stdout_path)
        status["stderr_log"] = str(stderr_path)
        if completed.returncode != 0:
            status["status"] = "failed"
            _write_json(status_path, status)
            raise SystemExit(completed.returncode)

        last_line = ""
        for line in stdout_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_line = line.strip()
        try:
            trainer_output = json.loads(last_line)
        except json.JSONDecodeError as exc:
            status["status"] = "failed"
            status["error"] = "Trainer finished but did not emit a JSON run_dir record"
            _write_json(status_path, status)
            raise RuntimeError(status["error"]) from exc

        status["status"] = "completed"
        status["run_dir"] = trainer_output["run_dir"]
        _write_json(status_path, status)
        print(json.dumps({"run_id": record["run_id"], "run_dir": status["run_dir"]}))
    except BaseException:
        if status.get("status") == "running":
            status["status"] = "failed"
            status["finished_at"] = _utc_now()
            _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
