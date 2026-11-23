#!/usr/bin/env python3
"""Compile the minimal missing-baseline specification into a frozen plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_argument(argv: list[str], name: str, value) -> None:
    if value is not None:
        argv.extend(["--" + name.replace("_", "-"), str(value)])


def build_plan(repo_root: Path, spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", -1)) != 1:
        raise ValueError("Only baseline specification schema version 1 is supported")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src/sacc-drama"], cwd=repo_root, text=True
    ).strip()
    spec_sha = _sha256(spec_path)
    selection_start_set = None
    selection_value = spec["common_args"].get("validation_start_set")
    if selection_value:
        selection_path = Path(selection_value)
        if not selection_path.is_absolute():
            selection_path = repo_root / selection_path
        selection_path = selection_path.resolve()
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_payload.get("difficulty") != spec["difficulty"]:
            raise ValueError("Baseline selection set difficulty mismatch")
        if int(selection_payload.get("count", -1)) != int(spec["common_args"]["eval_episodes"]):
            raise ValueError("Baseline selection set count mismatch")
        selection_start_set = {
            "path": str(selection_path),
            "file_sha256": _sha256(selection_path),
            "start_set_id": selection_payload["start_set_id"],
            "count": selection_payload["count"],
        }
    seen: set[tuple[str, int]] = set()
    records: list[dict] = []
    for method in spec["methods"]:
        algo = str(method["algo"]).lower()
        for seed_value in method["seeds"]:
            seed = int(seed_value)
            if (algo, seed) in seen:
                raise ValueError(f"Duplicate baseline run: {(algo, seed)}")
            seen.add((algo, seed))
            resolved = dict(spec["common_args"])
            resolved.update(method.get("overrides", {}))
            run_id = f"{spec['name']}__{algo}__seed{seed}"
            argv = ["--algo", algo, "--difficulty", spec["difficulty"], "--seed", str(seed)]
            for name, value in resolved.items():
                _append_argument(argv, name, value)
            argv.extend(["--exp-name", run_id])
            records.append(
                {
                    "schema_version": 1,
                    "index": len(records),
                    "run_id": run_id,
                    "status": "planned",
                    "reviewer_items": spec["reviewer_items"],
                    "experiment_name": spec["name"],
                    "algo": algo,
                    "difficulty": spec["difficulty"],
                    "train_seed": seed,
                    "git_commit": commit,
                    "source_tree": source_tree,
                    "spec_path": str(spec_path.relative_to(repo_root)),
                    "spec_sha256": spec_sha,
                    "training_start_protocol": spec["training_start_protocol"],
                    "selection_start_protocol": spec["selection_start_protocol"],
                    "selection_rule": spec["selection_rule"],
                    "selection_start_set": selection_start_set,
                    "argv": argv,
                    "resolved_args": resolved,
                }
            )
    return {
        "schema_version": 1,
        "experiment_name": spec["name"],
        "git_commit": commit,
        "source_tree": source_tree,
        "spec_path": str(spec_path.relative_to(repo_root)),
        "spec_sha256": spec_sha,
        "run_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else repo_root / args.spec
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    plan = build_plan(repo_root, spec_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "run_count": plan["run_count"]}))


if __name__ == "__main__":
    main()
