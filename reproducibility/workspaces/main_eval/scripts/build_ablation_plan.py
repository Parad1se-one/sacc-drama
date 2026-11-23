#!/usr/bin/env python3
"""Compile a declarative ablation specification into traceable run records."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


REPO_ROOT_TOKEN = "{repo_root}"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _git_tree(repo_root: Path, relative_path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{relative_path}"],
        cwd=repo_root,
        text=True,
    ).strip()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_argument(argv: list[str], name: str, value) -> None:
    if isinstance(value, bool):
        if value:
            argv.append(_flag(name))
        return
    if value is None:
        return
    argv.extend([_flag(name), str(value)])


def _load_start_set(repo_root: Path, relative_path: str) -> dict:
    path = (repo_root / relative_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("start_set_id") is None:
        raise ValueError(f"Start set has no start_set_id: {path}")
    if int(payload.get("count", -1)) != len(payload.get("starts", [])):
        raise ValueError(f"Start-set count mismatch: {path}")
    return {
        "path": relative_path,
        "file_sha256": _sha256(path),
        "start_set_id": payload["start_set_id"],
        "split": payload["split"],
        "count": payload["count"],
    }


def build_plan(repo_root: Path, spec_path: Path) -> dict:
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", -1)) != 1:
        raise ValueError("Only ablation specification schema version 1 is supported")

    seeds = [int(seed) for seed in spec["seeds"]]
    variants = list(spec["variants"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be unique")
    variant_names = [variant["name"] for variant in variants]
    if len(variant_names) != len(set(variant_names)):
        raise ValueError("Variant names must be unique")

    selection = _load_start_set(repo_root, spec["selection_start_set"])
    final_random = _load_start_set(repo_root, spec["final_random_start_set"])
    if selection["start_set_id"] == final_random["start_set_id"]:
        raise ValueError("Selection and final-test start sets must be different")
    if int(spec["common_args"]["eval_episodes"]) != int(selection["count"]):
        raise ValueError("eval_episodes must equal the selection start-set count")

    commit = _git_commit(repo_root)
    source_tree = _git_tree(repo_root, "src/sacc-drama")
    spec_sha = _sha256(spec_path)
    records: list[dict] = []
    index = 0
    for variant in variants:
        for seed in seeds:
            run_id = f"{spec['name']}__{variant['name']}__seed{seed}"
            args = dict(spec["common_args"])
            args.update(variant.get("overrides", {}))
            argv = ["-m", "sacc.experiments.train_drama"]
            _append_argument(argv, "difficulty", spec["difficulty"])
            _append_argument(argv, "seed", seed)
            _append_argument(argv, "train_start_protocol", spec["train_start_protocol"])
            _append_argument(
                argv,
                "validation_start_set",
                f"{REPO_ROOT_TOKEN}/{selection['path']}",
            )
            _append_argument(argv, "exp_name", run_id)
            for name, value in args.items():
                _append_argument(argv, name, value)

            records.append(
                {
                    "schema_version": 1,
                    "index": index,
                    "run_id": run_id,
                    "status": "planned",
                    "reviewer_item": spec["reviewer_item"],
                    "experiment_name": spec["name"],
                    "variant": variant["name"],
                    "variant_description": variant["description"],
                    "difficulty": spec["difficulty"],
                    "train_seed": seed,
                    "git_commit": commit,
                    "source_tree": source_tree,
                    "spec_path": str(spec_path.relative_to(repo_root)),
                    "spec_sha256": spec_sha,
                    "selection_start_set": selection,
                    "final_random_start_set": final_random,
                    "argv": argv,
                    "resolved_args": args,
                }
            )
            index += 1

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
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else repo_root / args.spec
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    plan = build_plan(repo_root, spec_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "run_count": plan["run_count"],
                "git_commit": plan["git_commit"],
                "source_tree": plan["source_tree"],
                "spec_sha256": plan["spec_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
