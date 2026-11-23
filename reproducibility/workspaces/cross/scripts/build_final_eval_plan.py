#!/usr/bin/env python3
"""Resolve and verify selected checkpoints into a frozen final-evaluation plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess

try:
    from scripts.summarize_ablation import summarize_run
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
    from summarize_ablation import summarize_run  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint(
    entry: dict,
    roots: dict[str, Path],
    baseline_records_dir: Path,
    ablation_records_dir: Path,
    repo_root: Path,
) -> dict:
    source = entry["source"]
    if source == "explicit":
        path = roots[entry["root"]] / entry["path"]
        expected_sha = entry["sha256"]
        selected_at_episode = int(entry["selected_at_episode"])
        provenance = {"source": source, "root": entry["root"], "relative_path": entry["path"]}
    elif source == "run_record":
        record_kind = entry.get("record_kind", "baseline")
        default_records_dir = (
            baseline_records_dir if record_kind == "baseline" else ablation_records_dir
        )
        configured_records_dir = Path(entry.get("records_dir", default_records_dir))
        records_dir = (
            configured_records_dir
            if configured_records_dir.is_absolute()
            else repo_root / configured_records_dir
        )
        status_path = records_dir / f"{entry['run_id']}.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise ValueError(f"Training run is not complete: {status_path}")
        if int(status.get("train_seed", -1)) != int(entry["seed"]):
            raise ValueError(f"Training-seed mismatch: {status_path}")
        if record_kind == "baseline":
            if status.get("algo") != entry["method"]:
                raise ValueError(f"Baseline method mismatch: {status_path}")
            path = Path(status["checkpoint_path"])
            expected_sha = status["checkpoint_sha256"]
            workspace = Path(status["cwd"])
            eval_dirs = list(
                (workspace / "runs" / entry["method"]).glob(f"{entry['run_id']}_*/eval")
            )
            if len(eval_dirs) != 1:
                raise ValueError(
                    f"Expected one evaluation directory for {entry['run_id']}, found {len(eval_dirs)}"
                )
            candidates = []
            for eval_path in eval_dirs[0].glob("eval_ep*.json"):
                episode = int(eval_path.stem.split("ep")[-1])
                summary = json.loads(eval_path.read_text(encoding="utf-8"))["summary"]
                candidates.append(
                    (
                        float(summary["mean_completion_ratio"]),
                        float(summary["success_rate"]),
                        -episode,
                    )
                )
            if not candidates:
                raise ValueError(f"No checkpoint-selection evaluations found for {entry['run_id']}")
            selected_at_episode = -max(candidates)[2]
        elif record_kind == "ablation":
            expected_variant = str(
                entry.get(
                    "variant",
                    "full" if str(entry["method"]).lower() == "sacc" else entry["method"],
                )
            )
            if status.get("variant") != expected_variant:
                raise ValueError(
                    f"Ablation variant mismatch for {status_path}: "
                    f"found {status.get('variant')!r}, expected {expected_variant!r}"
                )
            status_for_summary = {**status, "status_path": str(status_path)}
            selected = summarize_run(status, status_for_summary, Path(status["run_dir"]))
            path = Path(selected["selected_checkpoint"])
            expected_sha = _sha256(path)
            selected_at_episode = int(selected["selected_checkpoint_episode"])
        else:
            raise ValueError(f"Unsupported run-record kind: {record_kind}")
        provenance = {
            "source": source,
            "record_kind": record_kind,
            "run_id": entry["run_id"],
            "variant": status.get("variant"),
            "run_record_path": str(status_path),
            "run_record_sha256": _sha256(status_path),
        }
    else:
        raise ValueError(f"Unsupported checkpoint source: {source}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"Checkpoint digest mismatch: {path}")
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": path.stat().st_size,
        "selected_at_episode": selected_at_episode,
        "provenance": provenance,
    }


def build_plan(
    repo_root: Path,
    spec_path: Path,
    *,
    robust_root: Path,
    historical_root: Path,
    baseline_records_dir: Path,
    ablation_records_dir: Path,
) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", -1)) != 1:
        raise ValueError("Only final-evaluation specification schema version 1 is supported")
    roots = {"robust": robust_root.resolve(), "historical": historical_root.resolve()}
    protocols: dict[str, dict] = {}
    for protocol in spec["protocols"]:
        path = repo_root / protocol["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _sha256(path) != protocol["file_sha256"]:
            raise ValueError(f"Protocol file digest mismatch: {path}")
        if payload["start_set_id"] != protocol["start_set_id"]:
            raise ValueError(f"Protocol payload digest mismatch: {path}")
        if int(payload["count"]) != int(protocol["count"]):
            raise ValueError(f"Protocol episode count mismatch: {path}")
        protocols[protocol["name"]] = {**protocol, "absolute_path": str(path.resolve())}

    checkpoints: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for entry in spec["checkpoints"]:
        key = (str(entry["method"]).lower(), int(entry["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate method/seed checkpoint: {key}")
        seen.add(key)
        checkpoints.append(
            {
                **entry,
                **_resolve_checkpoint(
                    entry,
                    roots,
                    baseline_records_dir,
                    ablation_records_dir,
                    repo_root,
                ),
            }
        )
    expected_seeds = {
        int(seed)
        for seed in spec.get("expected_seeds", (42, 123, 456, 789, 2024))
    }
    expected_methods = tuple(
        str(method).lower()
        for method in spec.get("expected_methods", ("sacc", "happo", "ippo"))
    )
    if not expected_methods or len(set(expected_methods)) != len(expected_methods):
        raise ValueError("expected_methods must contain unique method labels")
    reference_method = str(spec.get("reference_method", expected_methods[0])).lower()
    if reference_method not in expected_methods:
        raise ValueError("reference_method must be listed in expected_methods")
    comparison_methods = tuple(
        str(method).lower()
        for method in spec.get(
            "comparison_methods",
            [method for method in expected_methods if method != reference_method],
        )
    )
    if any(method not in expected_methods or method == reference_method for method in comparison_methods):
        raise ValueError("comparison_methods must be non-reference entries from expected_methods")
    execution_modes = tuple(
        str(mode).lower()
        for mode in spec.get("policy_modes", (spec.get("policy_mode", "greedy"),))
    )
    if not execution_modes or len(set(execution_modes)) != len(execution_modes):
        raise ValueError("policy_modes must contain unique execution modes")
    unsupported_modes = set(execution_modes) - {
        "greedy",
        "tempered",
        "cycle_aware_greedy",
    }
    if unsupported_modes:
        raise ValueError(f"Unsupported policy modes: {sorted(unsupported_modes)}")
    tempered_temperature = None
    if "tempered" in execution_modes:
        if "tempered_temperature" not in spec:
            raise ValueError("Tempered evaluation requires an explicit temperature")
        tempered_temperature = float(spec["tempered_temperature"])
        if not isfinite(tempered_temperature) or tempered_temperature <= 0:
            raise ValueError("Tempered evaluation temperature must be positive and finite")
    for method in expected_methods:
        method_seeds = {seed for candidate, seed in seen if candidate == method}
        if method_seeds != expected_seeds:
            raise ValueError(f"{method} seeds are {sorted(method_seeds)}, expected {sorted(expected_seeds)}")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    source_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:src/sacc-drama"], cwd=repo_root, text=True
    ).strip()
    records: list[dict] = []
    training_episodes_by_method = {
        "sacc": 1000,
        "happo": 3000,
        "ippo": 3000,
        **{
            str(method).lower(): int(episodes)
            for method, episodes in spec.get("training_episodes_by_method", {}).items()
        },
    }
    for checkpoint in checkpoints:
        training_episodes = checkpoint.get(
            "training_episodes",
            training_episodes_by_method.get(str(checkpoint["method"]).lower()),
        )
        if training_episodes is None:
            raise ValueError(
                f"No training episode count configured for method {checkpoint['method']!r}"
            )
        for protocol_name in protocols:
            protocol = protocols[protocol_name]
            for execution_mode in execution_modes:
                mode_suffix = f"__{execution_mode}" if len(execution_modes) > 1 else ""
                run_id = f"{spec['name']}__{checkpoint['method']}__seed{checkpoint['seed']}__{protocol_name}{mode_suffix}"
                records.append(
                    {
                        "schema_version": 1,
                        "index": len(records),
                        "run_id": run_id,
                        "status": "planned",
                        "reviewer_items": spec["reviewer_items"],
                        "experiment_name": spec["name"],
                        "method": checkpoint["method"],
                        "train_seed": checkpoint["seed"],
                        "training_episodes": int(training_episodes),
                        "difficulty": spec["difficulty"],
                        "evaluation_seed": spec["evaluation_seed"],
                        "max_steps": spec["max_steps"],
                        "policy_mode": execution_mode,
                        "execution_mode": execution_mode,
                        **({"evaluation_temperature": tempered_temperature}
                           if execution_mode == "tempered" else {}),
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                        "selected_at_episode": checkpoint["selected_at_episode"],
                        "checkpoint_provenance": checkpoint["provenance"],
                        "protocol_name": protocol_name,
                        "start_set_path": protocol["absolute_path"],
                        "start_set_file_sha256": protocol["file_sha256"],
                        "start_set_id": protocol["start_set_id"],
                        "episodes": protocol["count"],
                        "git_commit": commit,
                        "source_tree": source_tree,
                    }
                )
    return {
        "schema_version": 1,
        "experiment_name": spec["name"],
        "git_commit": commit,
        "source_tree": source_tree,
        "spec_path": str(spec_path.relative_to(repo_root)),
        "spec_sha256": _sha256(spec_path),
        "run_count": len(records),
        "checkpoint_count": len(checkpoints),
        "expected_methods": list(expected_methods),
        "reference_method": reference_method,
        "execution_modes": list(execution_modes),
        "comparison_methods": list(comparison_methods),
        "protocols": protocols,
        "checkpoints": checkpoints,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--robust-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-records-dir",
        type=Path,
        default=Path("experiments/baseline_run_records"),
    )
    parser.add_argument(
        "--ablation-records-dir",
        type=Path,
        default=Path("experiments/run_records"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else repo_root / args.spec
    records_dir = (
        args.baseline_records_dir
        if args.baseline_records_dir.is_absolute()
        else repo_root / args.baseline_records_dir
    )
    ablation_records_dir = (
        args.ablation_records_dir
        if args.ablation_records_dir.is_absolute()
        else repo_root / args.ablation_records_dir
    )
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    plan = build_plan(
        repo_root,
        spec_path.resolve(),
        robust_root=args.robust_root,
        historical_root=args.historical_root,
        baseline_records_dir=records_dir,
        ablation_records_dir=ablation_records_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "run_count": plan["run_count"]}))


if __name__ == "__main__":
    main()
