#!/usr/bin/env python3
"""Strictly validate and aggregate the matched final-evaluation results."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev

try:
    from scripts.summarize_ablation import _exact_sign_flip_p
    from scripts.seed_statistics import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        bootstrap_mean_ci,
        interquartile_mean,
    )
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
    from summarize_ablation import _exact_sign_flip_p  # type: ignore[no-redef]
    from seed_statistics import (  # type: ignore[no-redef]
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        bootstrap_mean_ci,
        interquartile_mean,
    )


METRICS = (
    "mean_completion_ratio",
    "success_rate",
    "mean_steps",
    "mean_environment_return",
    "mean_objective_return",
)
DIAGNOSTIC_METRICS = (
    "mean_stay_fraction",
    "mean_blocked_fraction",
    "mean_missed_perform_rate",
    "mean_max_no_progress_streak",
    "mean_progress_steps",
    "terminal_to_distribute",
    "terminal_to_vibrate",
    "terminal_to_level",
    "terminal_to_cover",
    "terminal_completed",
)
ALL_METRICS = METRICS + DIAGNOSTIC_METRICS


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rows(plan: dict, records_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for record in plan["records"]:
        status_path = records_dir / f"{record['run_id']}.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise ValueError(f"Final-evaluation record is not complete: {status_path}")
        if status.get("run_id") != record["run_id"]:
            raise ValueError(f"Final-evaluation run ID mismatch: {status_path}")
        if status.get("source_tree") != record["source_tree"]:
            raise ValueError(f"Final-evaluation source-tree mismatch: {status_path}")
        if status.get("start_set_file_sha256") != record["start_set_file_sha256"]:
            raise ValueError(f"Final-evaluation start-set digest mismatch: {status_path}")
        if status.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint provenance mismatch: {status_path}")
        result_path = Path(status["result_path"])
        if _sha256(result_path) != status["result_sha256"]:
            raise ValueError(f"Evaluation result digest mismatch: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload["start_set_id"] != record["start_set_id"]:
            raise ValueError(f"Start-set identifier mismatch: {result_path}")
        if payload["algo"] != record["method"]:
            raise ValueError(f"Method label mismatch: {result_path}")
        if len(payload["episodes"]) != int(record["episodes"]):
            raise ValueError(f"Episode count mismatch: {result_path}")
        if (
            payload.get("difficulty") != record["difficulty"]
            or payload.get("seed") != record["evaluation_seed"]
            or payload.get("max_steps") != record["max_steps"]
            or payload.get("policy_mode") != record["policy_mode"]
            or float(payload.get("temperature", -1)) != float(record["temperature"])
            or Path(payload.get("checkpoint_path", "")).resolve()
            != Path(record["checkpoint_path"]).resolve()
        ):
            raise ValueError(f"Evaluator protocol mismatch: {result_path}")
        start_path = Path(record["start_set_path"])
        if _sha256(start_path) != record["start_set_file_sha256"]:
            raise ValueError(f"Frozen start set changed: {start_path}")
        start_set = json.loads(start_path.read_text(encoding="utf-8"))
        if (
            start_set.get("start_set_id") != record["start_set_id"]
            or start_set.get("count") != record["episodes"]
            or start_set.get("difficulty") != record["difficulty"]
            or payload.get("start_split") != start_set.get("split")
        ):
            raise ValueError(f"Frozen start-set metadata mismatch: {result_path}")
        if record["protocol_name"] == "random" and payload.get("start_protocol") != "versioned_random":
            raise ValueError(f"Expected random-start evaluation: {result_path}")
        for index, (episode, starts) in enumerate(
            zip(payload["episodes"], start_set["starts"], strict=True), start=1
        ):
            if episode.get("episode") != index or episode.get("start_positions") != starts:
                raise ValueError(f"Episode {index} did not use the frozen starts: {result_path}")
        summary = payload["summary"]
        diagnostic = payload["diagnostic_summary"]
        terminal = diagnostic["mean_terminal_stage_counts"]
        rows.append(
            {
                "run_id": record["run_id"],
                "method": record["method"],
                "train_seed": int(record["train_seed"]),
                "protocol": record["protocol_name"],
                "start_set_id": record["start_set_id"],
                "episodes": int(record["episodes"]),
                "actor_observation_mode": payload["actor_observation_mode"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                **{metric: float(summary[metric]) for metric in METRICS},
                "mean_stay_fraction": float(diagnostic["mean_stay_fraction"]),
                "mean_blocked_fraction": float(diagnostic["mean_blocked_fraction"]),
                "mean_missed_perform_rate": float(diagnostic["mean_missed_perform_rate"]),
                "mean_max_no_progress_streak": float(diagnostic["mean_max_no_progress_streak"]),
                "mean_progress_steps": float(diagnostic["mean_progress_steps"]),
                **{f"terminal_{key}": float(value) for key, value in terminal.items()},
            }
        )
    if len(rows) != int(plan["run_count"]):
        raise ValueError("Collected row count does not match the plan")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    methods = list(dict.fromkeys(str(row["method"]).lower() for row in rows))
    if not methods:
        raise ValueError("Final evaluation contains no method rows")
    protocols = list(dict.fromkeys(row["protocol"] for row in rows))
    reference_method = methods[0]
    expected_seeds = {
        int(row["train_seed"]) for row in rows if row["method"] == reference_method
    }
    if not expected_seeds:
        raise ValueError(f"Final evaluation contains no {reference_method} seed rows")
    output: list[dict] = []
    for method in methods:
        for protocol in protocols:
            group = [row for row in rows if row["method"] == method and row["protocol"] == protocol]
            group_seeds = {int(row["train_seed"]) for row in group}
            if group_seeds != expected_seeds:
                raise ValueError(
                    f"Seed mismatch for {method}/{protocol}: "
                    f"found {sorted(group_seeds)}, expected {sorted(expected_seeds)}"
                )
            item = {"method": method, "protocol": protocol, "n_seeds": len(group), "metrics": {}}
            for metric in ALL_METRICS:
                seed_values = [
                    {"seed": int(row["train_seed"]), "value": float(row[metric])}
                    for row in sorted(group, key=lambda row: int(row["train_seed"]))
                ]
                values = [item["value"] for item in seed_values]
                item["metrics"][metric] = {
                    "mean": float(mean(values)),
                    "sample_std": float(stdev(values)),
                    "seed_values": seed_values,
                }
                if metric in METRICS:
                    item["metrics"][metric]["iqm"] = interquartile_mean(values)
                    item["metrics"][metric]["seed_bootstrap_mean_95ci"] = bootstrap_mean_ci(values)
            output.append(item)
    return output


def paired_effects(rows: list[dict]) -> list[dict]:
    if not any(row["method"] == "sacc" for row in rows):
        return []
    by_key = {(row["method"], row["protocol"], row["train_seed"]): row for row in rows}
    protocols = list(dict.fromkeys(row["protocol"] for row in rows))
    effects: list[dict] = []
    for protocol in protocols:
        for baseline in ("happo", "ippo"):
            baseline_rows = [
                row
                for row in rows
                if row["method"] == baseline and row["protocol"] == protocol
            ]
            if not baseline_rows:
                continue
            sacc_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == "sacc" and row["protocol"] == protocol
            }
            baseline_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == baseline and row["protocol"] == protocol
            }
            if baseline_seeds != sacc_seeds:
                raise ValueError(
                    f"Paired seed mismatch for {baseline}/{protocol}: "
                    f"found {sorted(baseline_seeds)}, expected {sorted(sacc_seeds)}"
                )
            seeds = sorted(sacc_seeds)
            metrics: dict[str, dict] = {}
            for metric in (
                "mean_completion_ratio",
                "success_rate",
                "mean_steps",
                "mean_environment_return",
            ):
                differences = []
                for seed in seeds:
                    sacc = by_key[("sacc", protocol, seed)][metric]
                    other = by_key[(baseline, protocol, seed)][metric]
                    difference = other - sacc if metric == "mean_steps" else sacc - other
                    differences.append(float(difference))
                metrics[metric] = {
                    "orientation": "positive_favors_sacc",
                    "mean_difference": float(mean(differences)),
                    "iqm_paired_difference": interquartile_mean(differences),
                    "sample_std": float(stdev(differences)) if len(differences) > 1 else None,
                    "seeds": seeds,
                    "differences": differences,
                    "positive_pairs": sum(value > 0 for value in differences),
                    "zero_pairs": sum(value == 0 for value in differences),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                    "paired_seed_bootstrap_mean_difference_95ci": bootstrap_mean_ci(differences),
                }
            effects.append(
                {"protocol": protocol, "baseline": baseline, "n_pairs": len(seeds), "metrics": metrics}
            )
    return effects


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows = collect_rows(plan, args.records_dir)
    payload = {
        "schema_version": 1,
        "experiment_name": plan["experiment_name"],
        "plan_sha256": _sha256(args.plan),
        "source_tree": plan["source_tree"],
        "cross_method_reward_metric": "mean_environment_return",
        "statistical_unit": "independent_training_seed",
        "bootstrap": {
            "method": "paired training-seed percentile bootstrap of the mean difference",
            "confidence_level": 0.95,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        },
        "aggregates": aggregate(rows),
        "paired_effects": paired_effects(rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.csv_out, rows)
    print(json.dumps({"rows": len(rows), "json_out": str(args.json_out), "csv_out": str(args.csv_out)}))


if __name__ == "__main__":
    main()
