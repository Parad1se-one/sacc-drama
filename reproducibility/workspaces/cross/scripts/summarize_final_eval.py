#!/usr/bin/env python3
"""Strictly validate and aggregate the matched final-evaluation results."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isclose, isfinite
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
BASE_DIAGNOSTIC_METRICS = (
    "mean_stay_fraction",
    "mean_blocked_fraction",
    "mean_missed_perform_rate",
    "mean_max_no_progress_streak",
    "mean_progress_steps",
    "mean_cycle_breaks",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_close(actual: object, expected: float, field: str, path: Path) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {path}") from exc
    if not isfinite(value) or not isclose(value, expected, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"{field} disagrees with raw episodes: {path}")


def _validate_episode_payload(payload: dict, record: dict, start_set: dict, path: Path) -> None:
    """Recheck the frozen episode starts and recompute seed-level metrics."""
    starts = start_set["starts"]
    count = int(record["episodes"])
    if (start_set["start_set_id"] != record["start_set_id"]
            or int(start_set["count"]) != count or len(starts) != count):
        raise ValueError(f"Frozen start-set metadata mismatch: {path}")
    if len({tuple(tuple(pos) for pos in item) for item in starts}) != count:
        raise ValueError(f"Frozen test starts are not unique: {path}")
    for field, expected in (
        ("difficulty", record["difficulty"]),
        ("max_steps", record["max_steps"]),
        ("seed", record["evaluation_seed"]),
        ("start_split", start_set["split"]),
        ("unique_start_count", count),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"Evaluation {field} mismatch: {path}")
    if Path(payload["checkpoint_path"]).resolve() != Path(record["checkpoint_path"]).resolve():
        raise ValueError(f"Evaluated checkpoint path mismatch: {path}")

    # Only a successful task-node transition gives positive raw reward (+1).
    reward_caps = {
        "easy": 8, "hard": 40, "large": 96,
        "linear6": 72, "cross5": 84,
    }
    difficulty = str(record["difficulty"])
    if difficulty not in reward_caps:
        raise ValueError(f"No audited raw-reward cap for {difficulty}: {path}")
    episodes = payload["episodes"]
    if len(episodes) != count:
        raise ValueError(f"Episode count mismatch: {path}")
    core = {
        "mean_completion_ratio": [], "success_rate": [], "mean_steps": [],
        "mean_environment_return": [], "mean_objective_return": [],
    }
    diagnostics = {
        "mean_stay_fraction": [], "mean_blocked_fraction": [],
        "mean_missed_perform_rate": [], "mean_max_no_progress_streak": [],
        "mean_progress_steps": [], "mean_cycle_breaks": [],
    }
    terminal_keys = {
        key for episode in episodes
        for key in episode["diagnostics"]["terminal_stage_counts"]
    }
    terminal = {key: [] for key in terminal_keys}
    for index, (episode, expected_start) in enumerate(zip(episodes, starts), 1):
        if episode.get("episode") != index:
            raise ValueError(f"Evaluation episode order mismatch: {path}")
        if episode.get("start_positions") != expected_start:
            raise ValueError(f"Actual test start differs from frozen manifest: {path}")
        completion = float(episode["completion_ratio"])
        success = float(episode["success"])
        steps = float(episode["steps"])
        environment_return = float(episode["environment_return"])
        objective_return = float(episode["objective_return"])
        if not all(isfinite(value) for value in (
                completion, success, steps, environment_return, objective_return)):
            raise ValueError(f"Nonfinite raw evaluation metric: {path}")
        if not 0 <= completion <= 1 or success not in (0.0, 1.0):
            raise ValueError(f"Invalid completion or success: {path}")
        if success != float(completion >= 1.0 - 1e-6):
            raise ValueError(f"Success disagrees with completion: {path}")
        if not steps.is_integer() or not 1 <= steps <= int(record["max_steps"]):
            raise ValueError(f"Invalid episode step count: {path}")
        if environment_return > reward_caps[difficulty] * completion + 1e-6:
            raise ValueError(f"Raw environment return exceeds progress bound: {path}")
        for field, value in (
            ("mean_completion_ratio", completion), ("success_rate", success),
            ("mean_steps", steps), ("mean_environment_return", environment_return),
            ("mean_objective_return", objective_return),
        ):
            core[field].append(value)
        diag = episode["diagnostics"]
        denominator = max(int(steps) * len(expected_start), 1)
        opportunities = int(diag["perform_opportunities"])
        diagnostics["mean_stay_fraction"].append(
            float(diag["status_counts"].get("stayed", 0)) / denominator)
        diagnostics["mean_blocked_fraction"].append(
            float(diag["status_counts"].get("blocked", 0)) / denominator)
        diagnostics["mean_missed_perform_rate"].append(
            float(diag["missed_perform_opportunities"]) / opportunities
            if opportunities else 0.0)
        for field, source in (
            ("mean_max_no_progress_streak", "max_no_progress_streak"),
            ("mean_progress_steps", "progress_steps"),
            ("mean_cycle_breaks", "cycle_breaks"),
        ):
            diagnostics[field].append(float(diag.get(source, 0)))
        for key in terminal_keys:
            terminal[key].append(float(diag["terminal_stage_counts"].get(key, 0)))
    for field, values in core.items():
        _require_close(payload["summary"][field], mean(values), field, path)
    for field, values in diagnostics.items():
        _require_close(payload["diagnostic_summary"][field], mean(values), field, path)
    for key, values in terminal.items():
        _require_close(
            payload["diagnostic_summary"]["mean_terminal_stage_counts"][key],
            mean(values), f"terminal_{key}", path,
        )

def collect_rows(plan: dict, records_dir: Path) -> list[dict]:
    rows: list[dict] = []
    seen_keys: set[tuple[str, str, str, int]] = set()
    for record in plan["records"]:
        status_path = records_dir / f"{record['run_id']}.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("run_id") != record["run_id"]:
            raise ValueError(f"Final-evaluation run ID mismatch: {status_path}")
        if status.get("status") != "completed":
            raise ValueError(f"Final-evaluation record is not complete: {status_path}")
        if status.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint provenance mismatch: {status_path}")
        if _sha256(Path(record["checkpoint_path"])) != record["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed: {status_path}")
        start_set_path = Path(record["start_set_path"])
        if _sha256(start_set_path) != record["start_set_file_sha256"]:
            raise ValueError(f"Frozen start manifest changed: {start_set_path}")
        start_set = json.loads(start_set_path.read_text(encoding="utf-8"))
        result_path = Path(status["result_path"])
        if _sha256(result_path) != status["result_sha256"]:
            raise ValueError(f"Evaluation result digest mismatch: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload["start_set_id"] != record["start_set_id"]:
            raise ValueError(f"Start-set identifier mismatch: {result_path}")
        if payload["algo"] != record["method"]:
            raise ValueError(f"Method label mismatch: {result_path}")
        execution_mode = str(
            record.get("execution_mode", record.get("policy_mode", "greedy"))
        )
        if str(payload.get("policy_mode", execution_mode)) != execution_mode:
            raise ValueError(f"Execution-mode mismatch: {result_path}")
        if "evaluation_temperature" in record and not isclose(
            float(payload.get("temperature", "nan")), float(record["evaluation_temperature"])
        ):
            raise ValueError(f"Execution-temperature mismatch: {result_path}")
        _validate_episode_payload(payload, record, start_set, result_path)
        key = (
            str(record["method"]), execution_mode,
            str(record["protocol_name"]), int(record["train_seed"]),
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate method/mode/protocol/seed: {key}")
        seen_keys.add(key)
        summary = payload["summary"]
        diagnostic = payload["diagnostic_summary"]
        terminal = diagnostic["mean_terminal_stage_counts"]
        rows.append(
            {
                "run_id": record["run_id"],
                "method": record["method"],
                "execution_mode": execution_mode,
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
                "mean_cycle_breaks": float(diagnostic.get("mean_cycle_breaks", 0.0)),
                **{f"terminal_{key}": float(value) for key, value in terminal.items()},
            }
        )
    if len(rows) != int(plan["run_count"]):
        raise ValueError("Collected row count does not match the plan")
    return rows


def _metric_names(rows: list[dict]) -> tuple[str, ...]:
    terminal_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("terminal_")
        }
    )
    return METRICS + BASE_DIAGNOSTIC_METRICS + tuple(terminal_metrics)


def aggregate(
    rows: list[dict],
    *,
    methods: list[str] | tuple[str, ...] | None = None,
    reference_method: str = "sacc",
) -> list[dict]:
    observed_methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    if methods is None:
        preferred = [method for method in ("sacc", "happo", "ippo") if method in observed_methods]
        methods = [*preferred, *[method for method in observed_methods if method not in preferred]]
    protocols = list(dict.fromkeys(row["protocol"] for row in rows))
    execution_modes = list(
        dict.fromkeys(str(row.get("execution_mode", "greedy")) for row in rows)
    )
    expected_seeds = {
        int(row["train_seed"]) for row in rows if row["method"] == reference_method
    }
    if not expected_seeds:
        raise ValueError(f"Final evaluation contains no {reference_method} rows")
    metric_names = _metric_names(rows)
    output: list[dict] = []
    for method in methods:
        for execution_mode in execution_modes:
            for protocol in protocols:
                group = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["protocol"] == protocol
                    and str(row.get("execution_mode", "greedy")) == execution_mode
                ]
                group_seeds = {int(row["train_seed"]) for row in group}
                if group_seeds != expected_seeds:
                    raise ValueError(
                        f"Seed mismatch for {method}/{execution_mode}/{protocol}: "
                        f"found {sorted(group_seeds)}, expected {sorted(expected_seeds)}"
                    )
                item = {
                    "method": method,
                    "execution_mode": execution_mode,
                    "protocol": protocol,
                    "n_seeds": len(group),
                    "metrics": {},
                }
                for metric in metric_names:
                    seed_values = [
                        {"seed": int(row["train_seed"]), "value": float(row.get(metric, 0.0))}
                        for row in sorted(group, key=lambda row: int(row["train_seed"]))
                    ]
                    values = [item["value"] for item in seed_values]
                    item["metrics"][metric] = {
                        "mean": float(mean(values)),
                        "sample_std": float(stdev(values)) if len(values) > 1 else None,
                        "iqm": interquartile_mean(values),
                        "seed_values": seed_values,
                    }
                output.append(item)
    return output


def _paired_effects_single_mode(
    rows: list[dict],
    *,
    execution_mode: str,
    reference_method: str = "sacc",
    comparison_methods: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    by_key = {(row["method"], row["protocol"], row["train_seed"]): row for row in rows}
    protocols = list(dict.fromkeys(row["protocol"] for row in rows))
    observed_methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    if comparison_methods is None:
        preferred = [method for method in ("happo", "ippo") if method in observed_methods]
        comparison_methods = [
            *preferred,
            *[
                method
                for method in observed_methods
                if method != reference_method and method not in preferred
            ],
        ]
    effects: list[dict] = []
    for protocol in protocols:
        for baseline in comparison_methods:
            reference_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == reference_method and row["protocol"] == protocol
            }
            baseline_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == baseline and row["protocol"] == protocol
            }
            if baseline_seeds != reference_seeds:
                raise ValueError(
                    f"Paired seed mismatch for {baseline}/{protocol}: "
                    f"found {sorted(baseline_seeds)}, expected {sorted(reference_seeds)}"
                )
            seeds = sorted(reference_seeds)
            metrics: dict[str, dict] = {}
            for metric in (
                "mean_completion_ratio",
                "success_rate",
                "mean_steps",
                "mean_environment_return",
            ):
                differences = []
                for seed in seeds:
                    reference = by_key[(reference_method, protocol, seed)][metric]
                    other = by_key[(baseline, protocol, seed)][metric]
                    difference = (
                        other - reference if metric == "mean_steps" else reference - other
                    )
                    differences.append(float(difference))
                metrics[metric] = {
                    "orientation": f"positive_favors_{reference_method}",
                    "mean_difference": float(mean(differences)),
                    "sample_std": float(stdev(differences)) if len(differences) > 1 else None,
                    "seeds": seeds,
                    "differences": differences,
                    "positive_pairs": sum(value > 0 for value in differences),
                    "zero_pairs": sum(value == 0 for value in differences),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                    "paired_seed_bootstrap_mean_difference_95ci": bootstrap_mean_ci(differences),
                }
            effects.append(
                {
                    "execution_mode": execution_mode,
                    "protocol": protocol,
                    "reference_method": reference_method,
                    "baseline": baseline,
                    "n_pairs": len(seeds),
                    "metrics": metrics,
                }
            )
    return effects


def paired_effects(
    rows: list[dict],
    *,
    reference_method: str = "sacc",
    comparison_methods: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    execution_modes = list(
        dict.fromkeys(str(row.get("execution_mode", "greedy")) for row in rows)
    )
    effects: list[dict] = []
    for execution_mode in execution_modes:
        mode_rows = [
            row
            for row in rows
            if str(row.get("execution_mode", "greedy")) == execution_mode
        ]
        effects.extend(
            _paired_effects_single_mode(
                mode_rows,
                execution_mode=execution_mode,
                reference_method=reference_method,
                comparison_methods=comparison_methods,
            )
        )
    return effects


def paired_execution_effects(
    rows: list[dict],
    *,
    reference_mode: str = "greedy",
    comparison_mode: str = "cycle_aware_greedy",
) -> list[dict]:
    modes = {str(row.get("execution_mode", "greedy")) for row in rows}
    if reference_mode not in modes or comparison_mode not in modes:
        return []
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    protocols = list(dict.fromkeys(str(row["protocol"]) for row in rows))
    by_key = {
        (
            str(row["method"]),
            str(row["protocol"]),
            int(row["train_seed"]),
            str(row.get("execution_mode", "greedy")),
        ): row
        for row in rows
    }
    effects: list[dict] = []
    for method in methods:
        for protocol in protocols:
            reference_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == method
                and row["protocol"] == protocol
                and str(row.get("execution_mode", "greedy")) == reference_mode
            }
            comparison_seeds = {
                int(row["train_seed"])
                for row in rows
                if row["method"] == method
                and row["protocol"] == protocol
                and str(row.get("execution_mode", "greedy")) == comparison_mode
            }
            if comparison_seeds != reference_seeds:
                raise ValueError(
                    f"Paired execution seed mismatch for {method}/{protocol}: "
                    f"found {sorted(comparison_seeds)}, expected {sorted(reference_seeds)}"
                )
            seeds = sorted(reference_seeds)
            metrics: dict[str, dict] = {}
            for metric in (
                "mean_completion_ratio",
                "success_rate",
                "mean_steps",
                "mean_environment_return",
            ):
                differences = []
                for seed in seeds:
                    reference = by_key[(method, protocol, seed, reference_mode)][metric]
                    comparison = by_key[(method, protocol, seed, comparison_mode)][metric]
                    difference = (
                        reference - comparison
                        if metric == "mean_steps"
                        else comparison - reference
                    )
                    differences.append(float(difference))
                metrics[metric] = {
                    "orientation": f"positive_favors_{comparison_mode}",
                    "mean_difference": float(mean(differences)),
                    "sample_std": float(stdev(differences)) if len(differences) > 1 else None,
                    "seeds": seeds,
                    "differences": differences,
                    "positive_pairs": sum(value > 0 for value in differences),
                    "zero_pairs": sum(value == 0 for value in differences),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                    "paired_seed_bootstrap_mean_difference_95ci": bootstrap_mean_ci(differences),
                }
            effects.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "reference_execution_mode": reference_mode,
                    "comparison_execution_mode": comparison_mode,
                    "n_pairs": len(seeds),
                    "metrics": metrics,
                }
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
        "reference_method": plan.get("reference_method", "sacc"),
        "comparison_methods": plan.get("comparison_methods", ["happo", "ippo"]),
        "aggregates": aggregate(
            rows,
            methods=plan.get("expected_methods"),
            reference_method=plan.get("reference_method", "sacc"),
        ),
        "paired_effects": paired_effects(
            rows,
            reference_method=plan.get("reference_method", "sacc"),
            comparison_methods=plan.get("comparison_methods", ["happo", "ippo"]),
        ),
        "paired_execution_effects": paired_execution_effects(rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.csv_out, rows)
    print(json.dumps({"rows": len(rows), "json_out": str(args.json_out), "csv_out": str(args.csv_out)}))


if __name__ == "__main__":
    main()
