#!/usr/bin/env python3
"""Validate and aggregate completed runs from one SACC ablation plan."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean, stdev

try:
    from scripts.seed_statistics import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        bootstrap_mean_ci,
        exact_two_sided_sign_flip_p,
        holm_adjust,
    )
except ModuleNotFoundError:  # Support direct ``python scripts/...`` execution.
    from seed_statistics import (  # type: ignore[no-redef]
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        bootstrap_mean_ci,
        exact_two_sided_sign_flip_p,
        holm_adjust,
    )


EVAL_NAME = re.compile(r"^eval_ep(?P<episode>[0-9]+)\.json$")
RUN_METRICS = (
    "final_completion",
    "final_success",
    "final_steps",
    "final_environment_return",
    "completion_auc",
    "first_eval_nonzero_success_episode",
    "first_eval_full_success_episode",
    "last_five_eval_success",
    "last_five_eval_completion",
    "selected_checkpoint_episode",
    "selected_completion",
    "selected_success",
    "selected_steps",
)
PAIRED_METRICS = {
    "completion_auc": "baseline_minus_variant",
    "last_five_eval_completion": "baseline_minus_variant",
    "last_five_eval_success": "baseline_minus_variant",
    "selected_completion": "baseline_minus_variant",
    "selected_success": "baseline_minus_variant",
    "selected_steps": "variant_minus_baseline",
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_episode(path: Path) -> int:
    match = EVAL_NAME.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected evaluation filename: {path}")
    return int(match.group("episode"))


def _load_evaluations(run_dir: Path, expected_start_set_id: str) -> list[tuple[int, dict]]:
    paths = [path for path in (run_dir / "eval").glob("eval_ep*.json") if EVAL_NAME.match(path.name)]
    evaluations = sorted(((_evaluation_episode(path), _read_json(path)) for path in paths), key=lambda item: item[0])
    if not evaluations:
        raise ValueError(f"No evaluation files found in {run_dir / 'eval'}")
    episodes = [episode for episode, _ in evaluations]
    if episodes != sorted(set(episodes)):
        raise ValueError(f"Duplicate or unordered evaluation episodes in {run_dir}")
    for episode, payload in evaluations:
        if payload.get("start_set_id") != expected_start_set_id:
            raise ValueError(
                f"Evaluation at episode {episode} used start set {payload.get('start_set_id')}, "
                f"expected {expected_start_set_id}"
            )
    return evaluations


def _normalized_auc(points: list[tuple[int, float]]) -> float:
    """Trapezoidal AUC from (episode=0, completion=0), divided by final episode."""
    if not points or points[-1][0] <= 0:
        raise ValueError("AUC requires at least one positive evaluation episode")
    area = 0.0
    previous_episode = 0
    previous_value = 0.0
    for episode, value in points:
        if episode <= previous_episode:
            raise ValueError("Evaluation episodes must be strictly increasing")
        area += (episode - previous_episode) * (previous_value + float(value)) / 2.0
        previous_episode = episode
        previous_value = float(value)
    return area / points[-1][0]


def _best_evaluation(evaluations: list[tuple[int, dict]]) -> tuple[int, dict]:
    """Mirror trainer selection: completion, success, then fewer steps; keep first tie."""
    best_episode, best_payload = evaluations[0]
    best_summary = best_payload["summary"]
    for episode, payload in evaluations[1:]:
        summary = payload["summary"]
        candidate = (
            float(summary["mean_completion_ratio"]),
            float(summary["success_rate"]),
            -float(summary["mean_steps"]),
        )
        incumbent = (
            float(best_summary["mean_completion_ratio"]),
            float(best_summary["success_rate"]),
            -float(best_summary["mean_steps"]),
        )
        if candidate > incumbent:
            best_episode, best_payload, best_summary = episode, payload, summary
    return best_episode, best_payload


def summarize_run(record: dict, status: dict, run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing trainer summary: {summary_path}")
    trainer_summary = _read_json(summary_path)
    evaluations = _load_evaluations(
        run_dir,
        expected_start_set_id=record["selection_start_set"]["start_set_id"],
    )
    final_episode, final_payload = evaluations[-1]
    final_summary = final_payload["summary"]
    stopped_at = int(trainer_summary["stopped_at_episode"])
    if final_episode != stopped_at:
        raise ValueError(
            f"Final evaluation episode {final_episode} does not match stopped_at_episode {stopped_at}"
        )
    if int(record["resolved_args"]["early_stop_patience_evals"]) == 0:
        expected_episodes = int(record["resolved_args"]["episodes"])
        if stopped_at != expected_episodes or bool(trainer_summary.get("stopped_early")):
            raise ValueError(f"Full-budget run stopped at episode {stopped_at}, expected {expected_episodes}")

    best_episode, best_payload = _best_evaluation(evaluations)
    best_summary = best_payload["summary"]
    checkpoint_path = run_dir / "checkpoints" / "best_by_completion.pt"
    if not checkpoint_path.is_file():
        raise ValueError(f"Missing selected checkpoint: {checkpoint_path}")

    last_five = [payload["summary"] for _, payload in evaluations[-5:]]
    nonzero_success = [
        episode
        for episode, payload in evaluations
        if float(payload["summary"]["success_rate"]) > 0.0
    ]
    full_success = [
        episode
        for episode, payload in evaluations
        if float(payload["summary"]["success_rate"]) >= 1.0 - 1e-12
    ]
    auc_points = [
        (episode, float(payload["summary"]["mean_completion_ratio"]))
        for episode, payload in evaluations
    ]

    return {
        "run_id": record["run_id"],
        "variant": record["variant"],
        "train_seed": int(record["train_seed"]),
        "run_dir": str(run_dir),
        "source_tree": record["source_tree"],
        "selection_start_set_id": record["selection_start_set"]["start_set_id"],
        "evaluation_count": len(evaluations),
        "final_evaluation_episode": final_episode,
        "final_completion": float(final_summary["mean_completion_ratio"]),
        "final_success": float(final_summary["success_rate"]),
        "final_steps": float(final_summary["mean_steps"]),
        "final_environment_return": float(final_summary["mean_environment_return"]),
        "completion_auc": _normalized_auc(auc_points),
        "first_eval_nonzero_success_episode": nonzero_success[0] if nonzero_success else None,
        "first_eval_full_success_episode": full_success[0] if full_success else None,
        "last_five_eval_success": float(mean(row["success_rate"] for row in last_five)),
        "last_five_eval_completion": float(mean(row["mean_completion_ratio"] for row in last_five)),
        "selected_checkpoint_episode": best_episode,
        "selected_checkpoint": str(checkpoint_path),
        "selected_completion": float(best_summary["mean_completion_ratio"]),
        "selected_success": float(best_summary["success_rate"]),
        "selected_steps": float(best_summary["mean_steps"]),
        "status_sha256": _sha256(Path(status["status_path"])) if status.get("status_path") else None,
        "summary_sha256": _sha256(summary_path),
    }


def _aggregate(rows: list[dict], variant_order: list[str]) -> list[dict]:
    output: list[dict] = []
    for variant in variant_order:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        aggregate: dict = {
            "variant": variant,
            "n": len(variant_rows),
            "seeds": [row["train_seed"] for row in variant_rows],
        }
        for metric in RUN_METRICS:
            values = [float(row[metric]) for row in variant_rows if row[metric] is not None]
            aggregate[metric] = {
                "n": len(values),
                "mean": float(mean(values)) if values else None,
                "sample_std": float(stdev(values)) if len(values) > 1 else None,
            }
        output.append(aggregate)
    return output


def _exact_sign_flip_p(differences: list[float]) -> float | None:
    """Backward-compatible wrapper for the shared exact test."""

    return exact_two_sided_sign_flip_p(differences)


def _paired_effects(rows: list[dict], variant_order: list[str], baseline: str = "full") -> list[dict]:
    by_variant_seed = {
        (row["variant"], int(row["train_seed"])): row
        for row in rows
    }
    baseline_seeds = sorted(
        int(row["train_seed"])
        for row in rows
        if row["variant"] == baseline
    )
    output: list[dict] = []
    for variant in variant_order:
        if variant == baseline:
            continue
        effect = {"baseline": baseline, "variant": variant, "metrics": {}}
        for metric, direction in PAIRED_METRICS.items():
            differences: list[float] = []
            seeds: list[int] = []
            for seed in baseline_seeds:
                baseline_row = by_variant_seed.get((baseline, seed))
                variant_row = by_variant_seed.get((variant, seed))
                if baseline_row is None or variant_row is None:
                    continue
                baseline_value = baseline_row[metric]
                variant_value = variant_row[metric]
                if baseline_value is None or variant_value is None:
                    continue
                if direction == "baseline_minus_variant":
                    difference = float(baseline_value) - float(variant_value)
                else:
                    difference = float(variant_value) - float(baseline_value)
                differences.append(difference)
                seeds.append(seed)
            effect["metrics"][metric] = {
                "direction": direction,
                "n": len(differences),
                "seeds": seeds,
                "differences": differences,
                "mean_difference_favoring_baseline": float(mean(differences)) if differences else None,
                "sample_std": float(stdev(differences)) if len(differences) > 1 else None,
                "positive_pairs": sum(value > 0 for value in differences),
                "zero_pairs": sum(value == 0 for value in differences),
                "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                "paired_seed_bootstrap_mean_difference_95ci": bootstrap_mean_ci(differences),
            }
        output.append(effect)
    auc_p_values = [
        effect["metrics"]["completion_auc"]["exact_two_sided_sign_flip_p"]
        for effect in output
    ]
    for effect, adjusted in zip(output, holm_adjust(auc_p_values)):
        effect["metrics"]["completion_auc"]["holm_adjusted_p"] = adjusted
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--records-dir", type=Path, default=Path("experiments/run_records"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/reports"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    records_dir = args.records_dir if args.records_dir.is_absolute() else repo_root / args.records_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    plan = _read_json(plan_path)

    completed_rows: list[dict] = []
    incomplete: list[dict] = []
    for record in plan["records"]:
        status_path = records_dir / f"{record['run_id']}.json"
        if not status_path.is_file():
            incomplete.append({"run_id": record["run_id"], "status": "missing"})
            continue
        status = _read_json(status_path)
        status["status_path"] = str(status_path)
        if status.get("status") != "completed":
            incomplete.append({"run_id": record["run_id"], "status": status.get("status")})
            continue
        for key in ("run_id", "variant", "train_seed", "source_tree", "spec_sha256"):
            if status.get(key) != record.get(key):
                raise ValueError(f"Run record mismatch for {record['run_id']}: {key}")
        run_dir = Path(status["run_dir"])
        completed_rows.append(summarize_run(record, status, run_dir))

    if incomplete and not args.allow_partial:
        preview = ", ".join(item["run_id"] for item in incomplete[:3])
        raise RuntimeError(
            f"Plan is incomplete ({len(incomplete)}/{plan['run_count']} runs); first missing: {preview}. "
            "Use --allow-partial only for operational inspection."
        )

    variant_order = list(dict.fromkeys(record["variant"] for record in plan["records"]))
    completed_rows.sort(key=lambda row: (variant_order.index(row["variant"]), row["train_seed"]))
    aggregate = _aggregate(completed_rows, variant_order)
    paired_effects = _paired_effects(completed_rows, variant_order)
    payload = {
        "schema_version": 1,
        "experiment_name": plan["experiment_name"],
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source_tree": plan["source_tree"],
        "planned_runs": int(plan["run_count"]),
        "completed_runs": len(completed_rows),
        "incomplete_runs": incomplete,
        "scope": "checkpoint-selection start set only; unseen final-test metrics are reported separately",
        "definitions": {
            "completion_auc": "trapezoidal area of selection-set mean completion from (episode 0, value 0), normalized by the final evaluation episode",
            "first_eval_nonzero_success_episode": "first scheduled selection evaluation with success_rate > 0; null if never observed",
            "first_eval_full_success_episode": "first scheduled selection evaluation with success_rate = 1; null if never observed",
            "last_five_eval_success": "mean selection-set success rate across the final five scheduled evaluations",
            "sample_std": "sample standard deviation across independent training seeds (ddof=1); null when n < 2",
            "final_environment_return": "raw DRAMA environment return; objective_return is deliberately excluded",
            "paired_effects": "seed-matched differences oriented so positive values favor Full; exact two-sided sign-flip p-values are descriptive and uncorrected for multiple comparisons",
            "bootstrap_ci": f"percentile 95% CI from {BOOTSTRAP_RESAMPLES} paired training-seed resamples with seed {BOOTSTRAP_SEED}",
            "auc_multiplicity": "Full-versus-four-ablation completion-AUC sign-flip p-values additionally receive Holm adjustment",
        },
        "per_run": completed_rows,
        "aggregate": aggregate,
        "paired_effects": paired_effects,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{plan['experiment_name']}_aggregate.json"
    csv_path = output_dir / f"{plan['experiment_name']}_per_run.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, completed_rows)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "completed_runs": len(completed_rows),
                "incomplete_runs": len(incomplete),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
