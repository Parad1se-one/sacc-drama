#!/usr/bin/env python3
"""Strict seven-method Table 6 statistics on the common fresh holdout."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, stdev
import sys


SEEDS = (42, 43, 44, 45, 46)
METHODS = ("sacc", "qmix", "vdn", "mappo", "ippo", "happo", "mahrm")
METRICS = ("mean_completion_ratio", "success_rate", "mean_steps", "mean_environment_return")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("easy", "hard"), required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = repo / "experiments/mahrm_direct_20260924"
    sys.path.insert(0, str(repo))
    from scripts.seed_statistics import bootstrap_mean_ci, exact_two_sided_sign_flip_p, interquartile_mean
    from scripts.summarize_final_eval import collect_rows

    difficulty = args.difficulty
    plan_path = root / f"plans/table6_mahrm_t035_{difficulty}_holdout_v1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original_rows = collect_rows(plan, root / f"eval_records/{difficulty}")
    test_path = root / f"protocols/{difficulty}_test.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    expected_starts = test["starts"]
    if len(expected_starts) != 100 or len({str(s) for s in expected_starts}) != 100:
        raise RuntimeError("Final test does not have 100 distinct starts")

    rows = []
    for source in original_rows:
        if source["method"] not in METHODS[:-1] or source["episodes"] != 100:
            raise RuntimeError("Unexpected original-method row")
        rows.append({k: source[k] for k in ("method", "train_seed", *METRICS)})

    for seed in SEEDS:
        train_path = root / f"records/{difficulty}_seed{seed}.json"
        train = json.loads(train_path.read_text(encoding="utf-8"))
        result_path = root / f"eval_records/{difficulty}_seed{seed}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = root / f"checkpoints/{difficulty}_seed{seed}.pkl"
        if train["status"] != "completed" or result["status"] != "completed":
            raise RuntimeError(f"Incomplete MAHRM seed {seed}")
        if train["trained_episodes"] != (3000 if difficulty == "easy" else 1200):
            raise RuntimeError("MAHRM training budget mismatch")
        if train["checkpoint_sha256"] != result["checkpoint_sha256"] or digest(checkpoint) != train["checkpoint_sha256"]:
            raise RuntimeError("MAHRM checkpoint hash mismatch")
        if result["start_set_id"] != test["start_set_id"] or result["start_set_sha256"] != digest(test_path):
            raise RuntimeError("MAHRM test manifest mismatch")
        if len(result["episodes"]) != 100:
            raise RuntimeError("MAHRM episode count mismatch")
        max_reward = 8 if difficulty == "easy" else 40
        for episode, starts in zip(result["episodes"], expected_starts, strict=True):
            if episode["start_positions"] != starts or not 0 <= episode["completion_ratio"] <= 1:
                raise RuntimeError("MAHRM episode start or completion invalid")
            if episode["environment_return"] > max_reward + 1e-9:
                raise RuntimeError("MAHRM environment reward exceeds physical upper bound")
        rows.append({"method": "mahrm", "train_seed": seed,
                     "mean_completion_ratio": float(mean(e["completion_ratio"] for e in result["episodes"])),
                     "success_rate": float(mean(e["success"] for e in result["episodes"])),
                     "mean_steps": float(mean(e["steps"] for e in result["episodes"])),
                     "mean_environment_return": float(mean(e["environment_return"] for e in result["episodes"]))})

    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 rows, received {len(rows)}")
    by_method = {method: {r["train_seed"]: r for r in rows if r["method"] == method} for method in METHODS}
    if any(set(group) != set(SEEDS) for group in by_method.values()):
        raise RuntimeError("One or more methods lack the five frozen training seeds")

    method_stats = {}
    paired = {}
    for method in METHODS:
        method_stats[method] = {}
        for metric in METRICS:
            points = [by_method[method][seed][metric] for seed in SEEDS]
            method_stats[method][metric] = {
                "mean": mean(points), "sample_sd": stdev(points), "seed_points": dict(zip(SEEDS, points)),
                "bootstrap_95ci": bootstrap_mean_ci(points), "iqm": interquartile_mean(points),
            }
        if method != "sacc":
            paired[method] = {}
            for metric in METRICS:
                differences = [by_method["sacc"][seed][metric] - by_method[method][seed][metric] for seed in SEEDS]
                paired[method][metric] = {
                    "mean_difference_sacc_minus_method": mean(differences),
                    "seed_differences": dict(zip(SEEDS, differences)),
                    "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(differences),
                    "paired_bootstrap_95ci": bootstrap_mean_ci(differences),
                }
    report = {"schema_version": 1, "difficulty": difficulty, "methods": METHODS,
              "training_seeds": SEEDS, "test_episodes_per_seed": 100,
              "statistical_unit": "independent_training_seed",
              "test_start_set_id": test["start_set_id"], "test_manifest_sha256": digest(test_path),
              "six_method_plan_sha256": digest(plan_path), "rows": rows,
              "method_statistics": method_stats, "paired_vs_sacc": paired}
    output = root / f"summaries/table6_mahrm_t035_{difficulty}_holdout_v1.json"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite summary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "train_seed", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": str(output), "sha256": digest(output), "methods": len(method_stats), "records": len(rows)}))


if __name__ == "__main__":
    main()

