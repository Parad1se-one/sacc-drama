#!/usr/bin/env python3
"""Freeze MAHRM training, selection and seven-method fresh test starts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


SEEDS = (42, 43, 44, 45, 46)
BUDGET = {"easy": 3000, "hard": 1200}
GENERATOR_SEED = {"easy": 2026092401, "hard": 2026092402}


def positions(starts):
    return tuple(tuple(map(int, p)) for p in starts)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing protocol: {output}")
    sys.path.insert(0, str(repo / "src/sacc-drama/DRAMA"))
    from DRAMA.specs import get_spec_data
    from DRAMA.starts import _build_payload, eligible_start_cells, validate_start_positions

    previous = {difficulty: set() for difficulty in BUDGET}
    old_manifest_hashes = {}
    for path in sorted((repo / "experiments/protocols").rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        difficulty = data.get("difficulty")
        if difficulty in previous and "starts" in data:
            previous[difficulty].update(positions(s) for s in data["starts"])
            old_manifest_hashes[str(path.relative_to(repo))] = digest(path)

    # All six prior methods share seeded baseline training starts.  SACC's
    # start stream is read from its provenance rather than inferred.
    sacc_metrics_hashes = {}
    for difficulty, budget in BUDGET.items():
        config = get_spec_data(difficulty)["config"]
        minimum = 2 if difficulty == "hard" else 1
        from DRAMA.starts import sample_start_positions
        for generated_seed in range(43, 47 + budget):
            previous[difficulty].add(sample_start_positions(config, seed=generated_seed,
                exclude_task_cells=True, min_pairwise_manhattan=minimum))
        spec_path = repo / f"experiments/specs/original_submission_v8_{difficulty}_all_methods_5seed_v1.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        for entry in spec["checkpoints"]:
            if entry["method"] != "sacc":
                continue
            record_path = Path(entry["training_record"])
            record = json.loads(record_path.read_text(encoding="utf-8"))
            metrics = Path(record["run_dir"]) / "training_metrics.jsonl"
            lines = metrics.read_text(encoding="utf-8").splitlines()
            if len(lines) != budget:
                raise RuntimeError(f"Unexpected training length: {metrics}: {len(lines)}")
            for line in lines:
                previous[difficulty].add(positions(json.loads(line)["start_positions"]))
            sacc_metrics_hashes[f"{difficulty}_{entry['seed']}"] = digest(metrics)

    output.mkdir(parents=True)
    report = {"schema_version": 1, "purpose": "MAHRM direct comparison and seven-method T0.35 Table 6",
              "generation_seeds": GENERATOR_SEED, "budgets": BUDGET,
              "prior_manifest_hashes": old_manifest_hashes, "sacc_metrics_hashes": sacc_metrics_hashes,
              "manifests": {}}
    for difficulty, budget in BUDGET.items():
        config = get_spec_data(difficulty)["config"]
        minimum = 2 if difficulty == "hard" else 1
        cells = eligible_start_cells(config, exclude_task_cells=True)
        rng = np.random.default_rng(GENERATOR_SEED[difficulty])
        seen = set(previous[difficulty])
        existing = set(previous[difficulty])
        counts = {**{f"train_seed{seed}": budget for seed in SEEDS}, "selection": 20, "test": 100}
        for split, count in counts.items():
            starts = []
            for _ in range(count):
                for _attempt in range(100_000):
                    indices = rng.choice(len(cells), size=4, replace=False)
                    candidate = tuple(cells[int(i)] for i in indices)
                    if candidate in seen:
                        continue
                    try:
                        validate_start_positions(config, candidate, exclude_task_cells=True,
                            min_pairwise_manhattan=minimum)
                    except ValueError:
                        continue
                    seen.add(candidate)
                    starts.append(candidate)
                    break
                else:
                    raise RuntimeError(f"Start space exhausted for {difficulty}/{split}")
            payload = _build_payload(difficulty=difficulty, split=f"mahrm_20260924_{split}",
                generator_seed=GENERATOR_SEED[difficulty], starts=starts,
                exclude_task_cells=True, min_pairwise_manhattan=minimum)
            path = output / f"{difficulty}_{split}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            report["manifests"][f"{difficulty}_{split}"] = {
                "path": str(path), "sha256": digest(path), "start_set_id": payload["start_set_id"],
                "count": count, "prior_overlap": len(set(starts) & existing),
            }
        if any(value["prior_overlap"] for key, value in report["manifests"].items()
               if key.startswith(difficulty + "_")):
            raise RuntimeError("Protocol overlaps an earlier start")
    report_path = output / "protocol_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(report_path), "sha256": digest(report_path),
                      "manifest_count": len(report["manifests"])}, indent=2))


if __name__ == "__main__":
    main()

