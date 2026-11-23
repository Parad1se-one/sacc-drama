"""Deterministic start-state generation and versioned start-set manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .specs import get_spec_data

START_SET_SCHEMA_VERSION = 1
Position = tuple[int, int]
StartConfiguration = tuple[Position, ...]


def robot_order(config: dict) -> list[str]:
    """Return the role order used by environment joint actions and start sets."""

    roles: list[str] = []
    for robot_type, info in config["robots"].items():
        roles.extend([robot_type] * len(info["init_positions"]))
    return roles


def fixed_start_positions(config: dict) -> StartConfiguration:
    positions: list[Position] = []
    for info in config["robots"].values():
        positions.extend(tuple(map(int, pos)) for pos in info["init_positions"])
    return tuple(positions)


def task_cells(config: dict) -> set[Position]:
    floors = config["floors"]
    cells: set[Position] = set()
    for key in ("to_distribute", "to_vibrate", "to_level", "to_cover"):
        cells.update(tuple(map(int, pos)) for pos in floors[key])
    return cells


def eligible_start_cells(config: dict, *, exclude_task_cells: bool = True) -> list[Position]:
    blocked = {
        tuple(map(int, pos))
        for pos in config["obstacles"]["static"] + config["obstacles"]["dynamic"]
    }
    if exclude_task_cells:
        blocked.update(task_cells(config))
    width = int(config["grid_width"])
    height = int(config["grid_height"])
    return [(x, y) for x in range(width) for y in range(height) if (x, y) not in blocked]


def validate_start_positions(
    config: dict,
    positions: Sequence[Sequence[int]],
    *,
    exclude_task_cells: bool = False,
    min_pairwise_manhattan: int = 1,
) -> StartConfiguration:
    """Validate and normalize one role-ordered robot start configuration."""

    normalized: StartConfiguration = tuple((int(pos[0]), int(pos[1])) for pos in positions)
    expected = int(config["num_robots"])
    if len(normalized) != expected:
        raise ValueError(f"Expected {expected} robot positions, received {len(normalized)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Robot start positions must be unique")

    width = int(config["grid_width"])
    height = int(config["grid_height"])
    obstacles = {
        tuple(map(int, pos))
        for pos in config["obstacles"]["static"] + config["obstacles"]["dynamic"]
    }
    forbidden_tasks = task_cells(config) if exclude_task_cells else set()
    for pos in normalized:
        x, y = pos
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Robot start position {pos} is outside the {width}x{height} grid")
        if pos in obstacles:
            raise ValueError(f"Robot start position {pos} overlaps an obstacle")
        if pos in forbidden_tasks:
            raise ValueError(f"Robot start position {pos} overlaps a construction target")

    minimum = int(min_pairwise_manhattan)
    if minimum < 1:
        raise ValueError("min_pairwise_manhattan must be at least 1")
    for index, first in enumerate(normalized):
        for second in normalized[index + 1 :]:
            distance = abs(first[0] - second[0]) + abs(first[1] - second[1])
            if distance < minimum:
                raise ValueError(
                    f"Robot starts {first} and {second} have Manhattan distance {distance}, "
                    f"below the required minimum {minimum}"
                )
    return normalized


def sample_start_positions(
    config: dict,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    exclude_task_cells: bool = True,
    min_pairwise_manhattan: int = 1,
    excluded: Iterable[StartConfiguration] = (),
    max_attempts: int = 100_000,
) -> StartConfiguration:
    """Sample one valid, role-ordered start without mutating global RNG state."""

    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    cells = eligible_start_cells(config, exclude_task_cells=exclude_task_cells)
    num_robots = int(config["num_robots"])
    if len(cells) < num_robots:
        raise ValueError(f"Only {len(cells)} eligible cells are available for {num_robots} robots")

    excluded_keys = {tuple(tuple(pos) for pos in starts) for starts in excluded}
    for _ in range(max_attempts):
        indices = generator.choice(len(cells), size=num_robots, replace=False)
        candidate = tuple(cells[int(index)] for index in indices)
        if candidate in excluded_keys:
            continue
        try:
            return validate_start_positions(
                config,
                candidate,
                exclude_task_cells=exclude_task_cells,
                min_pairwise_manhattan=min_pairwise_manhattan,
            )
        except ValueError:
            continue
    raise RuntimeError(
        "Unable to sample a new valid start configuration; reduce the requested count or distance constraint"
    )


def default_min_pairwise_manhattan(difficulty: str) -> int:
    """Use uniqueness on Easy and one-cell separation on harder splits."""

    return 2 if difficulty.lower() in {"hard", "large"} else 1


def _start_set_digest(payload_without_id: Mapping) -> str:
    serialized = json.dumps(payload_without_id, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _build_payload(
    *,
    difficulty: str,
    split: str,
    generator_seed: int,
    starts: Sequence[StartConfiguration],
    exclude_task_cells: bool,
    min_pairwise_manhattan: int,
    sampling: str = "role_ordered_without_replacement_within_episode",
) -> dict:
    config = get_spec_data(difficulty)["config"]
    payload = {
        "schema_version": START_SET_SCHEMA_VERSION,
        "benchmark": "DRAMA",
        "difficulty": difficulty.lower(),
        "split": str(split),
        "generator_seed": int(generator_seed),
        "count": len(starts),
        "robot_order": robot_order(config),
        "fixed_start_positions": [list(pos) for pos in fixed_start_positions(config)],
        "generation": {
            "exclude_task_cells": bool(exclude_task_cells),
            "min_pairwise_manhattan": int(min_pairwise_manhattan),
            "sampling": str(sampling),
            "duplicates_across_episodes": False,
        },
        "starts": [[list(pos) for pos in start] for start in starts],
    }
    payload["start_set_id"] = _start_set_digest(payload)
    return payload


def build_fixed_start_set(difficulty: str, *, split: str = "fixed_test") -> dict:
    """Build a versioned one-layout manifest for the benchmark's fixed start."""

    difficulty = difficulty.lower()
    config = get_spec_data(difficulty)["config"]
    start = validate_start_positions(
        config,
        fixed_start_positions(config),
        exclude_task_cells=False,
        min_pairwise_manhattan=1,
    )
    return _build_payload(
        difficulty=difficulty,
        split=split,
        generator_seed=0,
        starts=[start],
        exclude_task_cells=False,
        min_pairwise_manhattan=1,
        sampling="benchmark_fixed_role_order",
    )


def generate_disjoint_start_sets(
    difficulty: str,
    split_counts: Mapping[str, int],
    *,
    seed: int,
    exclude_task_cells: bool = True,
    min_pairwise_manhattan: int | None = None,
) -> dict[str, dict]:
    """Generate deterministic, mutually disjoint start sets for named splits."""

    difficulty = difficulty.lower()
    config = get_spec_data(difficulty)["config"]
    minimum = (
        default_min_pairwise_manhattan(difficulty)
        if min_pairwise_manhattan is None
        else int(min_pairwise_manhattan)
    )
    rng = np.random.default_rng(seed)
    seen: set[StartConfiguration] = set()
    payloads: dict[str, dict] = {}
    for split, count in split_counts.items():
        if int(count) <= 0:
            raise ValueError(f"Split {split!r} must request at least one start")
        starts: list[StartConfiguration] = []
        for _ in range(int(count)):
            start = sample_start_positions(
                config,
                rng=rng,
                exclude_task_cells=exclude_task_cells,
                min_pairwise_manhattan=minimum,
                excluded=seen,
            )
            starts.append(start)
            seen.add(start)
        payloads[str(split)] = _build_payload(
            difficulty=difficulty,
            split=str(split),
            generator_seed=seed,
            starts=starts,
            exclude_task_cells=exclude_task_cells,
            min_pairwise_manhattan=minimum,
        )
    return payloads


def validate_start_set(payload: Mapping, *, expected_difficulty: str | None = None) -> dict:
    normalized = dict(payload)
    start_set_id = normalized.pop("start_set_id", None)
    if start_set_id is None:
        raise ValueError("Start-set manifest is missing start_set_id")
    if int(normalized.get("schema_version", -1)) != START_SET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported start-set schema version: {normalized.get('schema_version')}")
    if normalized.get("benchmark") != "DRAMA":
        raise ValueError("Start-set manifest is not for the DRAMA benchmark")
    difficulty = str(normalized["difficulty"]).lower()
    if expected_difficulty is not None and difficulty != expected_difficulty.lower():
        raise ValueError(f"Start set is for {difficulty}, expected {expected_difficulty.lower()}")
    if _start_set_digest(normalized) != start_set_id:
        raise ValueError("Start-set digest does not match its contents")

    config = get_spec_data(difficulty)["config"]
    generation = dict(normalized["generation"])
    starts = [
        validate_start_positions(
            config,
            positions,
            exclude_task_cells=bool(generation["exclude_task_cells"]),
            min_pairwise_manhattan=int(generation["min_pairwise_manhattan"]),
        )
        for positions in normalized["starts"]
    ]
    if len(starts) != int(normalized["count"]):
        raise ValueError("Start-set count does not match the number of starts")
    if len(set(starts)) != len(starts):
        raise ValueError("Start-set manifest contains duplicate configurations")
    if list(normalized["robot_order"]) != robot_order(config):
        raise ValueError("Start-set robot order does not match the benchmark configuration")

    normalized["starts"] = starts
    normalized["start_set_id"] = start_set_id
    return normalized


def load_start_set(path: str | Path, *, expected_difficulty: str | None = None) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_start_set(payload, expected_difficulty=expected_difficulty)


def write_start_set(path: str | Path, payload: Mapping) -> Path:
    validated = validate_start_set(payload)
    serializable = dict(validated)
    serializable["starts"] = [[list(pos) for pos in start] for start in validated["starts"]]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
