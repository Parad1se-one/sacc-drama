from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .env import DramaBenchmarkEnv
from .specs import ACTION_STR, TASK_STAGES, BENCHMARK_SPECS, get_spec_data


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    meta: dict
    config: dict

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.config["grid_width"], self.config["grid_height"]

    @property
    def num_agents(self) -> int:
        return self.config["num_robots"]

    @property
    def task_cell_count(self) -> int:
        floors = self.config["floors"]
        return sum(len(floors[key]) for key in ("to_distribute", "to_vibrate", "to_level", "to_cover"))

    @property
    def total_stage_tasks(self) -> int:
        return self.task_cell_count * len(TASK_STAGES)

    @property
    def obstacle_count(self) -> int:
        obstacles = self.config["obstacles"]
        return len(obstacles["static"]) + len(obstacles["dynamic"])

    @property
    def region_count(self) -> int:
        return len(self.config.get("regions", {}))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meta": self.meta,
            "config": self.config,
            "derived": {
                "grid_shape": self.grid_shape,
                "num_agents": self.num_agents,
                "task_cell_count": self.task_cell_count,
                "total_stage_tasks": self.total_stage_tasks,
                "obstacle_count": self.obstacle_count,
                "region_count": self.region_count,
                "action_space": ACTION_STR,
                "task_stages": list(TASK_STAGES),
            },
        }


def list_benchmarks() -> list[str]:
    return list(BENCHMARK_SPECS.keys())


def get_benchmark(name: str) -> BenchmarkSpec:
    data = get_spec_data(name)
    config = data["config"]
    meta = data["meta"]
    meta.setdefault("total_grids", sum(len(config["floors"][key]) for key in ("to_distribute", "to_vibrate", "to_level", "to_cover")))
    meta.setdefault("num_robots", config["num_robots"])
    meta.setdefault("grid_shape", [config["grid_width"], config["grid_height"]])
    meta.setdefault("obstacle_count", len(config["obstacles"]["static"]) + len(config["obstacles"]["dynamic"]))
    return BenchmarkSpec(name=name.lower(), meta=meta, config=config)


def get_config(name: str) -> dict:
    return get_benchmark(name).config


def get_metadata(name: str) -> dict:
    return get_benchmark(name).meta


def make_env(name: str, *, seed: int | None = None, start_positions=None) -> DramaBenchmarkEnv:
    env = DramaBenchmarkEnv(config=get_config(name))
    env.reset(seed=seed, start_positions=start_positions)
    return env


def export_manifest(path: str | Path) -> Path:
    output_path = Path(path)
    payload = {name: get_benchmark(name).to_dict() for name in list_benchmarks()}
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
