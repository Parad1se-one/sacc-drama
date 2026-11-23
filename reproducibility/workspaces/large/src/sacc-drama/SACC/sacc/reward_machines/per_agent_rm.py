"""Per-agent reward machines for the fixed-role DRAMA robots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

TASK_FRONTIERS = (
    "floors_to_distribute",
    "floors_to_vibrate",
    "floors_to_level",
    "floors_to_cover",
)

ROLE_TO_STAGE = {
    "material_distributer": "distribute_floor",
    "vibrator": "vibrate_floor",
    "leveler": "level_floor",
    "film_cover": "cover_surface",
}

STAGE_TO_FRONTIER = {
    "distribute_floor": "floors_to_distribute",
    "vibrate_floor": "floors_to_vibrate",
    "level_floor": "floors_to_level",
    "cover_surface": "floors_to_cover",
}

STAGE_INDEX = {
    "distribute_floor": 0,
    "vibrate_floor": 1,
    "level_floor": 2,
    "cover_surface": 3,
}


@dataclass
class LocalRMConfig:
    reward_stage_event: float = 1.0
    reward_stage_done: float = 1.5
    reward_all_done: float = 4.0
    penalty_idle: float = -0.01
    penalty_no_effect: float = -0.10
    penalty_blocked: float = -0.05
    distance_shaping_coeff: float = 0.03


class LocalRewardMachine:
    """A lightweight reward machine bound to one fixed DRAMA robot role."""

    obs_dim = 9

    def __init__(self, agent_idx: int, robot_type: str, total_cells: int, config: LocalRMConfig):
        if robot_type not in ROLE_TO_STAGE:
            raise ValueError(f"Unsupported DRAMA robot type: {robot_type}")
        self.agent_idx = int(agent_idx)
        self.robot_type = robot_type
        self.stage_name = ROLE_TO_STAGE[robot_type]
        self.stage_index = STAGE_INDEX[self.stage_name]
        self.frontier_attr = STAGE_TO_FRONTIER[self.stage_name]
        self.total_cells = max(int(total_cells), 1)
        self.config = config
        self.prev_distance: float | None = None
        self.prev_stage_done = False

    def reset(self, env) -> None:
        metrics = self._metrics(env)
        self.prev_distance = metrics["nearest_distance"]
        self.prev_stage_done = metrics["stage_done"]

    def step(self, event: str | None, env) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        before_done = self.prev_stage_done
        before_distance = self.prev_distance

        metrics = self._metrics(env)
        available_count = metrics["available_count"]
        tracking_count = metrics["tracking_count"]
        nearest_distance = metrics["nearest_distance"]
        stage_done = metrics["stage_done"]

        if event == self.stage_name:
            reward += self.config.reward_stage_event
        elif event == "blocked":
            reward += self.config.penalty_blocked
        elif event == "no effect":
            reward += self.config.penalty_no_effect
        elif event == "stayed" and available_count > 0:
            reward += self.config.penalty_idle

        if before_distance is not None and nearest_distance is not None and tracking_count > 0:
            reward += self.config.distance_shaping_coeff * (before_distance - nearest_distance)

        if stage_done and not before_done:
            reward += self.config.reward_stage_done

        self.prev_distance = nearest_distance
        self.prev_stage_done = stage_done
        return float(reward), metrics

    def observe(self, env) -> np.ndarray:
        metrics = self._metrics(env)
        stage_index_norm = self.stage_index / max(len(TASK_FRONTIERS) - 1, 1)
        available_ratio = metrics["available_count"] / self.total_cells
        progressed_ratio = metrics["progressed_count"] / self.total_cells
        upstream_clear_ratio = 1.0 - (metrics["upstream_count"] / self.total_cells)
        wait_flag = 1.0 if metrics["wait_flag"] else 0.0
        on_target = 1.0 if metrics["on_target"] else 0.0
        nearest_distance_norm = metrics["nearest_distance_norm"]
        stage_done = 1.0 if metrics["stage_done"] else 0.0
        guidance_active = 1.0 if metrics["guidance_active"] else 0.0
        return np.array(
            [
                stage_index_norm,
                available_ratio,
                progressed_ratio,
                upstream_clear_ratio,
                wait_flag,
                on_target,
                nearest_distance_norm,
                guidance_active,
                stage_done,
            ],
            dtype=np.float32,
        )

    def _metrics(self, env) -> dict[str, Any]:
        frontier = getattr(env, self.frontier_attr)
        robot_pos = tuple(env.robots[self.agent_idx]["pos"])
        available_count = len(frontier)
        upstream_count = self._count_upstream(env)
        progressed_count = self._count_progressed(env)
        tracking_frontier, guidance_active = self._tracking_frontier(env, frontier)
        tracking_count = len(tracking_frontier)
        nearest_distance = self._nearest_distance(robot_pos, tracking_frontier, env)
        max_distance = max(env.width * env.height - 1, 1)
        nearest_distance_norm = 1.0 if nearest_distance is None else min(nearest_distance / max_distance, 1.0)
        stage_done = available_count == 0 and upstream_count == 0
        wait_flag = available_count == 0 and not stage_done
        return {
            "available_count": available_count,
            "tracking_count": tracking_count,
            "upstream_count": upstream_count,
            "progressed_count": progressed_count,
            "nearest_distance": nearest_distance,
            "nearest_distance_norm": nearest_distance_norm,
            "on_target": robot_pos in tracking_frontier,
            "wait_flag": wait_flag,
            "guidance_active": guidance_active,
            "stage_done": stage_done,
        }

    def _tracking_frontier(self, env, active_frontier: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], bool]:
        if active_frontier:
            return set(active_frontier), False

        if self.stage_index == 0:
            return set(), False

        # When the stage is locked, guide the agent toward the closest releasing
        # upstream frontier instead of the union of all upstream cells.
        for frontier_attr in reversed(TASK_FRONTIERS[: self.stage_index]):
            frontier = set(getattr(env, frontier_attr))
            if frontier:
                return frontier, True
        return set(), False

    def _count_upstream(self, env) -> int:
        count = 0
        for frontier_attr in TASK_FRONTIERS[: self.stage_index]:
            count += len(getattr(env, frontier_attr))
        return count

    def _count_progressed(self, env) -> int:
        count = len(env.floors_completed)
        for frontier_attr in TASK_FRONTIERS[self.stage_index + 1 :]:
            count += len(getattr(env, frontier_attr))
        return count

    @staticmethod
    def _nearest_distance(robot_pos: tuple[int, int], frontier: set[tuple[int, int]], env) -> int | None:
        if not frontier:
            return None
        if robot_pos in frontier:
            return 0

        traversable = np.ones((env.width, env.height), dtype=bool)
        for x, y in env.obstacles | env.dynamic_obstacles:
            traversable[x, y] = False

        queue: deque[tuple[tuple[int, int], int]] = deque([(robot_pos, 0)])
        visited = {robot_pos}
        while queue:
            (x, y), distance = queue.popleft()
            for nx, ny in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                if not (0 <= nx < env.width and 0 <= ny < env.height):
                    continue
                if (nx, ny) in visited or not traversable[nx, ny]:
                    continue
                if (nx, ny) in frontier:
                    return distance + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
        return None


class PerAgentRewardMachineSuite:
    """Manager that owns one local reward machine for each DRAMA agent."""

    def __init__(self, config: dict, **kwargs):
        rm_config = LocalRMConfig(**kwargs)
        total_cells = self._count_total_cells(config)
        agent_types = self._flatten_agent_types(config)
        self.agent_rms = [
            LocalRewardMachine(agent_idx=i, robot_type=robot_type, total_cells=total_cells, config=rm_config)
            for i, robot_type in enumerate(agent_types)
        ]
        self.config = rm_config
        self.obs_dim = LocalRewardMachine.obs_dim

    def reset(self, env) -> None:
        for rm in self.agent_rms:
            rm.reset(env)

    def step(self, events: list[str | None], env) -> tuple[list[float], list[np.ndarray], list[dict[str, Any]]]:
        rewards: list[float] = []
        obs_list: list[np.ndarray] = []
        debug: list[dict[str, Any]] = []
        for rm, event in zip(self.agent_rms, events):
            reward, metrics = rm.step(event, env)
            rewards.append(reward)
            obs_list.append(rm.observe(env))
            debug.append(metrics)

        if env.is_all_done():
            bonus = self.config.reward_all_done / max(len(rewards), 1)
            rewards = [reward + bonus for reward in rewards]

        return rewards, obs_list, debug

    def get_obs_list(self, env) -> list[np.ndarray]:
        return [rm.observe(env) for rm in self.agent_rms]

    def get_global_progress(self, env) -> np.ndarray:
        return np.concatenate(self.get_obs_list(env), axis=0).astype(np.float32)

    @staticmethod
    def _count_total_cells(config: dict) -> int:
        floors = config["floors"]
        return sum(len(floors[key]) for key in ("to_distribute", "to_vibrate", "to_level", "to_cover"))

    @staticmethod
    def _flatten_agent_types(config: dict) -> list[str]:
        flattened: list[str] = []
        for robot_type, info in config["robots"].items():
            for _ in info["init_positions"]:
                flattened.append(robot_type)
        return flattened
