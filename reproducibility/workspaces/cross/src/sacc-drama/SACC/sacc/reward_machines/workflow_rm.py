"""Reward-machine interfaces for configurable workflow DAGs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WorkflowRMConfig:
    reward_stage_event: float = 1.0
    reward_stage_done: float = 1.5
    reward_all_done: float = 4.0
    penalty_idle: float = -0.01
    penalty_no_effect: float = -0.10
    penalty_blocked: float = -0.05
    distance_shaping_coeff: float = 0.03


def _nearest_distance(robot_pos, frontier, env) -> int | None:
    if not frontier:
        return None
    if robot_pos in frontier:
        return 0
    traversable = np.ones((env.width, env.height), dtype=bool)
    for x, y in env.obstacles | env.dynamic_obstacles:
        traversable[x, y] = False
    queue = deque([(robot_pos, 0)])
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


class WorkflowLocalRewardMachine:
    """Role-conditioned SACC progress abstraction for one workflow agent."""

    obs_dim = 9

    def __init__(self, agent_idx: int, role: str, env, config: WorkflowRMConfig):
        self.agent_idx = int(agent_idx)
        self.role = role
        self.assigned_nodes = tuple(env.role_to_nodes[role])
        if not self.assigned_nodes:
            raise ValueError(f"Workflow role {role!r} has no assigned nodes")
        self.config = config
        self.total_assignments = max(len(env.task_cells) * len(self.assigned_nodes), 1)
        self.prev_distance: float | None = None
        self.prev_stage_done = False

    def reset(self, env) -> None:
        metrics = self._metrics(env)
        self.prev_distance = metrics["nearest_distance"]
        self.prev_stage_done = metrics["stage_done"]

    def step(self, event: str | None, env) -> tuple[float, dict[str, Any]]:
        metrics = self._metrics(env)
        reward = 0.0
        valid_events = {
            str(env.node_by_id[node_id].get("event", f"complete:{node_id}"))
            for node_id in self.assigned_nodes
        }
        if event in valid_events:
            reward += self.config.reward_stage_event
        elif event == "blocked":
            reward += self.config.penalty_blocked
        elif event == "no effect":
            reward += self.config.penalty_no_effect
        elif event == "stayed" and metrics["available_count"] > 0:
            reward += self.config.penalty_idle
        if (
            self.prev_distance is not None
            and metrics["nearest_distance"] is not None
            and metrics["tracking_count"] > 0
        ):
            reward += self.config.distance_shaping_coeff * (
                self.prev_distance - metrics["nearest_distance"]
            )
        if metrics["stage_done"] and not self.prev_stage_done:
            reward += self.config.reward_stage_done
        self.prev_distance = metrics["nearest_distance"]
        self.prev_stage_done = metrics["stage_done"]
        return float(reward), metrics

    def observe(self, env) -> np.ndarray:
        metrics = self._metrics(env)
        denominator = max(len(env.topological_order) - 1, 1)
        return np.array(
            [
                metrics["target_node_index"] / denominator,
                metrics["available_count"] / self.total_assignments,
                metrics["progressed_count"] / self.total_assignments,
                metrics["upstream_clear_ratio"],
                float(metrics["wait_flag"]),
                float(metrics["on_target"]),
                metrics["nearest_distance_norm"],
                float(metrics["guidance_active"]),
                float(metrics["stage_done"]),
            ],
            dtype=np.float32,
        )

    def _metrics(self, env) -> dict[str, Any]:
        active_by_node = {
            node_id: set(env.frontiers[node_id]) for node_id in self.assigned_nodes
        }
        active_cells: set[tuple[int, int]] = set()
        for frontier in active_by_node.values():
            active_cells.update(frontier)
        available_count = sum(len(frontier) for frontier in active_by_node.values())
        progressed_count = 0
        for completed in env.completed_by_cell.values():
            progressed_count += sum(node_id in completed for node_id in self.assigned_nodes)
        pending_assignments = [
            (node_id, cell)
            for node_id in self.assigned_nodes
            for cell, completed in env.completed_by_cell.items()
            if node_id not in completed
        ]
        stage_done = not pending_assignments
        active_nodes = [node_id for node_id, cells in active_by_node.items() if cells]
        if active_nodes:
            target_node_index = sum(
                env.node_index[node_id] * len(active_by_node[node_id]) for node_id in active_nodes
            ) / available_count
        else:
            pending_nodes = {node_id for node_id, _ in pending_assignments}
            target_node = min(
                pending_nodes or set(self.assigned_nodes), key=env.node_index.__getitem__
            )
            target_node_index = float(env.node_index[target_node])

        dependency_total = 0
        satisfied_dependencies = 0
        for node_id, cell in pending_assignments:
            completed = env.completed_by_cell[cell]
            for predecessor in env.predecessors[node_id]:
                dependency_total += 1
                satisfied_dependencies += int(predecessor in completed)
        upstream_clear_ratio = (
            1.0 if dependency_total == 0 else satisfied_dependencies / dependency_total
        )

        tracking = set(active_cells)
        guidance_active = False
        if not tracking and not stage_done:
            guidance_active = True
            releasing_nodes: set[str] = set()
            for node_id, cell in pending_assignments:
                completed = env.completed_by_cell[cell]
                for predecessor in env.predecessors[node_id] - completed:
                    releasing_nodes.update(
                        self._nearest_releasing_nodes(predecessor, cell, env)
                    )
            for node_id in releasing_nodes:
                tracking.update(env.frontiers[node_id])
        robot_pos = tuple(env.robots[self.agent_idx]["pos"])
        nearest_distance = _nearest_distance(robot_pos, tracking, env)
        max_distance = max(env.width * env.height - 1, 1)
        return {
            "available_count": available_count,
            "tracking_count": len(tracking),
            "progressed_count": progressed_count,
            "upstream_clear_ratio": float(upstream_clear_ratio),
            "nearest_distance": nearest_distance,
            "nearest_distance_norm": (
                1.0 if nearest_distance is None else min(nearest_distance / max_distance, 1.0)
            ),
            "on_target": robot_pos in active_cells,
            "wait_flag": not active_cells and not stage_done,
            "guidance_active": guidance_active,
            "stage_done": stage_done,
            "target_node_index": float(target_node_index),
        }

    @staticmethod
    def _nearest_releasing_nodes(node_id: str, cell: tuple[int, int], env) -> set[str]:
        """Find currently executable ancestors that can release one blocked assignment."""

        completed = env.completed_by_cell[cell]
        queue = deque([node_id])
        visited: set[str] = set()
        releasing: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in visited or current in completed:
                continue
            visited.add(current)
            if cell in env.frontiers[current]:
                releasing.add(current)
                continue
            queue.extend(
                predecessor
                for predecessor in env.predecessors[current]
                if predecessor not in completed
            )
        return releasing


class WorkflowPerAgentRewardMachineSuite:
    """Per-agent role-conditioned RM suite used by SACC."""

    def __init__(self, config: dict, env, **kwargs):
        self.config = WorkflowRMConfig(**kwargs)
        self.agent_rms = [
            WorkflowLocalRewardMachine(index, str(robot["role"]), env, self.config)
            for index, robot in enumerate(env.robots)
        ]
        self.obs_dim = WorkflowLocalRewardMachine.obs_dim

    def reset(self, env) -> None:
        for rm in self.agent_rms:
            rm.reset(env)

    def step(self, events, env):
        rewards, observations, debug = [], [], []
        for rm, event in zip(self.agent_rms, events):
            reward, metrics = rm.step(event, env)
            rewards.append(reward)
            observations.append(rm.observe(env))
            debug.append(metrics)
        if env.is_all_done():
            bonus = self.config.reward_all_done / max(len(rewards), 1)
            rewards = [reward + bonus for reward in rewards]
        return rewards, observations, debug

    def get_obs_list(self, env):
        return [rm.observe(env) for rm in self.agent_rms]

    def get_global_progress(self, env):
        return np.concatenate(self.get_obs_list(env), axis=0).astype(np.float32)

    @staticmethod
    def _count_total_cells(config: dict) -> int:
        return len(config["task_cells"])


class WorkflowTeamRewardMachineSuite:
    """Shared team-RM baseline with global node progress and transition reward."""

    def __init__(self, config: dict, env, **kwargs):
        self.config = WorkflowRMConfig(**kwargs)
        self.node_ids = tuple(env.topological_order)
        self.num_agents = env.num_agents
        self.obs_dim = 2 * len(self.node_ids) + 1
        self.prev_node_done = {node_id: False for node_id in self.node_ids}

    def reset(self, env) -> None:
        self.prev_node_done = {
            node_id: all(
                node_id in completed for completed in env.completed_by_cell.values()
            )
            for node_id in self.node_ids
        }

    def _observe(self, env) -> np.ndarray:
        cells = max(len(env.task_cells), 1)
        completed, active = [], []
        for node_id in self.node_ids:
            completed.append(
                sum(node_id in nodes for nodes in env.completed_by_cell.values()) / cells
            )
            active.append(len(env.frontiers[node_id]) / cells)
        return np.array([*completed, *active, env.get_completion_ratio()], dtype=np.float32)

    def step(self, events, env):
        transition_events = sum(
            event is not None and event not in {"moved", "stayed", "blocked", "no effect"}
            for event in events
        )
        shared_total = self.config.reward_stage_event * transition_events
        newly_done = 0
        for node_id in self.node_ids:
            node_done = all(
                node_id in completed for completed in env.completed_by_cell.values()
            )
            newly_done += int(node_done and not self.prev_node_done[node_id])
            self.prev_node_done[node_id] = node_done
        shared_total += self.config.reward_stage_done * newly_done
        if env.is_all_done():
            shared_total += self.config.reward_all_done
        per_agent = shared_total / max(self.num_agents, 1)
        rewards = [float(per_agent)] * self.num_agents
        obs = self._observe(env)
        debug = [
            {
                "team_transition_events": transition_events,
                "team_newly_done_nodes": newly_done,
                "stage_done": env.is_all_done(),
            }
            for _ in range(self.num_agents)
        ]
        return rewards, [obs.copy() for _ in range(self.num_agents)], debug

    def get_obs_list(self, env):
        obs = self._observe(env)
        return [obs.copy() for _ in range(self.num_agents)]

    def get_global_progress(self, env):
        return self._observe(env)

    @staticmethod
    def _count_total_cells(config: dict) -> int:
        return len(config["task_cells"])
