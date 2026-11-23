"""Generic DAG workflow environment used by the Reviewer 3 extensions."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from gymnasium import spaces

from .starts import fixed_start_positions, validate_start_positions


def topological_order(node_ids: list[str], edges: list[tuple[str, str]]) -> tuple[str, ...]:
    """Validate a workflow DAG and return a deterministic topological order."""

    known = set(node_ids)
    if len(known) != len(node_ids):
        raise ValueError("Workflow node ids must be unique")
    successors: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for source, target in edges:
        if source not in known or target not in known:
            raise ValueError(f"Workflow edge {(source, target)} references an unknown node")
        successors[source].append(target)
        indegree[target] += 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for target in successors[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(ordered) != len(node_ids):
        raise ValueError("Workflow graph must be acyclic")
    return tuple(ordered)


class WorkflowBenchmarkEnv:
    """Grid-world construction environment whose per-cell task is a configurable DAG."""

    NUM_ACTIONS = 6

    def __init__(self, config: dict):
        self._config = config
        self.width = int(config["grid_width"])
        self.height = int(config["grid_height"])
        self.robots_config = config["robots"]
        workflow = config["workflow"]
        self.nodes = tuple(dict(node) for node in workflow["nodes"])
        self.node_ids = tuple(str(node["id"]) for node in self.nodes)
        self.node_by_id = {str(node["id"]): node for node in self.nodes}
        self.edges = tuple((str(a), str(b)) for a, b in workflow["edges"])
        self.topological_order = topological_order(list(self.node_ids), list(self.edges))
        self.node_index = {node_id: index for index, node_id in enumerate(self.topological_order)}
        self.predecessors: dict[str, set[str]] = {node_id: set() for node_id in self.node_ids}
        self.successors: dict[str, set[str]] = {node_id: set() for node_id in self.node_ids}
        for source, target in self.edges:
            self.predecessors[target].add(source)
            self.successors[source].add(target)
        self.role_to_nodes: dict[str, list[str]] = defaultdict(list)
        for node_id in self.topological_order:
            self.role_to_nodes[str(self.node_by_id[node_id]["role"])].append(node_id)

        self.task_cells = {tuple(map(int, cell)) for cell in config["task_cells"]}
        self.total_stage_tasks = max(len(self.task_cells) * len(self.node_ids), 1)
        self.robots = self._init_robots()
        self.num_agents = len(self.robots)
        self.obstacles = {tuple(map(int, cell)) for cell in config["obstacles"]["static"]}
        self.dynamic_obstacles = {tuple(map(int, cell)) for cell in config["obstacles"]["dynamic"]}
        self.completed_by_cell: dict[tuple[int, int], set[str]] = {
            cell: set() for cell in self.task_cells
        }
        self.frontiers: dict[str, set[tuple[int, int]]] = {
            node_id: set() for node_id in self.node_ids
        }
        self.passable_map = np.ones((self.width, self.height), dtype=bool)
        self.frontier_maps: dict[str, np.ndarray] = {
            node_id: np.zeros((self.width, self.height), dtype=int)
            for node_id in self.node_ids
        }
        self._recompute_frontiers()
        self._update_passability()
        self._update_task_maps()

        map_cells = self.width * self.height
        other_pos_len = 2 * (self.num_agents - 1)
        obs_dim = 2 + other_pos_len + (1 + len(self.node_ids)) * map_cells
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

    def _init_robots(self, start_positions=None) -> list[dict]:
        if start_positions is None:
            normalized = fixed_start_positions(self._config)
        else:
            normalized = validate_start_positions(
                self._config,
                start_positions,
                exclude_task_cells=False,
                min_pairwise_manhattan=1,
            )
        robots: list[dict] = []
        position_index = 0
        for role, info in self.robots_config.items():
            for _ in info["init_positions"]:
                robots.append(
                    {
                        "type": role,
                        "role": role,
                        "pos": normalized[position_index],
                        "tasks": list(self.role_to_nodes.get(role, ())),
                    }
                )
                position_index += 1
        self.start_positions = tuple(normalized)
        return robots

    def reset(self, config: dict | None = None, *, seed: int | None = None, start_positions=None):
        if config is not None and config != self._config:
            raise ValueError("WorkflowBenchmarkEnv does not support changing workflow schema on reset")
        self.seed = seed
        self.robots = self._init_robots(start_positions=start_positions)
        self.num_agents = len(self.robots)
        self.completed_by_cell = {cell: set() for cell in self.task_cells}
        self.frontiers = {node_id: set() for node_id in self.node_ids}
        self.passable_map = np.ones((self.width, self.height), dtype=bool)
        self.frontier_maps = {
            node_id: np.zeros((self.width, self.height), dtype=int)
            for node_id in self.node_ids
        }
        self._recompute_frontiers()
        self._update_passability()
        self._update_task_maps()
        return self._get_state()

    def _recompute_frontiers(self) -> None:
        for node_id in self.node_ids:
            self.frontiers[node_id].clear()
        for cell, completed in self.completed_by_cell.items():
            for node_id in self.topological_order:
                if node_id not in completed and self.predecessors[node_id].issubset(completed):
                    self.frontiers[node_id].add(cell)

    def _update_passability(self) -> None:
        self.passable_map[:, :] = True
        for x, y in self.obstacles | self.dynamic_obstacles:
            self.passable_map[x, y] = False
        for robot in self.robots:
            x, y = robot["pos"]
            self.passable_map[x, y] = False

    def _update_task_maps(self) -> None:
        for node_id, task_map in self.frontier_maps.items():
            task_map[:, :] = 0
            for x, y in self.frontiers[node_id]:
                task_map[x, y] = 1

    def get_task_maps(self) -> list[np.ndarray]:
        return [self.passable_map, *[self.frontier_maps[node] for node in self.topological_order]]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def executable_nodes(self, robot: dict) -> list[str]:
        role = str(robot["role"])
        pos = tuple(robot["pos"])
        return [
            node_id
            for node_id in self.role_to_nodes.get(role, ())
            if pos in self.frontiers[node_id]
        ]

    def get_action_mask(self, agent_idx: int) -> np.ndarray:
        robot = self.robots[agent_idx]
        x, y = robot["pos"]
        candidates = ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y))
        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)
        for action, (nx, ny) in enumerate(candidates):
            if self._in_bounds(nx, ny) and self.passable_map[nx, ny]:
                mask[action] = True
        mask[4] = bool(self.executable_nodes(robot))
        mask[5] = True
        return mask

    def get_action_masks(self) -> np.ndarray:
        return np.stack([self.get_action_mask(i) for i in range(self.num_agents)], axis=0)

    def _execute_task(self, robot: dict):
        candidates = self.executable_nodes(robot)
        if not candidates:
            return -1.0, {"status": "no effect"}
        node_id = min(candidates, key=self.node_index.__getitem__)
        cell = tuple(robot["pos"])
        self.completed_by_cell[cell].add(node_id)
        self._recompute_frontiers()
        return 1.0, {
            "status": str(self.node_by_id[node_id].get("event", f"complete:{node_id}")),
            "workflow_node": node_id,
        }

    def _move_robot(self, robot: dict, action: int):
        x, y = robot["pos"]
        new_x, new_y = x, y
        if action == 0:
            new_y += 1
        elif action == 1:
            new_y -= 1
        elif action == 2:
            new_x -= 1
        elif action == 3:
            new_x += 1
        elif action == 4:
            return self._execute_task(robot)
        elif action == 5:
            return 0.0, {"status": "stayed"}
        if self._in_bounds(new_x, new_y) and self.passable_map[new_x, new_y]:
            robot["pos"] = (new_x, new_y)
            return 0.0, {"status": "moved"}
        return -0.1, {"status": "blocked"}

    def step(self, actions):
        rewards = []
        infos = []
        for index, action in enumerate(actions):
            reward, info = self._move_robot(self.robots[index], int(action))
            rewards.append(reward)
            infos.append(info)
            self._update_passability()
        self._update_task_maps()
        return rewards, infos

    def get_completion_ratio(self) -> float:
        completed = sum(len(nodes) for nodes in self.completed_by_cell.values())
        return completed / self.total_stage_tasks

    def is_all_done(self) -> bool:
        return all(len(nodes) == len(self.node_ids) for nodes in self.completed_by_cell.values())

    def get_terminal_stage_counts(self) -> dict[str, int]:
        counts = {f"active_{node_id}": len(self.frontiers[node_id]) for node_id in self.topological_order}
        counts["completed_operations"] = sum(len(nodes) for nodes in self.completed_by_cell.values())
        counts["completed_cells"] = sum(
            len(nodes) == len(self.node_ids) for nodes in self.completed_by_cell.values()
        )
        return counts

    def _get_state(self) -> dict:
        return {
            "robots": [(robot["type"], robot["pos"]) for robot in self.robots],
            "workflow_frontiers": {
                node_id: list(self.frontiers[node_id]) for node_id in self.topological_order
            },
            "workflow_completed_by_cell": {
                str(cell): sorted(nodes, key=self.node_index.__getitem__)
                for cell, nodes in self.completed_by_cell.items()
            },
            "obstacles": list(self.obstacles | self.dynamic_obstacles),
        }
