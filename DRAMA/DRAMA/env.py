from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .specs import get_spec_data


class DramaBenchmarkEnv:
    NUM_ACTIONS = 6

    def __init__(self, config: dict | None = None):
        if config is None:
            config = get_spec_data("easy")["config"]
        self._config = config
        self.width = config["grid_width"]
        self.height = config["grid_height"]
        self.robots_config = config["robots"]
        self.task_dependencies = config["task_dependencies"]
        self._initial_task_cells = self._count_total_task_cells(config)

        self.robots = self._init_robots()
        self.num_agents = len(self.robots)
        self.obstacles = set(config["obstacles"]["static"])
        self.dynamic_obstacles = set(config["obstacles"]["dynamic"])

        self.floors_to_distribute = set(config["floors"]["to_distribute"])
        self.floors_to_vibrate = set(config["floors"]["to_vibrate"])
        self.floors_to_level = set(config["floors"]["to_level"])
        self.floors_to_cover = set(config["floors"]["to_cover"])
        self.floors_completed = set()

        self.passable_map = np.ones((self.width, self.height), dtype=bool)
        self.distribute_map = np.zeros((self.width, self.height), dtype=int)
        self.vibrate_map = np.zeros((self.width, self.height), dtype=int)
        self.level_map = np.zeros((self.width, self.height), dtype=int)
        self.cover_map = np.zeros((self.width, self.height), dtype=int)

        self._update_passability()
        self._update_task_map()

        map_cells = self.width * self.height
        other_pos_len = 2 * (self.num_agents - 1)
        obs_dim = 2 + other_pos_len + 5 * map_cells
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

    @staticmethod
    def _count_total_task_cells(config: dict) -> int:
        floors = config["floors"]
        return sum(len(floors[key]) for key in ("to_distribute", "to_vibrate", "to_level", "to_cover"))

    def _init_robots(self):
        robots = []
        for robot_type, info in self.robots_config.items():
            for pos in info["init_positions"]:
                robots.append({
                    "type": robot_type,
                    "pos": pos,
                    "tasks": info["tasks"],
                })
        return robots

    def _update_passability(self):
        self.passable_map[:, :] = True
        for (x, y) in self.obstacles | self.dynamic_obstacles:
            self.passable_map[x, y] = False
        for robot in self.robots:
            x, y = robot["pos"]
            self.passable_map[x, y] = False

    def _update_task_map(self):
        self.distribute_map[:, :] = 0
        for (x, y) in self.floors_to_distribute:
            self.distribute_map[x, y] = 1

        self.vibrate_map[:, :] = 0
        for (x, y) in self.floors_to_vibrate:
            self.vibrate_map[x, y] = 1

        self.level_map[:, :] = 0
        for (x, y) in self.floors_to_level:
            self.level_map[x, y] = 1

        self.cover_map[:, :] = 0
        for (x, y) in self.floors_to_cover:
            self.cover_map[x, y] = 1

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _can_execute_task(self, robot: dict) -> bool:
        if robot["type"] == "material_distributer":
            return robot["pos"] in self.floors_to_distribute
        if robot["type"] == "vibrator":
            return robot["pos"] in self.floors_to_vibrate
        if robot["type"] == "leveler":
            return robot["pos"] in self.floors_to_level
        if robot["type"] == "film_cover":
            return robot["pos"] in self.floors_to_cover
        return False

    def get_action_mask(self, agent_idx: int) -> np.ndarray:
        robot = self.robots[agent_idx]
        x, y = robot["pos"]
        candidates = (
            (x, y + 1),
            (x, y - 1),
            (x - 1, y),
            (x + 1, y),
        )

        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)
        for action, (nx, ny) in enumerate(candidates):
            if self._in_bounds(nx, ny) and self.passable_map[nx, ny]:
                mask[action] = True

        mask[4] = self._can_execute_task(robot)
        mask[5] = True
        return mask

    def get_action_masks(self) -> np.ndarray:
        return np.stack([self.get_action_mask(i) for i in range(self.num_agents)], axis=0)

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

    def _execute_task(self, robot: dict):
        if robot["type"] == "material_distributer":
            if robot["pos"] in self.floors_to_distribute:
                self.floors_to_distribute.remove(robot["pos"])
                self.floors_to_vibrate.add(robot["pos"])
                return 1.0, {"status": "distribute_floor"}
            return -1.0, {"status": "no effect"}

        if robot["type"] == "vibrator":
            if robot["pos"] in self.floors_to_vibrate:
                self.floors_to_vibrate.remove(robot["pos"])
                self.floors_to_level.add(robot["pos"])
                return 1.0, {"status": "vibrate_floor"}
            return -1.0, {"status": "no effect"}

        if robot["type"] == "leveler":
            if robot["pos"] in self.floors_to_level:
                self.floors_to_level.remove(robot["pos"])
                self.floors_to_cover.add(robot["pos"])
                return 1.0, {"status": "level_floor"}
            return -1.0, {"status": "no effect"}

        if robot["type"] == "film_cover":
            if robot["pos"] in self.floors_to_cover:
                self.floors_to_cover.remove(robot["pos"])
                self.floors_completed.add(robot["pos"])
                return 1.0, {"status": "cover_surface"}
            return -1.0, {"status": "no effect"}

        return -1.0, {"status": "no effect"}

    def step(self, actions):
        rewards = []
        infos = []
        for index, action in enumerate(actions):
            reward, info = self._move_robot(self.robots[index], int(action))
            rewards.append(reward)
            infos.append(info)
            self._update_passability()
        self._update_task_map()
        return rewards, infos

    def reset(self, config: dict | None = None, *, seed: int | None = None):
        if config is not None:
            self._config = config
            self.width = config["grid_width"]
            self.height = config["grid_height"]
            self.robots_config = config["robots"]
            self.task_dependencies = config["task_dependencies"]
            self._initial_task_cells = self._count_total_task_cells(config)
        else:
            config = self._config

        if seed is not None:
            np.random.seed(seed)

        self.robots = self._init_robots()
        self.num_agents = len(self.robots)
        self.obstacles = set(config["obstacles"]["static"])
        self.dynamic_obstacles = set(config["obstacles"]["dynamic"])
        self.floors_to_distribute = set(config["floors"]["to_distribute"])
        self.floors_to_vibrate = set(config["floors"]["to_vibrate"])
        self.floors_to_level = set(config["floors"]["to_level"])
        self.floors_to_cover = set(config["floors"]["to_cover"])
        self.floors_completed = set()
        self.passable_map = np.ones((self.width, self.height), dtype=bool)
        self.distribute_map = np.zeros((self.width, self.height), dtype=int)
        self.vibrate_map = np.zeros((self.width, self.height), dtype=int)
        self.level_map = np.zeros((self.width, self.height), dtype=int)
        self.cover_map = np.zeros((self.width, self.height), dtype=int)
        self._update_passability()
        self._update_task_map()
        return self._get_state()

    def get_completion_ratio(self) -> float:
        if self._initial_task_cells == 0:
            return 1.0
        completed_stages = 0
        completed_stages += len(self.floors_completed) * 4
        completed_stages += len(self.floors_to_cover) * 3
        completed_stages += len(self.floors_to_level) * 2
        completed_stages += len(self.floors_to_vibrate) * 1
        return completed_stages / (self._initial_task_cells * 4)

    def is_all_done(self) -> bool:
        return (
            len(self.floors_to_distribute) == 0
            and len(self.floors_to_vibrate) == 0
            and len(self.floors_to_level) == 0
            and len(self.floors_to_cover) == 0
        )

    def _get_state(self):
        return {
            "robots": [(robot["type"], robot["pos"]) for robot in self.robots],
            "floors_completed": list(self.floors_completed),
            "floors_to_distribute": list(self.floors_to_distribute),
            "floors_to_vibrate": list(self.floors_to_vibrate),
            "floors_to_level": list(self.floors_to_level),
            "floors_to_cover": list(self.floors_to_cover),
            "obstacles": list(self.obstacles | self.dynamic_obstacles),
        }

    def render(self, mode: str = "text"):
        if mode == "text":
            grid = [["." for _ in range(self.width)] for _ in range(self.height)]
            for x, y in self.obstacles:
                grid[y][x] = "#"
            for x, y in self.floors_to_distribute:
                grid[y][x] = "D"
            for x, y in self.floors_to_vibrate:
                grid[y][x] = "V"
            for x, y in self.floors_to_level:
                grid[y][x] = "L"
            for x, y in self.floors_to_cover:
                grid[y][x] = "C"
            for idx, robot in enumerate(self.robots):
                x, y = robot["pos"]
                grid[y][x] = str(idx)
            return "\n".join(" ".join(row) for row in reversed(grid))

        if mode == "matplotlib":
            import matplotlib.pyplot as plt

            grid = np.zeros((self.width, self.height), dtype=int)
            for x, y in self.obstacles:
                grid[x, y] = -1
            for index, robot in enumerate(self.robots):
                x, y = robot["pos"]
                grid[x, y] = index + 1
            plt.imshow(grid.T, origin="lower", cmap="tab20")
            plt.colorbar()
            plt.title("DRAMA Benchmark State")
            plt.show()
            return None

        raise ValueError(f"Unsupported render mode: {mode}")


GridAreaEnv = DramaBenchmarkEnv
