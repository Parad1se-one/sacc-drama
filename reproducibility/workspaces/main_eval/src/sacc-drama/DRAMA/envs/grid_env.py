# envs/grid_area_env.py
"""
Grid Area Environment — Floor-Only Construction (No Walls)

任务阶段 (仅地面):
    Stage 1: distribute_floor  (布料)
    Stage 2: vibrate_floor     (振捣)
    Stage 3: level_floor       (整平)
    Stage 4: cover_surface     (覆膜)

作者: DRAMA Runner 🎭
日期: 2026-03-11 (移除墙体作业)
"""

import numpy as np
from gymnasium import spaces
from configs.env_config import env_config
import matplotlib.pyplot as plt


class GridAreaEnv:
    NUM_ACTIONS = 6  # UP, DOWN, LEFT, RIGHT, PERFORM, STAY

    def __init__(self, config=env_config):
        self._config = config
        self.width = config["grid_width"]
        self.height = config["grid_height"]

        self.robots_config = config["robots"]
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
        self._update_passability()

        self.distribute_map = np.zeros((self.width, self.height), dtype=int)
        self.vibrate_map = np.zeros((self.width, self.height), dtype=int)
        self.level_map = np.zeros((self.width, self.height), dtype=int)
        self.cover_map = np.zeros((self.width, self.height), dtype=int)
        self._update_task_map()

        self.task_dependencies = config["task_dependencies"]

        map_cells = self.width * self.height
        other_pos_len = 2 * (self.num_agents - 1)
        obs_dim = 2 + other_pos_len + 5 * map_cells  # own_pos + others + 5 maps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

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
        for r in self.robots:
            x, y = r["pos"]
            self.passable_map[x, y] = False

    def _update_task_map(self, infos=None):
        """更新任务地图"""
        # distribute_map (仅地面)
        self.distribute_map[:, :] = 0
        for (x, y) in self.floors_to_distribute:
            self.distribute_map[x, y] = 1

        # vibrate_map (仅地面)
        self.vibrate_map[:, :] = 0
        for (x, y) in self.floors_to_vibrate:
            self.vibrate_map[x, y] = 1

        # level_map
        self.level_map[:, :] = 0
        for (x, y) in self.floors_to_level:
            self.level_map[x, y] = 1

        # cover_map
        self.cover_map[:, :] = 0
        for (x, y) in self.floors_to_cover:
            self.cover_map[x, y] = 1

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _can_execute_task(self, robot):
        """检查机器人当前是否可以合法执行施工动作。"""
        if robot["type"] == "material_distributer":
            return robot["pos"] in self.floors_to_distribute
        if robot["type"] == "vibrator":
            return robot["pos"] in self.floors_to_vibrate
        if robot["type"] == "leveler":
            return robot["pos"] in self.floors_to_level
        if robot["type"] == "film_cover":
            return robot["pos"] in self.floors_to_cover
        return False

    def get_action_mask(self, agent_idx):
        """
        返回指定 agent 的 6 维布尔动作掩码。

        约定:
            0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 PERFORM, 5 STAY
        """
        robot = self.robots[agent_idx]
        x, y = robot["pos"]
        candidates = (
            (x, y + 1),  # UP
            (x, y - 1),  # DOWN
            (x - 1, y),  # LEFT
            (x + 1, y),  # RIGHT
        )

        mask = np.zeros(self.NUM_ACTIONS, dtype=bool)
        for action, (nx, ny) in enumerate(candidates):
            if self._in_bounds(nx, ny) and self.passable_map[nx, ny]:
                mask[action] = True

        mask[4] = self._can_execute_task(robot)
        mask[5] = True
        return mask

    def get_action_masks(self):
        return np.stack([self.get_action_mask(i) for i in range(self.num_agents)], axis=0)

    def step(self, actions):
        rewards, infos = [], []

        for i, action in enumerate(actions):
            robot = self.robots[i]
            reward, info = self._move_robot(robot, action)
            rewards.append(reward)
            infos.append(info)
            self._update_passability()

        self._update_task_map()
        return rewards, infos

    def _move_robot(self, robot, action):
        """处理机器人动作（上下左右移动 + 执行任务）"""
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
            return 0, {"status": "stayed"}

        # 检查是否可通行
        if (0 <= new_x < self.width and 0 <= new_y < self.height
                and self.passable_map[new_x, new_y]):
            robot["pos"] = (new_x, new_y)
            rw = 0.0  # 移动不惩罚（鼓励探索）
            return rw, {"status": "moved"}
        else:
            return -0.1, {"status": "blocked"}

    def _execute_task(self, robot):
        """执行施工任务 (仅地面，无墙体)"""
        if robot["type"] == "material_distributer":
            if robot["pos"] in self.floors_to_distribute:
                self.floors_to_distribute.remove(robot["pos"])
                self.floors_to_vibrate.add(robot["pos"])
                return 1, {"status": "distribute_floor"}
            else:
                return -1, {"status": "no effect"}

        elif robot["type"] == "vibrator":
            if robot["pos"] in self.floors_to_vibrate:
                self.floors_to_vibrate.remove(robot["pos"])
                self.floors_to_level.add(robot["pos"])
                return 1, {"status": "vibrate_floor"}
            else:
                return -1, {"status": "no effect"}

        elif robot["type"] == "leveler":
            if robot["pos"] in self.floors_to_level:
                self.floors_to_level.remove(robot["pos"])
                self.floors_to_cover.add(robot["pos"])
                return 1, {"status": "level_floor"}
            else:
                return -1, {"status": "no effect"}

        elif robot["type"] == "film_cover":
            if robot["pos"] in self.floors_to_cover:
                self.floors_to_cover.remove(robot["pos"])
                self.floors_completed.add(robot["pos"])
                return 1, {"status": "cover_surface"}
            else:
                return -1, {"status": "no effect"}

        return -1, {"status": "no effect"}

    def reset(self, config=None, *, seed=None):
        if config is None:
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
        self._update_passability()

        self.distribute_map = np.zeros((self.width, self.height), dtype=int)
        self.vibrate_map = np.zeros((self.width, self.height), dtype=int)
        self.level_map = np.zeros((self.width, self.height), dtype=int)
        self.cover_map = np.zeros((self.width, self.height), dtype=int)
        self._update_task_map()

        self.task_dependencies = config["task_dependencies"]
        return self._get_state()

    def get_completion_ratio(self):
        """计算真实任务完成度 (0.0 ~ 1.0)

        任务流水线: distribute → vibrate → level → cover
        每个 grid 需要经过 4 个阶段才算完成。
        完成度 = 已完成的任务实例数 / 总任务实例数
        """
        # 总 grid 数 = 初始待布料的 floor 数
        total_floors = (len(self.floors_to_distribute) + len(self.floors_to_vibrate) +
                       len(self.floors_to_level) + len(self.floors_to_cover) +
                       len(self.floors_completed))

        # 每个 floor grid 有 4 个阶段
        total_tasks = total_floors * 4
        if total_tasks == 0:
            return 1.0

        # 已完成的任务实例数
        # floor: distribute(1) → vibrate(2) → level(3) → cover/completed(4)
        completed_tasks = 0
        # floors 已完成全部 4 阶段
        completed_tasks += len(self.floors_completed) * 4
        # floors 在 cover 阶段 = 完成了 3 阶段
        completed_tasks += len(self.floors_to_cover) * 3
        # floors 在 level 阶段 = 完成了 2 阶段
        completed_tasks += len(self.floors_to_level) * 2
        # floors 在 vibrate 阶段 = 完成了 1 阶段
        completed_tasks += len(self.floors_to_vibrate) * 1

        return completed_tasks / total_tasks

    def is_all_done(self):
        """检查是否所有任务都已完成"""
        return (len(self.floors_to_distribute) == 0 and
                len(self.floors_to_vibrate) == 0 and
                len(self.floors_to_level) == 0 and
                len(self.floors_to_cover) == 0)

    def _get_state(self):
        """返回环境状态"""
        state = {
            "robots": [(r["type"], r["pos"]) for r in self.robots],
            "floors_completed": list(self.floors_completed),
            "floors_to_distribute": list(self.floors_to_distribute),
            "floors_to_vibrate": list(self.floors_to_vibrate),
            "floors_to_level": list(self.floors_to_level),
            "floors_to_cover": list(self.floors_to_cover),
            "obstacles": list(self.obstacles | self.dynamic_obstacles)
        }
        return state

    def render(self):
        """简单可视化"""
        grid = np.zeros((self.width, self.height), dtype=int)
        for (x, y) in self.obstacles:
            grid[x, y] = -1
        for i, r in enumerate(self.robots):
            x, y = r["pos"]
            grid[x, y] = i + 1

        plt.imshow(grid.T, origin="lower", cmap="tab20")
        plt.colorbar()
        plt.title("Grid Environment State")
        plt.show()
