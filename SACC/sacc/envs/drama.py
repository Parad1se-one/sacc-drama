"""DRAMA benchmark wrapper with per-agent reward-machine augmentation."""

from __future__ import annotations

import numpy as np

from sacc.bootstrap import ensure_drama_on_path
from sacc.reward_machines.per_agent_rm import PerAgentRewardMachineSuite, ROLE_TO_STAGE

ensure_drama_on_path()

from DRAMA import DramaBenchmarkEnv, get_config, get_metadata  # noqa: E402

TASK_MAP_ATTRS = (
    "passable_map",
    "distribute_map",
    "vibrate_map",
    "level_map",
    "cover_map",
)


class DramaRMEnv:
    def __init__(
        self,
        difficulty: str = "easy",
        *,
        seed: int | None = None,
        env_reward_scale: float = 1.0,
        rm_reward_scale: float = 1.0,
        team_progress_bonus_coeff: float = 0.25,
        rm_kwargs: dict | None = None,
    ):
        self.difficulty = difficulty.lower()
        self.config = get_config(self.difficulty)
        self.metadata = get_metadata(self.difficulty)
        self.env = DramaBenchmarkEnv(config=self.config)
        self.rm = PerAgentRewardMachineSuite(self.config, **(rm_kwargs or {}))
        self.env_reward_scale = float(env_reward_scale)
        self.rm_reward_scale = float(rm_reward_scale)
        self.team_progress_bonus_coeff = float(team_progress_bonus_coeff)
        self.num_agents = self.config["num_robots"]
        self.action_n = self.env.NUM_ACTIONS
        self.total_stage_tasks = max(self.rm._count_total_cells(self.config) * 4, 1)
        self.seed = seed
        self.episode_steps = 0
        self.prev_completion_ratio = 0.0
        self.reset(seed=seed)
        self.base_obs_dim = len(self._build_base_obs(0))
        self.rm_obs_dim = self.rm.obs_dim
        self.obs_dim = len(self.build_obs_list()[0])
        self.state_dim = len(self.build_state())

    def reset(self, *, seed: int | None = None):
        self.seed = self.seed if seed is None else seed
        self.env.reset(seed=self.seed)
        self.rm.reset(self.env)
        self.episode_steps = 0
        self.prev_completion_ratio = self.get_completion_ratio()
        return self.env._get_state()

    def build_obs_list(self, *, include_rm: bool = True) -> list[np.ndarray]:
        rm_obs_list = self.rm.get_obs_list(self.env)
        if not include_rm:
            rm_obs_list = [np.zeros_like(rm_obs) for rm_obs in rm_obs_list]
        return [
            np.concatenate([self._build_base_obs(agent_idx), rm_obs_list[agent_idx]], axis=0).astype(np.float32)
            for agent_idx in range(self.num_agents)
        ]

    def build_state(self, *, include_rm: bool = True) -> np.ndarray:
        positions = [robot["pos"] for robot in self.env.robots]
        flat_pos = np.array([coord for pos in positions for coord in pos], dtype=np.float32)
        task_maps = [getattr(self.env, attr).astype(np.float32).flatten() for attr in TASK_MAP_ATTRS]
        rm_progress = self.rm.get_global_progress(self.env)
        if not include_rm:
            rm_progress = np.zeros_like(rm_progress)
        completion = np.array([self.get_completion_ratio()], dtype=np.float32)
        return np.concatenate([flat_pos, *task_maps, rm_progress, completion], axis=0).astype(np.float32)

    def get_action_masks(self) -> np.ndarray:
        return self.env.get_action_masks().astype(bool)

    def step(self, actions: list[int]) -> dict:
        completion_before = self.get_completion_ratio()
        env_rewards, infos = self.env.step(actions)
        events = [info.get("status") for info in infos]
        rm_rewards, rm_obs, rm_debug = self.rm.step(events, self.env)
        completion_after = self.get_completion_ratio()
        progress_delta = max((completion_after - completion_before) * self.total_stage_tasks, 0.0)
        progress_bonus = self.team_progress_bonus_coeff * progress_delta
        total_rewards = [
            self.env_reward_scale * float(env_reward)
            + self.rm_reward_scale * float(rm_reward)
            + progress_bonus
            for env_reward, rm_reward in zip(env_rewards, rm_rewards)
        ]
        self.episode_steps += 1
        self.prev_completion_ratio = completion_after
        return {
            "rewards": total_rewards,
            "env_rewards": [float(reward) for reward in env_rewards],
            "rm_rewards": [float(reward) for reward in rm_rewards],
            "progress_bonus": float(progress_bonus),
            "infos": infos,
            "events": events,
            "rm_obs": rm_obs,
            "rm_debug": rm_debug,
            "done": self.is_all_done(),
            "completion_ratio": completion_after,
        }

    def get_completion_ratio(self) -> float:
        return float(self.env.get_completion_ratio())

    def is_all_done(self) -> bool:
        return bool(self.env.is_all_done())

    def _build_base_obs(self, agent_idx: int) -> np.ndarray:
        positions = [robot["pos"] for robot in self.env.robots]
        own_pos = positions[agent_idx]
        others = [coord for idx, pos in enumerate(positions) if idx != agent_idx for coord in pos]
        task_maps = [getattr(self.env, attr).astype(np.float32).flatten() for attr in TASK_MAP_ATTRS]
        return np.concatenate(
            [
                np.array([own_pos[0], own_pos[1]], dtype=np.float32),
                np.array(others, dtype=np.float32),
                *task_maps,
            ],
            axis=0,
        ).astype(np.float32)
