"""
Hierarchical FSM Reward Machine for DRAMA

Architecture:
    Per-Grid FSM  →  Per-Region FSM  →  Global FSM

Per-Grid Stage (0-4):
    0: PENDING      — in floors_to_distribute
    1: DISTRIBUTED  — in floors_to_vibrate
    2: VIBRATED     — in floors_to_level
    3: LEVELED      — in floors_to_cover
    4: COMPLETED    — in floors_completed

Region Phase = min(stage of all grids in that region)
    Phase advances when ALL grids in the region reach the next stage.

Global Phase = min(phase of all regions)
    Global completion when all regions reach phase 4.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


NUM_STAGES = 4

STAGE_NAMES = {
    0: "PENDING",
    1: "DISTRIBUTED",
    2: "VIBRATED",
    3: "LEVELED",
    4: "COMPLETED",
}

EVENT_TO_STAGE_ADVANCE = {
    "distribute_floor": (0, 1),
    "vibrate_floor":    (1, 2),
    "level_floor":      (2, 3),
    "cover_surface":    (3, 4),
}

REWARD_MACHINE_STATES = {
    "nodes": [
        {"id": f"S{i}", "label": STAGE_NAMES[i]}
        for i in range(NUM_STAGES + 1)
    ],
    "transitions": [
        {"from": f"S{i}", "to": f"S{i+1}", "condition": name}
        for name, (i, _) in EVENT_TO_STAGE_ADVANCE.items()
    ],
}


class EnhancedRewardMachine:
    """
    Hierarchical FSM Reward Machine.

    Tracks per-grid progress through the construction pipeline,
    aggregates into region-level and global-level FSM states,
    and produces structured reward signals at each level.
    """

    def __init__(
        self,
        task_dependencies: Dict[str, list],
        num_agents: int,
        total_grids: Optional[int] = None,
        regions: Optional[Dict[str, List[tuple]]] = None,
        reward_grid_advance: float = 10.0,
        reward_region_phase: float = 50.0,
        reward_region_done: float = 100.0,
        reward_all_done: float = 500.0,
        penalty_idle: float = -0.05,
        penalty_invalid: float = -0.5,
        penalty_blocked: float = -1.0,
        penalty_oscillation: float = 0.0,
        distance_shaping_coeff: float = 0.1,
    ):
        self.task_dependencies = task_dependencies
        self.num_agents = num_agents
        self.total_grids = total_grids
        self._regions_config = regions

        self.reward_grid_advance = reward_grid_advance
        self.reward_region_phase = reward_region_phase
        self.reward_region_done = reward_region_done
        self.reward_all_done = reward_all_done
        self.penalty_invalid = penalty_invalid
        self.penalty_idle = penalty_idle
        self.penalty_blocked = penalty_blocked
        self.penalty_oscillation = penalty_oscillation
        self.distance_shaping_coeff = distance_shaping_coeff

        self._grid_stages: Dict[tuple, int] = {}
        self._region_phases: Dict[str, int] = {}
        self._regions: Dict[str, List[tuple]] = {}
        self._grid_to_region: Dict[tuple, str] = {}
        self._all_done_triggered: bool = False
        self._prev_grid_stages: Dict[tuple, int] = {}
        self._prev_potentials: List[float] = []
        self._prev_positions: List[tuple] = []
        self._prev_prev_positions: List[tuple] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, env=None) -> None:
        """Reset all FSM states. Optionally seed grid list from env."""
        self._all_done_triggered = False
        self._grid_stages = {}
        self._region_phases = {}
        self._prev_grid_stages = {}
        self._prev_potentials = []
        self._prev_positions = []
        self._prev_prev_positions = []

        if env is not None:
            self._init_from_env(env)

    def step(
        self,
        events: List[Optional[str]],
        env=None,
    ) -> Tuple[List[float], Dict[str, Any], Dict[str, Any]]:
        """
        Process one timestep of events and return RM rewards.

        Args:
            events: per-agent status strings from env.step()
            env: environment reference for reading grid progress

        Returns:
            rm_rewards: list[float] of length num_agents
            rm_states: dict with current FSM state snapshot
            info: dict with reward breakdown
        """
        rm_rewards = [0.0] * self.num_agents

        if env is not None and not self._grid_stages:
            self._init_from_env(env)

        if env is not None:
            self._prev_grid_stages = dict(self._grid_stages)
            self._sync_grid_stages(env)
            current_positions = [tuple(robot["pos"]) for robot in env.robots]
        else:
            current_positions = []

        reward_breakdown = {
            "grid_advance": 0.0,
            "region_phase": 0.0,
            "region_done": 0.0,
            "global_done": 0.0,
            "penalty": 0.0,
            "oscillation_penalty": 0.0,
        }

        for i, event in enumerate(events):
            if event is None:
                continue

            if event in EVENT_TO_STAGE_ADVANCE:
                rm_rewards[i] += self.reward_grid_advance
                reward_breakdown["grid_advance"] += self.reward_grid_advance
            elif event == "no effect":
                rm_rewards[i] += self.penalty_invalid
                reward_breakdown["penalty"] += self.penalty_invalid
            elif event == "stayed":
                rm_rewards[i] += self.penalty_idle
                reward_breakdown["penalty"] += self.penalty_idle
            elif event == "blocked":
                rm_rewards[i] += self.penalty_blocked
                reward_breakdown["penalty"] += self.penalty_blocked
            elif (
                env is not None
                and event == "moved"
                and self.penalty_oscillation != 0.0
                and len(self._prev_prev_positions) == self.num_agents
                and current_positions[i] == self._prev_prev_positions[i]
            ):
                rm_rewards[i] += self.penalty_oscillation
                reward_breakdown["penalty"] += self.penalty_oscillation
                reward_breakdown["oscillation_penalty"] += self.penalty_oscillation

        if env is not None and self._regions:
            region_bonus, done_bonus, global_bonus = self._compute_transition_rewards()
            if region_bonus > 0 or done_bonus > 0 or global_bonus > 0:
                per_agent = (region_bonus + done_bonus + global_bonus) / self.num_agents
                for i in range(self.num_agents):
                    rm_rewards[i] += per_agent
                reward_breakdown["region_phase"] += region_bonus
                reward_breakdown["region_done"] += done_bonus
                reward_breakdown["global_done"] += global_bonus

        if env is not None and self.distance_shaping_coeff > 0:
            shaping = self._compute_distance_shaping(env)
            for i in range(self.num_agents):
                rm_rewards[i] += shaping[i]
            reward_breakdown["distance_shaping"] = sum(shaping)

        rm_states = {
            "grid_stages": dict(self._grid_stages),
            "region_phases": dict(self._region_phases),
            "all_done": self._all_done_triggered,
        }

        info = {
            "total_rm_reward": sum(rm_rewards),
            "reward_breakdown": reward_breakdown,
        }

        if env is not None:
            if self._prev_positions:
                self._prev_prev_positions = list(self._prev_positions)
            else:
                self._prev_prev_positions = list(current_positions)
            self._prev_positions = list(current_positions)

        return rm_rewards, rm_states, info

    def get_progress_obs(self, env=None) -> np.ndarray:
        """
        Return observation vector encoding RM state for agents.

        Layout: [per_grid_progress..., per_region_phase..., global_ratio]
        """
        if env is not None and not self._grid_stages:
            self._init_from_env(env)

        if not self._grid_stages:
            n = self.total_grids or 0
            return np.zeros(n, dtype=np.float32)

        if env is not None:
            self._sync_grid_stages(env)

        sorted_grids = sorted(self._grid_stages.keys())
        grid_progress = np.array(
            [self._grid_stages[g] / NUM_STAGES for g in sorted_grids],
            dtype=np.float32,
        )

        if self._regions:
            sorted_regions = sorted(self._regions.keys())
            region_progress = np.array(
                [self._region_phases.get(r, 0) / NUM_STAGES for r in sorted_regions],
                dtype=np.float32,
            )
            completed = sum(1 for s in self._grid_stages.values() if s >= NUM_STAGES)
            global_ratio = np.array(
                [completed / max(len(self._grid_stages), 1)],
                dtype=np.float32,
            )
            return np.concatenate([grid_progress, region_progress, global_ratio])

        return grid_progress

    def trigger_all_done_reward(self) -> float:
        """Backward-compatible: manually trigger global done bonus."""
        if not self._all_done_triggered:
            self._all_done_triggered = True
            return self.reward_all_done / self.num_agents
        return 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_from_env(self, env) -> None:
        """Build grid list and region mapping from environment state."""
        state = env._get_state()

        all_grids: set[tuple] = set()
        for key in ("floors_to_distribute", "floors_to_vibrate",
                     "floors_to_level", "floors_to_cover", "floors_completed"):
            for pos in state.get(key, []):
                all_grids.add(tuple(pos))

        for g in all_grids:
            self._grid_stages[g] = self._stage_from_env_state(g, state)

        self._prev_grid_stages = dict(self._grid_stages)

        config = getattr(env, "_config", None)
        regions_cfg = None
        if config and "regions" in config:
            regions_cfg = config["regions"]
        elif self._regions_config:
            regions_cfg = self._regions_config

        if regions_cfg:
            self._regions = {
                name: [tuple(p) for p in grids]
                for name, grids in regions_cfg.items()
            }
        else:
            self._regions = {
                f"grid_{g[0]}_{g[1]}": [g] for g in sorted(all_grids)
            }

        self._grid_to_region = {}
        for name, grids in self._regions.items():
            for g in grids:
                self._grid_to_region[g] = name

        self._update_region_phases()

        if self.total_grids is None:
            self.total_grids = len(all_grids)

        if env is not None:
            positions = [tuple(robot["pos"]) for robot in env.robots]
            self._prev_positions = list(positions)
            self._prev_prev_positions = list(positions)

    def _stage_from_env_state(self, grid: tuple, state: dict) -> int:
        """Determine the current stage of a grid from environment state dicts."""
        completed = [tuple(p) for p in state.get("floors_completed", [])]
        if grid in completed:
            return 4
        to_cover = [tuple(p) for p in state.get("floors_to_cover", [])]
        if grid in to_cover:
            return 3
        to_level = [tuple(p) for p in state.get("floors_to_level", [])]
        if grid in to_level:
            return 2
        to_vibrate = [tuple(p) for p in state.get("floors_to_vibrate", [])]
        if grid in to_vibrate:
            return 1
        return 0

    def _sync_grid_stages(self, env) -> None:
        """Update grid stages from current environment state."""
        state = env._get_state()
        for g in self._grid_stages:
            self._grid_stages[g] = self._stage_from_env_state(g, state)

    def _update_region_phases(self) -> None:
        """Recompute region phases from current grid stages."""
        for name, grids in self._regions.items():
            stages = [self._grid_stages.get(g, 0) for g in grids]
            self._region_phases[name] = min(stages) if stages else 0

    def _compute_transition_rewards(self) -> Tuple[float, float, float]:
        """
        Compare previous vs current region phases.
        Returns (region_phase_bonus, region_done_bonus, global_done_bonus).
        """
        prev_phases: Dict[str, int] = {}
        for name, grids in self._regions.items():
            stages = [self._prev_grid_stages.get(g, 0) for g in grids]
            prev_phases[name] = min(stages) if stages else 0

        self._update_region_phases()

        region_phase_bonus = 0.0
        region_done_bonus = 0.0

        for name in self._regions:
            old_phase = prev_phases.get(name, 0)
            new_phase = self._region_phases[name]

            advances = new_phase - old_phase
            if advances > 0:
                region_phase_bonus += self.reward_region_phase * advances

                if new_phase >= NUM_STAGES and old_phase < NUM_STAGES:
                    region_done_bonus += self.reward_region_done

        global_done_bonus = 0.0
        if not self._all_done_triggered:
            if all(p >= NUM_STAGES for p in self._region_phases.values()):
                self._all_done_triggered = True
                global_done_bonus = self.reward_all_done

        return region_phase_bonus, region_done_bonus, global_done_bonus

    # ------------------------------------------------------------------
    # Distance-based potential shaping
    # ------------------------------------------------------------------

    _TASK_ORDER = ["distribute_floor", "vibrate_floor", "level_floor", "cover_surface"]
    _STAGE_TO_TARGET_SET = {
        0: "floors_to_distribute",
        1: "floors_to_vibrate",
        2: "floors_to_level",
        3: "floors_to_cover",
    }

    def _compute_distance_shaping(self, env) -> List[float]:
        """
        Potential-based reward shaping: reward = coeff * (phi' - phi)
        phi(agent) = -min_manhattan_distance to nearest relevant target grid.
        """
        cur_potentials = self._agent_potentials(env)
        shaping = [0.0] * self.num_agents

        if self._prev_potentials and len(self._prev_potentials) == self.num_agents:
            for i in range(self.num_agents):
                shaping[i] = self.distance_shaping_coeff * (cur_potentials[i] - self._prev_potentials[i])

        self._prev_potentials = cur_potentials
        return shaping

    def _agent_potentials(self, env) -> List[float]:
        """Compute potential for each agent based on distance to its relevant targets."""
        state = env._get_state()
        potentials = []

        target_sets = {
            "distribute_floor": set(map(tuple, state.get("floors_to_distribute", []))),
            "vibrate_floor": set(map(tuple, state.get("floors_to_vibrate", []))),
            "level_floor": set(map(tuple, state.get("floors_to_level", []))),
            "cover_surface": set(map(tuple, state.get("floors_to_cover", []))),
        }

        for robot in env.robots:
            rx, ry = robot["pos"]
            task_name = robot["tasks"][0] if robot["tasks"] else None
            targets = target_sets.get(task_name, set()) if task_name else set()

            if not targets:
                potentials.append(0.0)
                continue

            min_dist = min(abs(rx - tx) + abs(ry - ty) for tx, ty in targets)
            max_possible = env.width + env.height
            potentials.append(-min_dist / max_possible)

        return potentials
