#!/usr/bin/env python3
"""Auditable tabular MAHRM adaptation for the four-role DRAMA simulator.

The paper's hierarchy is instantiated as team-completion -> construction-cell
stage RMs -> primitive (role, cell, stage) event RMs.  The fixed DRAMA roles
restrict legal agent partitions: each agent can pursue only events for its
own stage.  The upper policy selects a cell subtask for every agent, while
the lower policies learn masked primitive actions with counterfactual QRM
updates.  Options terminate on a selected RM transition or after 50 steps.
This is an adaptation, not the authors' unpublished experiment code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MAX_OPTION_STEPS = 50
SELECTION_INTERVAL = 200
ROLE_EVENTS = ("distribute_floor", "vibrate_floor", "level_floor", "cover_surface")


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _positions(starts) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(map(int, point)) for point in starts)


def _stages(env, cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    stages = []
    for cell in cells:
        if cell in env.floors_to_distribute:
            stages.append(0)
        elif cell in env.floors_to_vibrate:
            stages.append(1)
        elif cell in env.floors_to_level:
            stages.append(2)
        elif cell in env.floors_to_cover:
            stages.append(3)
        elif cell in env.floors_completed:
            stages.append(4)
        else:
            raise RuntimeError(f"Construction cell {cell} absent from all RM states")
    return tuple(stages)


def _argmax(q: np.ndarray, choices: tuple[int, ...], rng: np.random.Generator, explore: bool) -> int:
    if explore and rng.random() < EPSILON:
        return int(rng.choice(choices))
    values = q[list(choices)]
    winners = [choices[i] for i in np.flatnonzero(values == values.max())]
    return int(rng.choice(winners)) if explore else int(winners[0])


@dataclass
class Episode:
    completion: float
    success: bool
    steps: int
    environment_return: float
    start_positions: tuple[tuple[int, int], ...]


class MAHRMDrama:
    def __init__(self, config: dict, *, seed: int):
        from DRAMA.env import DramaBenchmarkEnv

        self.env = DramaBenchmarkEnv(config)
        self.cells = tuple(sorted(tuple(map(int, p)) for p in config["floors"]["to_distribute"]))
        self.cell_lookup = {p: i for i, p in enumerate(self.cells)}
        self.rng = np.random.default_rng(seed)
        self.low_q: dict[tuple[int, int, tuple[int, int]], np.ndarray] = {}
        self.high_q: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def _low(self, role: int, cell: int, position: tuple[int, int]) -> np.ndarray:
        key = (role, cell, position)
        if key not in self.low_q:
            self.low_q[key] = np.zeros(6, dtype=np.float32)
        return self.low_q[key]

    def _high(self, role: int, stages: tuple[int, ...]) -> np.ndarray:
        key = (role, stages)
        if key not in self.high_q:
            self.high_q[key] = np.zeros(len(self.cells), dtype=np.float32)
        return self.high_q[key]

    def _eligible(self, role: int, stages: tuple[int, ...]) -> tuple[int, ...]:
        # A fixed-role robot cannot advance a cell whose own stage has passed.
        return tuple(i for i, stage in enumerate(stages) if stage <= role)

    def _choose_targets(self, stages: tuple[int, ...], train: bool) -> tuple[int, ...]:
        targets = []
        for role in range(4):
            candidates = self._eligible(role, stages)
            targets.append(_argmax(self._high(role, stages), candidates, self.rng, train) if candidates else -1)
        return tuple(targets)

    def run_episode(self, starts, *, max_steps: int, train: bool) -> Episode:
        self.env.reset(start_positions=starts)
        initial = _positions(starts)
        stages = _stages(self.env, self.cells)
        targets = self._choose_targets(stages, train)
        option_stages = stages
        option_steps = 0
        discounted_option_rewards = np.zeros(4, dtype=np.float32)
        total_return = 0.0

        for step in range(1, max_steps + 1):
            old_positions = tuple(tuple(robot["pos"]) for robot in self.env.robots)
            masks = self.env.get_action_masks()
            actions = []
            for role, target in enumerate(targets):
                if target < 0:
                    actions.append(5)
                    continue
                legal = tuple(int(a) for a in np.flatnonzero(masks[role]))
                actions.append(_argmax(self._low(role, target, old_positions[role]), legal, self.rng, train))

            rewards, infos = self.env.step(actions)
            total_return += float(sum(rewards))
            next_positions = tuple(tuple(robot["pos"]) for robot in self.env.robots)
            next_stages = _stages(self.env, self.cells)
            next_masks = self.env.get_action_masks()
            events: set[tuple[int, int]] = set()
            for role, info in enumerate(infos):
                if info["status"] == ROLE_EVENTS[role]:
                    cell = self.cell_lookup[old_positions[role]]
                    assert next_stages[cell] == stages[cell] + 1
                    events.add((role, cell))

            if train:
                # Counterfactual QRM: one observed transition updates each
                # primitive cell RM for that role, not only the selected RM.
                for role in range(4):
                    if targets[role] < 0:
                        continue
                    for cell in range(len(self.cells)):
                        old_q = self._low(role, cell, old_positions[role])
                        hit = (role, cell) in events
                        if next_stages[cell] > role and not hit:
                            continue
                        future = 0.0
                        if not hit:
                            future = float(self._low(role, cell, next_positions[role])[next_masks[role]].max())
                        td = float(hit) + GAMMA * future - float(old_q[actions[role]])
                        old_q[actions[role]] += ALPHA * td

            for role, target in enumerate(targets):
                if target >= 0 and (role, target) in events:
                    discounted_option_rewards[role] += GAMMA ** option_steps
            option_steps += 1
            stages = next_stages
            done = self.env.is_all_done()
            changed_selected_rm = any((role, target) in events for role, target in enumerate(targets) if target >= 0)
            option_end = done or changed_selected_rm or option_steps >= MAX_OPTION_STEPS or step == max_steps
            if option_end:
                if train:
                    for role, target in enumerate(targets):
                        if target < 0:
                            continue
                        q = self._high(role, option_stages)
                        candidates = self._eligible(role, stages)
                        next_value = float(self._high(role, stages)[list(candidates)].max()) if candidates and not done else 0.0
                        td = float(discounted_option_rewards[role]) + GAMMA ** option_steps * next_value - float(q[target])
                        q[target] += ALPHA * td
                if done:
                    return Episode(self.env.get_completion_ratio(), True, step, total_return, initial)
                targets = self._choose_targets(stages, train)
                option_stages = stages
                option_steps = 0
                discounted_option_rewards.fill(0)

        return Episode(self.env.get_completion_ratio(), False, max_steps, total_return, initial)

    def snapshot(self) -> dict:
        return {"low_q": self.low_q, "high_q": self.high_q, "cells": self.cells, "algorithm": "MAHRM_DRAMA_ADAPTED_V1"}

    def load(self, snapshot: dict) -> None:
        if snapshot["algorithm"] != "MAHRM_DRAMA_ADAPTED_V1" or tuple(snapshot["cells"]) != self.cells:
            raise ValueError("Incompatible checkpoint")
        self.low_q = snapshot["low_q"]
        self.high_q = snapshot["high_q"]


def _evaluate(model: MAHRMDrama, starts: list, max_steps: int) -> list[Episode]:
    return [model.run_episode(start, max_steps=max_steps, train=False) for start in starts]


def _manifest(path: Path, difficulty: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["difficulty"] != difficulty or payload["count"] != len(payload["starts"]):
        raise ValueError(f"Manifest mismatch: {path}")
    return payload


def _episode_json(episode: Episode, index: int) -> dict:
    return {
        "episode": index,
        "start_positions": [list(p) for p in episode.start_positions],
        "completion_ratio": episode.completion,
        "success": episode.success,
        "steps": episode.steps,
        "environment_return": episode.environment_return,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--difficulty", choices=("easy", "hard"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--train-starts", type=Path)
    parser.add_argument("--selection-starts", type=Path)
    parser.add_argument("--test-starts", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()

    from DRAMA.specs import get_spec_data

    model = MAHRMDrama(get_spec_data(args.difficulty)["config"], seed=args.seed)
    if args.mode == "train":
        if args.train_starts is None or args.selection_starts is None:
            parser.error("training requires --train-starts and --selection-starts")
        train = _manifest(args.train_starts, args.difficulty)
        selection = _manifest(args.selection_starts, args.difficulty)
        if len({_positions(s) for s in train["starts"]} & {_positions(s) for s in selection["starts"]}):
            raise RuntimeError("Training and selection starts overlap")
        record = {
            "status": "running", "algorithm": "MAHRM_DRAMA_ADAPTED_V1", "difficulty": args.difficulty,
            "seed": args.seed, "episodes": train["count"], "max_steps": args.max_steps,
            "train_starts_sha256": _sha256(args.train_starts),
            "selection_starts_sha256": _sha256(args.selection_starts),
            "checkpoint": str(args.checkpoint),
            "hyperparameters": {"alpha": ALPHA, "gamma": GAMMA, "epsilon": EPSILON,
                                "max_option_steps": MAX_OPTION_STEPS, "selection_interval": SELECTION_INTERVAL},
            "selection": [], "trained_episodes": 0,
        }
        if args.record.exists():
            raise RuntimeError(f"Record already exists: {args.record}")
        _atomic_json(args.record, record)
        best = (-1.0, -1.0)
        for episode, start in enumerate(train["starts"], 1):
            model.run_episode(start, max_steps=args.max_steps, train=True)
            if episode % SELECTION_INTERVAL == 0 or episode == train["count"]:
                rows = _evaluate(model, selection["starts"], args.max_steps)
                mean_completion = float(np.mean([r.completion for r in rows]))
                mean_success = float(np.mean([r.success for r in rows]))
                record["selection"].append({"episode": episode, "completion": mean_completion, "success": mean_success})
                record["trained_episodes"] = episode
                if (mean_completion, mean_success) > best:
                    best = (mean_completion, mean_success)
                    _atomic_pickle(args.checkpoint, model.snapshot())
                    record["selected_episode"] = episode
                    record["checkpoint_sha256"] = _sha256(args.checkpoint)
                _atomic_json(args.record, record)
                print(json.dumps({"episode": episode, "selection_completion": mean_completion,
                                  "selection_success": mean_success, "best_episode": record["selected_episode"]}), flush=True)
        record["status"] = "completed"
        _atomic_json(args.record, record)
    else:
        if args.test_starts is None:
            parser.error("evaluation requires --test-starts")
        if args.record.exists():
            raise RuntimeError(f"Evaluation record already exists: {args.record}")
        starts = _manifest(args.test_starts, args.difficulty)
        with args.checkpoint.open("rb") as stream:
            model.load(pickle.load(stream))
        rows = _evaluate(model, starts["starts"], args.max_steps)
        record = {
            "status": "completed", "algorithm": "MAHRM_DRAMA_ADAPTED_V1", "method": "mahrm",
            "difficulty": args.difficulty, "seed": args.seed, "max_steps": args.max_steps,
            "episodes": [_episode_json(row, i) for i, row in enumerate(rows)],
            "mean_completion_ratio": float(np.mean([r.completion for r in rows])),
            "mean_success_rate": float(np.mean([r.success for r in rows])),
            "mean_environment_return": float(np.mean([r.environment_return for r in rows])),
            "start_set_id": starts["start_set_id"], "start_set_sha256": _sha256(args.test_starts),
            "checkpoint_sha256": _sha256(args.checkpoint),
        }
        _atomic_json(args.record, record)
        print(json.dumps({"record": str(args.record), "completion": record["mean_completion_ratio"],
                          "success": record["mean_success_rate"]}), flush=True)


if __name__ == "__main__":
    main()

