from pathlib import Path

import numpy as np
import pytest
import torch

from DRAMA.starts import (
    default_min_pairwise_manhattan,
    load_start_set,
    sample_start_positions,
    validate_start_positions,
)
from train_baseline import make_env
from utils.baseline_eval import evaluate_policy, make_actor_selector


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_random_baseline_uses_submitted_start_distribution(difficulty):
    env, config, _ = make_env(difficulty, train_start_protocol="random", seed=123)
    minimum = default_min_pairwise_manhattan(difficulty)
    starts = []
    for offset in range(2):
        env.reset()
        positions = tuple(robot["pos"] for robot in env.robots)
        assert positions == sample_start_positions(
            config, seed=123 + offset, exclude_task_cells=True,
            min_pairwise_manhattan=minimum,
        )
        validate_start_positions(
            config, positions, exclude_task_cells=True,
            min_pairwise_manhattan=minimum,
        )
        starts.append(positions)
    assert starts[0] != starts[1]

    repeated, _, _ = make_env(difficulty, train_start_protocol="random", seed=123)
    repeated.reset()
    assert tuple(robot["pos"] for robot in repeated.robots) == starts[0]


def test_random_start_does_not_reseed_baseline_exploration():
    env, _, _ = make_env("easy", train_start_protocol="random", seed=42)
    original_state = np.random.get_state()
    try:
        np.random.seed(2718)
        expected = np.random.random()
        np.random.seed(2718)
        env.reset()
        assert np.random.random() == expected
    finally:
        np.random.set_state(original_state)


def test_fixed_baseline_start_is_unchanged():
    env, _, _ = make_env("easy")
    env.reset(seed=42)
    first = tuple(robot["pos"] for robot in env.robots)
    env.reset(seed=999)
    assert tuple(robot["pos"] for robot in env.robots) == first


def test_validation_uses_every_frozen_start_exactly_once():
    repo_root = Path(__file__).resolve().parents[4]
    start_path = repo_root / "experiments/protocols/v3/easy_submission_selection_starts.json"
    start_set = load_start_set(start_path, expected_difficulty="easy")
    kwargs = dict(
        action_selector=lambda observations, masks: [5] * len(observations),
        difficulty="easy", episodes=start_set["count"], seed=42,
        algo="ippo", max_steps=1, start_set_path=start_path,
    )
    result = evaluate_policy(**kwargs)
    assert result["selection_start_set"]["start_set_id"] == start_set["start_set_id"]
    assert [entry["start_positions"] for entry in result["episodes"]] == [
        [list(pos) for pos in start] for start in start_set["starts"]
    ]
    with pytest.raises(ValueError, match="Selection episodes"):
        evaluate_policy(**{**kwargs, "episodes": start_set["count"] - 1})



def test_frozen_selection_does_not_change_training_numpy_rng():
    repo_root = Path(__file__).resolve().parents[4]
    start_path = repo_root / "experiments/protocols/v3/easy_submission_selection_starts.json"
    start_set = load_start_set(start_path, expected_difficulty="easy")
    original_state = np.random.get_state()
    try:
        np.random.seed(2718)
        expected = np.random.random()
        np.random.seed(2718)
        evaluate_policy(
            action_selector=lambda observations, masks: [5] * len(observations),
            difficulty="easy", episodes=start_set["count"], seed=42,
            algo="ippo", max_steps=1, start_set_path=start_path,
        )
        assert np.random.random() == expected
    finally:
        np.random.set_state(original_state)


def test_tempered_selector_is_seeded_legal_and_rng_isolated():
    class Actor:
        def __call__(self, obs):
            return torch.tensor([[0.45, 0.35, 0.20]], device=obs.device)

    kwargs = dict(actors=[Actor()], policy_mode="tempered", temperature=0.35, rng_seed=17)
    torch_state = torch.get_rng_state()
    selector = make_actor_selector(**kwargs)
    first = [selector([[0.0]], [[True, False, True]])[0] for _ in range(30)]
    second_selector = make_actor_selector(**kwargs)
    second = [second_selector([[0.0]], [[True, False, True]])[0] for _ in range(30)]
    assert first == second
    assert set(first).issubset({0, 2})
    assert torch.equal(torch.get_rng_state(), torch_state)
