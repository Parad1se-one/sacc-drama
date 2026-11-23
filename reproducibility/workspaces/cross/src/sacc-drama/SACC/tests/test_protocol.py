from pathlib import Path

import pytest

from DRAMA import build_fixed_start_set, generate_disjoint_start_sets, write_start_set
from sacc.algorithms.networks import CategoricalActor
from sacc.envs.drama import DramaRMEnv
from sacc.utils.eval import _resolve_evaluation_starts


def test_sacc_fixed_and_random_start_protocols_are_explicit():
    fixed = DramaRMEnv(difficulty="easy", seed=42, start_protocol="fixed")
    fixed_first = fixed.get_start_positions()
    fixed.reset(seed=43)
    assert fixed.get_start_positions() == fixed_first

    random = DramaRMEnv(difficulty="easy", seed=42, start_protocol="random")
    random_first = random.get_start_positions()
    random.reset(seed=43)
    assert random.get_start_positions() != random_first


def test_versioned_start_set_controls_episode_count(tmp_path: Path):
    payload = generate_disjoint_start_sets("easy", {"selection": 4}, seed=101)["selection"]
    path = write_start_set(tmp_path / "selection.json", payload)
    starts, metadata = _resolve_evaluation_starts(
        difficulty="easy",
        episodes=4,
        seed=0,
        start_protocol="fixed",
        start_set_path=str(path),
    )
    assert len(starts) == 4
    assert metadata["start_set_id"] == payload["start_set_id"]
    assert metadata["unique_start_count"] == 4

    with pytest.raises(ValueError, match="contains 4"):
        _resolve_evaluation_starts(
            difficulty="easy",
            episodes=3,
            seed=0,
            start_protocol="fixed",
            start_set_path=str(path),
        )


def test_observation_dimensions_match_current_nine_feature_code():
    env = DramaRMEnv(difficulty="easy", seed=7)
    assert env.base_obs_dim == 188
    assert env.rm_obs_dim == 9
    assert env.obs_dim == 197
    assert env.state_dim == 225
    assert len(env.build_base_obs_list()) == env.num_agents
    assert all(len(obs) == 188 for obs in env.build_base_obs_list())


def test_large_scale_observation_and_state_dimensions():
    env = DramaRMEnv(difficulty="large", seed=7)
    assert env.num_agents == 8
    assert env.base_obs_dim == 336
    assert env.rm_obs_dim == 9
    assert env.obs_dim == 345
    assert env.state_dim == 409
    assert len(env.build_base_obs_list()) == 8
    assert all(len(obs) == 336 for obs in env.build_base_obs_list())
    assert env.total_stage_tasks == 96


def test_versioned_fixed_start_is_labelled_explicitly(tmp_path: Path):
    payload = build_fixed_start_set("easy")
    path = write_start_set(tmp_path / "fixed.json", payload)
    starts, metadata = _resolve_evaluation_starts(
        difficulty="easy",
        episodes=1,
        seed=0,
        start_protocol="fixed",
        start_set_path=str(path),
    )
    assert len(starts) == 1
    assert metadata["start_protocol"] == "versioned_fixed"
    assert metadata["unique_start_count"] == 1


def test_base_actor_checkpoint_is_auto_detected_and_evaluable(tmp_path: Path):
    import torch

    from sacc.utils.eval import evaluate_checkpoint

    actors = [CategoricalActor(188, 6) for _ in range(4)]
    checkpoint = tmp_path / "ippo.pth"
    torch.save(
        {
            "actors": [actor.state_dict() for actor in actors],
            "meta": {"algorithm": "ippo", "difficulty": "easy", "max_steps": 1},
        },
        checkpoint,
    )
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        episodes=1,
        seed=42,
        device="cpu",
        max_steps=1,
    )
    assert result["algo"] == "ippo"
    assert result["actor_observation_mode"] == "base"
    assert result["summary"]["mean_objective_return"] == result["summary"]["mean_environment_return"]
    assert result["diagnostic_summary"]["mean_progress_steps"] >= 0.0
    diagnostics = result["episodes"][0]["diagnostics"]
    assert sum(diagnostics["status_counts"].values()) == 4
    assert len(diagnostics["action_counts"]) == 6
