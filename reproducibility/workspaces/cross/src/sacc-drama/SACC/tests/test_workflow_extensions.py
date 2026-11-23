from __future__ import annotations

import numpy as np
import pytest

from DRAMA import get_config, make_env
from DRAMA.workflow_env import topological_order
from sacc.envs.drama import DramaRMEnv


def _perform(env, role: str, cell: tuple[int, int]) -> str:
    robot = next(robot for robot in env.robots if robot["role"] == role)
    original = robot["pos"]
    robot["pos"] = cell
    reward, info = env._execute_task(robot)
    robot["pos"] = original
    env._update_task_maps()
    assert reward == 1.0
    return str(info["workflow_node"])


def test_workflow_graph_rejects_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        topological_order(["a", "b"], [("a", "b"), ("b", "a")])


def test_linear6_requires_all_six_roles_in_order():
    env = make_env("linear6")
    cell = sorted(env.task_cells)[0]
    assert env.topological_order == ("l1", "l2", "l3", "l4", "l5", "l6")
    assert cell in env.frontiers["l1"]
    for index in range(1, 7):
        node_id = f"l{index}"
        assert _perform(env, f"role_{index}", cell) == node_id
        if index < 6:
            assert cell in env.frontiers[f"l{index + 1}"]
    assert env.completed_by_cell[cell] == set(env.node_ids)
    assert env.get_completion_ratio() == pytest.approx(6 / 72)


def test_cross5_branches_then_joins_before_reentrant_roles():
    env = make_env("cross5")
    cell = sorted(env.task_cells)[0]
    assert _perform(env, "role_1", cell) == "n1a"
    assert cell in env.frontiers["n2a"]
    assert cell in env.frontiers["n3"]

    assert _perform(env, "role_2", cell) == "n2a"
    assert cell not in env.frontiers["n1b"]
    assert _perform(env, "role_3", cell) == "n3"
    assert cell in env.frontiers["n1b"]

    assert _perform(env, "role_1", cell) == "n1b"
    assert _perform(env, "role_2", cell) == "n2b"
    assert _perform(env, "role_4", cell) == "n4"
    assert _perform(env, "role_5", cell) == "n5"
    assert env.completed_by_cell[cell] == set(env.node_ids)
    assert env.get_completion_ratio() == pytest.approx(7 / 84)


@pytest.mark.parametrize("difficulty,agents", [("linear6", 6), ("cross5", 5)])
def test_local_and_team_rm_interfaces_are_finite(difficulty: str, agents: int):
    local = DramaRMEnv(difficulty=difficulty, seed=42, rm_mode="local")
    team = DramaRMEnv(difficulty=difficulty, seed=42, rm_mode="team")
    assert local.num_agents == team.num_agents == agents
    assert local.rm_obs_dim == 9
    assert team.rm_obs_dim == 2 * len(team.env.node_ids) + 1
    assert all(np.isfinite(obs).all() for obs in local.build_obs_list())
    assert all(np.isfinite(obs).all() for obs in team.build_obs_list())
    assert all(
        np.array_equal(team.build_obs_list()[0][-team.rm_obs_dim :], obs[-team.rm_obs_dim :])
        for obs in team.build_obs_list()
    )


def test_cross5_local_rm_tracks_mixed_reentrant_frontiers():
    wrapped = DramaRMEnv(difficulty="cross5", seed=42, rm_mode="local")
    env = wrapped.env
    role1_rm = wrapped.rm.agent_rms[0]
    cell = sorted(env.task_cells)[0]

    initial = role1_rm._metrics(env)
    assert initial["available_count"] == len(env.task_cells)
    assert initial["target_node_index"] == 0.0

    assert _perform(env, "role_1", cell) == "n1a"
    assert _perform(env, "role_2", cell) == "n2a"
    assert _perform(env, "role_3", cell) == "n3"
    mixed = role1_rm._metrics(env)
    assert mixed["available_count"] == len(env.task_cells)
    assert 0.0 < mixed["target_node_index"] < env.node_index["n1b"]
    assert mixed["upstream_clear_ratio"] > initial["upstream_clear_ratio"]

    for other_cell in sorted(env.task_cells)[1:]:
        assert _perform(env, "role_1", other_cell) == "n1a"
    assert cell in env.frontiers["n1b"]


def test_cross5_reuses_roles_one_and_two():
    config = get_config("cross5")
    assert config["robots"]["role_1"]["tasks"] == ["n1a", "n1b"]
    assert config["robots"]["role_2"]["tasks"] == ["n2a", "n2b"]
