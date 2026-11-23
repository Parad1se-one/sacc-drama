import torch

from sacc.envs.drama import DramaRMEnv
from sacc.utils.eval import evaluate_policy
from sacc.utils.recovery import (
    active_agent_indices,
    active_frontiers,
    agent_target_cells,
    assigned_agent_targets,
    frontier_unblock_action,
    state_signature,
)


def _complete_nodes(env: DramaRMEnv, cell, *nodes: str) -> None:
    env.env.completed_by_cell[cell].update(nodes)
    env.env._recompute_frontiers()
    env.env._update_task_maps()


def test_linear6_recovery_pipelines_all_currently_executable_nodes():
    env = DramaRMEnv(difficulty="linear6", seed=7)
    cell = sorted(env.env.task_cells)[0]

    assert set(active_frontiers(env)) == {"l1"}
    assert active_agent_indices(env) == [0]

    _complete_nodes(env, cell, "l1")
    assert {"l1", "l2"}.issubset(active_frontiers(env))
    assert active_agent_indices(env) == [0, 1]
    assert cell in agent_target_cells(env, 1)


def test_cross5_recovery_keeps_parallel_branch_frontiers_active():
    env = DramaRMEnv(difficulty="cross5", seed=7)
    cell = sorted(env.env.task_cells)[0]

    _complete_nodes(env, cell, "n1a")
    frontiers = active_frontiers(env)
    assert {"n1a", "n2a", "n3"}.issubset(frontiers)
    assert active_agent_indices(env) == [0, 1, 2]
    assert cell in agent_target_cells(env, 1)
    assert cell in agent_target_cells(env, 2)


def test_cross5_reentrant_role_waits_for_join_then_targets_n1b():
    env = DramaRMEnv(difficulty="cross5", seed=7)
    cell = sorted(env.env.task_cells)[0]

    _complete_nodes(env, cell, "n1a", "n2a")
    assert cell in env.env.frontiers["n3"]
    assert cell not in env.env.frontiers["n1b"]

    before = state_signature(env)
    _complete_nodes(env, cell, "n3")
    assert cell in env.env.frontiers["n1b"]
    assert cell in agent_target_cells(env, 0)
    assert state_signature(env) != before


def test_large_same_role_agents_receive_distinct_recovery_targets():
    env = DramaRMEnv(difficulty="large", seed=7)
    assignments = assigned_agent_targets(env)

    assert active_agent_indices(env) == [0, 1]
    assert assignments[0]
    assert assignments[1]
    assert assignments[0].isdisjoint(assignments[1])


def test_cross5_foreign_role_releases_but_compatible_reentrant_role_keeps_frontier():
    env = DramaRMEnv(difficulty="cross5", seed=7)
    cell = sorted(env.env.task_cells)[0]
    env.env.robots[1]["pos"] = cell
    env.env._update_passability()
    mask = env.get_action_masks()[1]
    action = frontier_unblock_action(env, 1, mask, [2, 1, 0, 3, 5])
    assert action is not None

    _complete_nodes(env, cell, "n1a", "n2a", "n3")
    env.env.robots[0]["pos"] = cell
    env.env._update_passability()
    mask = env.get_action_masks()[0]
    assert frontier_unblock_action(env, 0, mask, [2, 1, 0, 3, 5]) is None


class _StayActor(torch.nn.Module):
    def forward(self, observation):
        probabilities = torch.zeros(
            (observation.shape[0], 6), dtype=observation.dtype, device=observation.device
        )
        probabilities[:, 5] = 1.0
        return probabilities


def test_cycle_aware_execution_makes_progress_when_greedy_policy_stays():
    device = torch.device("cpu")
    actors = [_StayActor() for _ in range(6)]
    greedy = evaluate_policy(
        actors=actors,
        difficulty="linear6",
        episodes=1,
        seed=11,
        max_steps=20,
        device=device,
        policy_mode="greedy",
    )
    recovered = evaluate_policy(
        actors=actors,
        difficulty="linear6",
        episodes=1,
        seed=11,
        max_steps=80,
        device=device,
        policy_mode="cycle_aware_greedy",
        cycle_break_no_progress_steps=0,
        cycle_break_repeat_visits=2,
    )

    assert greedy["summary"]["mean_completion_ratio"] == 0.0
    assert recovered["summary"]["mean_completion_ratio"] > 0.0
    assert recovered["diagnostic_summary"]["mean_cycle_breaks"] > 0.0


def test_cycle_aware_execution_traverses_cross5_branch_and_join():
    recovered = evaluate_policy(
        actors=[_StayActor() for _ in range(5)],
        difficulty="cross5",
        episodes=1,
        seed=13,
        max_steps=160,
        device=torch.device("cpu"),
        policy_mode="cycle_aware_greedy",
        cycle_break_no_progress_steps=0,
        cycle_break_repeat_visits=2,
    )

    episode = recovered["episodes"][0]
    assert episode["completion_ratio"] > 0.0
    assert episode["diagnostics"]["cycle_breaks"] > 0
    assert any(
        event.get("reason") in {"frontier_route", "frontier_route_recovery"}
        for event in episode["diagnostics"]["cycle_break_events"]
    )
