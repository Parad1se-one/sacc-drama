import json

import pytest

from DRAMA import (
    build_fixed_start_set,
    generate_disjoint_start_sets,
    get_config,
    load_start_set,
    make_env,
    validate_start_positions,
    write_start_set,
)


def _as_keys(payload):
    return {tuple(tuple(pos) for pos in start) for start in payload["starts"]}


@pytest.mark.parametrize("difficulty", ["easy", "hard", "large"])
def test_fixed_reset_is_independent_of_seed(difficulty):
    env = make_env(difficulty, seed=42)
    first = env.start_positions
    env.reset(seed=999)
    assert env.start_positions == first


def test_disjoint_start_sets_are_deterministic_and_unique():
    first = generate_disjoint_start_sets(
        "hard",
        {"selection": 20, "test": 30},
        seed=20260913,
    )
    second = generate_disjoint_start_sets(
        "hard",
        {"selection": 20, "test": 30},
        seed=20260913,
    )

    assert first["selection"]["start_set_id"] == second["selection"]["start_set_id"]
    assert first["test"]["start_set_id"] == second["test"]["start_set_id"]
    selection = _as_keys(first["selection"])
    test = _as_keys(first["test"])
    assert len(selection) == 20
    assert len(test) == 30
    assert selection.isdisjoint(test)


def test_hard_starts_exclude_targets_obstacles_and_adjacent_robots():
    payload = generate_disjoint_start_sets("hard", {"test": 40}, seed=9)["test"]
    config = get_config("hard")
    for start in payload["starts"]:
        validate_start_positions(
            config,
            start,
            exclude_task_cells=True,
            min_pairwise_manhattan=2,
        )


def test_start_set_round_trip_and_digest_validation(tmp_path):
    payload = generate_disjoint_start_sets("easy", {"test": 5}, seed=11)["test"]
    path = write_start_set(tmp_path / "easy_test.json", payload)
    loaded = load_start_set(path, expected_difficulty="easy")
    assert loaded["start_set_id"] == payload["start_set_id"]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["starts"][0][0] = [5, 5]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_start_set(path)


def test_environment_uses_explicit_role_ordered_starts():
    payload = generate_disjoint_start_sets("easy", {"test": 1}, seed=7)["test"]
    start = tuple(tuple(pos) for pos in payload["starts"][0])
    env = make_env("easy", start_positions=start)
    assert env.start_positions == start
    assert tuple(robot["pos"] for robot in env.robots) == start


def test_fixed_start_set_is_versioned_and_matches_benchmark_config(tmp_path):
    payload = build_fixed_start_set("easy")
    config = get_config("easy")
    expected = tuple(
        tuple(pos)
        for robot in config["robots"].values()
        for pos in robot["init_positions"]
    )
    assert payload["count"] == 1
    assert payload["generation"]["sampling"] == "benchmark_fixed_role_order"
    assert tuple(tuple(pos) for pos in payload["starts"][0]) == expected

    path = write_start_set(tmp_path / "easy_fixed.json", payload)
    assert load_start_set(path)["start_set_id"] == payload["start_set_id"]


def test_large_scale_split_exercises_map_agent_task_and_obstacle_scaling():
    config = get_config("large")
    env = make_env("large", seed=42)
    payload = generate_disjoint_start_sets("large", {"selection": 5, "test": 10}, seed=20260914)

    assert (env.width, env.height) == (8, 8)
    assert env.num_agents == 8
    assert len(config["floors"]["to_distribute"]) == 24
    assert len(config["obstacles"]["static"]) == 6
    assert _as_keys(payload["selection"]).isdisjoint(_as_keys(payload["test"]))
    for split in payload.values():
        for start in split["starts"]:
            validate_start_positions(
                config, start, exclude_task_cells=True, min_pairwise_manhattan=2
            )
