"""DAG-aware deterministic recovery helpers for stalled execution.

The helpers operate only on the public state of :class:`DramaRMEnv`.  They
support both the original four-stage benchmark and configurable workflow DAGs.
They intentionally do not participate in PPO rollout collection: recovery is
an execution/evaluation layer whose effect is measured on fixed checkpoints.
"""

from __future__ import annotations

from collections import defaultdict, deque


MOVE_DELTAS = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
LEGACY_STAGE_BY_FRONTIER = (
    ("floors_to_distribute", "material_distributer"),
    ("floors_to_vibrate", "vibrator"),
    ("floors_to_level", "leveler"),
    ("floors_to_cover", "film_cover"),
)


def _robot_role(robot: dict) -> str:
    return str(robot.get("role", robot.get("type", "")))


def active_frontiers(env) -> dict[str, set[tuple[int, int]]]:
    """Return every currently executable node frontier.

    Workflow DAGs may expose several incomparable nodes at once.  The legacy
    environment retains the validated earliest-nonempty-stage semantics.
    """

    benchmark = env.env
    if hasattr(benchmark, "frontiers") and hasattr(benchmark, "node_by_id"):
        return {
            str(node_id): set(benchmark.frontiers[node_id])
            for node_id in benchmark.topological_order
            if benchmark.frontiers[node_id]
        }
    for frontier_attr, role in LEGACY_STAGE_BY_FRONTIER:
        cells = set(getattr(benchmark, frontier_attr))
        if cells:
            return {role: cells}
    return {}


def agent_target_cells(env, agent_idx: int) -> set[tuple[int, int]]:
    benchmark = env.env
    role = _robot_role(benchmark.robots[agent_idx])
    frontiers = active_frontiers(env)
    if hasattr(benchmark, "node_by_id"):
        targets: set[tuple[int, int]] = set()
        for node_id, cells in frontiers.items():
            if str(benchmark.node_by_id[node_id]["role"]) == role:
                targets.update(cells)
        return targets
    return set(frontiers.get(role, ()))


def reserved_frontier(env) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for frontier in active_frontiers(env).values():
        cells.update(frontier)
    return cells


def active_agent_indices(env) -> list[int]:
    return [
        agent_idx
        for agent_idx in range(env.num_agents)
        if agent_target_cells(env, agent_idx)
    ]


def assigned_agent_targets(env) -> dict[int, set[tuple[int, int]]]:
    """Assign distinct nearby cells to same-role robots when possible.

    Linear-6 and Cross-5 have one robot per role, while Large has two.  The
    deterministic assignment prevents same-role robots from being routed to
    the same construction cell without introducing a learned coordinator.
    """

    benchmark = env.env
    groups: dict[str, list[int]] = defaultdict(list)
    for agent_idx in active_agent_indices(env):
        groups[_robot_role(benchmark.robots[agent_idx])].append(agent_idx)
    assignments: dict[int, set[tuple[int, int]]] = {}
    for agents in groups.values():
        agents = sorted(agents)
        union = set().union(*(agent_target_cells(env, idx) for idx in agents))
        if len(union) < len(agents):
            for idx in agents:
                assignments[idx] = agent_target_cells(env, idx)
            continue
        remaining_agents = set(agents)
        remaining_cells = set(union)
        while remaining_agents:
            distance, agent_idx, cell = min(
                (
                    abs(int(benchmark.robots[idx]["pos"][0]) - cell[0])
                    + abs(int(benchmark.robots[idx]["pos"][1]) - cell[1]),
                    idx,
                    cell,
                )
                for idx in remaining_agents
                for cell in remaining_cells
                if cell in agent_target_cells(env, idx)
            )
            del distance
            assignments[agent_idx] = {cell}
            remaining_agents.remove(agent_idx)
            remaining_cells.remove(cell)
    return assignments


def _distance_to_targets(
    env,
    *,
    source: tuple[int, int],
    targets: set[tuple[int, int]],
    agent_idx: int,
) -> int | None:
    benchmark = env.env
    if source in targets:
        return 0
    blocked = set(benchmark.obstacles) | set(benchmark.dynamic_obstacles)
    blocked.update(
        tuple(robot["pos"])
        for index, robot in enumerate(benchmark.robots)
        if index != agent_idx
    )
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        (x, y), depth = queue.popleft()
        for dx, dy in MOVE_DELTAS.values():
            neighbor = (x + dx, y + dy)
            if not (0 <= neighbor[0] < benchmark.width and 0 <= neighbor[1] < benchmark.height):
                continue
            if neighbor in visited:
                continue
            if neighbor in targets:
                return depth + 1
            if neighbor in blocked:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))
    return None


def frontier_route_action(
    env,
    agent_idx: int,
    action_mask,
    ranked_actions: list[int],
    *,
    targets: set[tuple[int, int]] | None = None,
) -> int | None:
    """Choose a legal policy-ranked move that strictly approaches a frontier."""

    benchmark = env.env
    targets = set(agent_target_cells(env, agent_idx) if targets is None else targets)
    start = tuple(benchmark.robots[agent_idx]["pos"])
    if not targets or start in targets:
        return None
    current_distance = _distance_to_targets(
        env, source=start, targets=targets, agent_idx=agent_idx
    )
    if current_distance is None:
        return None
    policy_rank = {action: rank for rank, action in enumerate(ranked_actions)}
    candidates = []
    for action, (dx, dy) in MOVE_DELTAS.items():
        if not bool(action_mask[action]):
            continue
        candidate = (start[0] + dx, start[1] + dy)
        candidate_distance = _distance_to_targets(
            env, source=candidate, targets=targets, agent_idx=agent_idx
        )
        if candidate_distance is not None and candidate_distance < current_distance:
            candidates.append((candidate_distance, policy_rank.get(action, len(policy_rank)), action))
    return min(candidates)[2] if candidates else None


def frontier_unblock_action(env, agent_idx: int, action_mask, ranked_actions: list[int]) -> int | None:
    """Choose a legal policy-ranked move that releases an incompatible frontier."""

    benchmark = env.env
    frontier = reserved_frontier(env)
    x, y = map(int, benchmark.robots[agent_idx]["pos"])
    if (x, y) not in frontier or (x, y) in agent_target_cells(env, agent_idx):
        return None
    fallback = None
    for action in ranked_actions:
        if action not in MOVE_DELTAS or not bool(action_mask[action]):
            continue
        dx, dy = MOVE_DELTAS[action]
        fallback = action if fallback is None else fallback
        if (x + dx, y + dy) not in frontier:
            return action
    return fallback


def frontier_reentry_guard_action(
    env, agent_idx: int, action_mask, ranked_actions: list[int], proposed_action: int
) -> int | None:
    """Return an alternative only when an incompatible agent re-enters a frontier."""

    benchmark = env.env
    frontier = reserved_frontier(env)
    compatible = agent_target_cells(env, agent_idx)
    x, y = map(int, benchmark.robots[agent_idx]["pos"])

    def destination(action: int) -> tuple[int, int]:
        dx, dy = MOVE_DELTAS.get(int(action), (0, 0))
        return x + dx, y + dy

    proposed = destination(proposed_action)
    if proposed not in frontier or proposed in compatible:
        return None
    for action in ranked_actions:
        candidate = destination(action)
        if bool(action_mask[action]) and (candidate not in frontier or candidate in compatible):
            return int(action)
    return None


def state_signature(env) -> tuple:
    benchmark = env.env
    positions = tuple(tuple(robot["pos"]) for robot in benchmark.robots)
    if hasattr(benchmark, "completed_by_cell"):
        completed = tuple(
            (tuple(cell), tuple(sorted(nodes, key=benchmark.node_index.__getitem__)))
            for cell, nodes in sorted(benchmark.completed_by_cell.items())
        )
        return positions, completed
    return (
        positions,
        tuple(sorted(benchmark.floors_to_distribute)),
        tuple(sorted(benchmark.floors_to_vibrate)),
        tuple(sorted(benchmark.floors_to_level)),
        tuple(sorted(benchmark.floors_to_cover)),
        tuple(sorted(benchmark.floors_completed)),
    )
