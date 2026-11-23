from __future__ import annotations

from copy import deepcopy

TASK_STAGES = (
    "distribute_floor",
    "vibrate_floor",
    "level_floor",
    "cover_surface",
)

ACTION_STR = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "PERFORM",
    5: "STAY",
}

# Public benchmark naming used in the submission.
BENCHMARK_SPECS = {
    "easy": {
        "meta": {
            "level": "Easy",
            "description": "2 construction cells, 2 regions, no obstacles",
            "complexity": "low",
            "expected_steps": "~20-40",
        },
        "config": {
            "grid_width": 6,
            "grid_height": 6,
            "num_robots": 4,
            "robots": {
                "material_distributer": {
                    "num": 1,
                    "tasks": ["distribute_floor"],
                    "init_positions": [(0, 0)],
                },
                "vibrator": {
                    "num": 1,
                    "tasks": ["vibrate_floor"],
                    "init_positions": [(1, 1)],
                },
                "leveler": {
                    "num": 1,
                    "tasks": ["level_floor"],
                    "init_positions": [(2, 0)],
                },
                "film_cover": {
                    "num": 1,
                    "tasks": ["cover_surface"],
                    "init_positions": [(2, 2)],
                },
            },
            "obstacles": {"static": [], "dynamic": []},
            "walls": {"init": [], "to_distribute": [], "to_vibrate": []},
            "floors": {
                "to_distribute": [(2, 2), (2, 3)],
                "to_vibrate": [],
                "to_level": [],
                "to_cover": [],
            },
            "regions": {"A": [(2, 2)], "B": [(2, 3)]},
            "task_dependencies": {
                "distribute_floor": [],
                "vibrate_floor": ["distribute_floor"],
                "level_floor": ["vibrate_floor"],
                "cover_surface": ["level_floor"],
            },
        },
    },
    "hard": {
        "meta": {
            "level": "Hard",
            "description": "10 construction cells, 4 regions, 1 static obstacle",
            "complexity": "high",
            "expected_steps": "~100-200",
        },
        "config": {
            "grid_width": 6,
            "grid_height": 6,
            "num_robots": 4,
            "robots": {
                "material_distributer": {
                    "num": 1,
                    "tasks": ["distribute_floor"],
                    "init_positions": [(0, 0)],
                },
                "vibrator": {
                    "num": 1,
                    "tasks": ["vibrate_floor"],
                    "init_positions": [(1, 1)],
                },
                "leveler": {
                    "num": 1,
                    "tasks": ["level_floor"],
                    "init_positions": [(2, 0)],
                },
                "film_cover": {
                    "num": 1,
                    "tasks": ["cover_surface"],
                    "init_positions": [(2, 2)],
                },
            },
            "obstacles": {"static": [(3, 3)], "dynamic": []},
            "walls": {"init": [], "to_distribute": [], "to_vibrate": []},
            "floors": {
                "to_distribute": [
                    (0, 1),
                    (0, 2),
                    (1, 2),
                    (0, 4),
                    (1, 4),
                    (1, 5),
                    (4, 0),
                    (5, 0),
                    (4, 4),
                    (5, 5),
                ],
                "to_vibrate": [],
                "to_level": [],
                "to_cover": [],
            },
            "regions": {
                "TOP_LEFT": [(0, 1), (0, 2), (1, 2)],
                "TOP_RIGHT": [(0, 4), (1, 4), (1, 5)],
                "BOTTOM_LEFT": [(4, 0), (5, 0)],
                "BOTTOM_RIGHT": [(4, 4), (5, 5)],
            },
            "task_dependencies": {
                "distribute_floor": [],
                "vibrate_floor": ["distribute_floor"],
                "level_floor": ["vibrate_floor"],
                "cover_surface": ["level_floor"],
            },
        },
    },
    "large": {
        "meta": {
            "level": "Large",
            "description": "24 construction cells, 4 regions, 6 static obstacles, 8 robots",
            "complexity": "scale-stress",
            "expected_steps": "~250-600",
        },
        "config": {
            "grid_width": 8,
            "grid_height": 8,
            "num_robots": 8,
            "robots": {
                "material_distributer": {
                    "num": 2,
                    "tasks": ["distribute_floor"],
                    "init_positions": [(0, 0), (0, 7)],
                },
                "vibrator": {
                    "num": 2,
                    "tasks": ["vibrate_floor"],
                    "init_positions": [(1, 0), (1, 7)],
                },
                "leveler": {
                    "num": 2,
                    "tasks": ["level_floor"],
                    "init_positions": [(2, 0), (2, 7)],
                },
                "film_cover": {
                    "num": 2,
                    "tasks": ["cover_surface"],
                    "init_positions": [(3, 0), (3, 7)],
                },
            },
            "obstacles": {
                "static": [(4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)],
                "dynamic": [],
            },
            "walls": {"init": [], "to_distribute": [], "to_vibrate": []},
            "floors": {
                "to_distribute": [
                    (1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3),
                    (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5),
                    (4, 1), (5, 1), (6, 1), (4, 2), (5, 2), (6, 2),
                    (4, 5), (5, 5), (6, 5), (4, 6), (5, 6), (6, 6),
                ],
                "to_vibrate": [],
                "to_level": [],
                "to_cover": [],
            },
            "regions": {
                "TOP_LEFT": [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)],
                "TOP_RIGHT": [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
                "BOTTOM_LEFT": [(4, 1), (5, 1), (6, 1), (4, 2), (5, 2), (6, 2)],
                "BOTTOM_RIGHT": [(4, 5), (5, 5), (6, 5), (4, 6), (5, 6), (6, 6)],
            },
            "task_dependencies": {
                "distribute_floor": [],
                "vibrate_floor": ["distribute_floor"],
                "level_floor": ["vibrate_floor"],
                "cover_surface": ["level_floor"],
            },
        },
    },
}

_WORKFLOW_TASK_CELLS = [
    (1, 2), (1, 3), (2, 2), (2, 3),
    (3, 1), (4, 1), (3, 2), (4, 2),
    (4, 4), (4, 5), (5, 4), (5, 5),
]
_WORKFLOW_REGIONS = {
    "TOP_LEFT": [(1, 2), (1, 3), (2, 2), (2, 3)],
    "BOTTOM_LEFT": [(3, 1), (4, 1), (3, 2), (4, 2)],
    "BOTTOM_RIGHT": [(4, 4), (4, 5), (5, 4), (5, 5)],
}
_WORKFLOW_OBSTACLES = [(3, 3), (3, 4), (5, 2), (5, 3)]


BENCHMARK_SPECS.update(
    {
        "linear6": {
            "meta": {
                "level": "Linear-6",
                "description": "6 robots and a six-node 1-2-3-4-5-6 workflow",
                "complexity": "workflow-depth",
                "expected_steps": "~180-500",
            },
            "config": {
                "grid_width": 7,
                "grid_height": 7,
                "num_robots": 6,
                "robots": {
                    "role_1": {"num": 1, "tasks": ["l1"], "init_positions": [(0, 0)]},
                    "role_2": {"num": 1, "tasks": ["l2"], "init_positions": [(0, 6)]},
                    "role_3": {"num": 1, "tasks": ["l3"], "init_positions": [(1, 0)]},
                    "role_4": {"num": 1, "tasks": ["l4"], "init_positions": [(1, 6)]},
                    "role_5": {"num": 1, "tasks": ["l5"], "init_positions": [(2, 0)]},
                    "role_6": {"num": 1, "tasks": ["l6"], "init_positions": [(2, 6)]},
                },
                "obstacles": {"static": _WORKFLOW_OBSTACLES, "dynamic": []},
                "task_cells": _WORKFLOW_TASK_CELLS,
                "regions": _WORKFLOW_REGIONS,
                "workflow": {
                    "nodes": [
                        {"id": "l1", "role": "role_1", "event": "linear_stage_1"},
                        {"id": "l2", "role": "role_2", "event": "linear_stage_2"},
                        {"id": "l3", "role": "role_3", "event": "linear_stage_3"},
                        {"id": "l4", "role": "role_4", "event": "linear_stage_4"},
                        {"id": "l5", "role": "role_5", "event": "linear_stage_5"},
                        {"id": "l6", "role": "role_6", "event": "linear_stage_6"},
                    ],
                    "edges": [
                        ["l1", "l2"], ["l2", "l3"], ["l3", "l4"],
                        ["l4", "l5"], ["l5", "l6"],
                    ],
                },
                "task_dependencies": {
                    "l1": [], "l2": ["l1"], "l3": ["l2"],
                    "l4": ["l3"], "l5": ["l4"], "l6": ["l5"],
                },
            },
        },
        "cross5": {
            "meta": {
                "level": "Cross-5",
                "description": "5 robots and a branched/re-entrant 1-2-3-1-2-4-5 workflow",
                "complexity": "workflow-structure",
                "expected_steps": "~200-600",
            },
            "config": {
                "grid_width": 7,
                "grid_height": 7,
                "num_robots": 5,
                "robots": {
                    "role_1": {
                        "num": 1,
                        "tasks": ["n1a", "n1b"],
                        "init_positions": [(0, 0)],
                    },
                    "role_2": {
                        "num": 1,
                        "tasks": ["n2a", "n2b"],
                        "init_positions": [(0, 6)],
                    },
                    "role_3": {"num": 1, "tasks": ["n3"], "init_positions": [(1, 0)]},
                    "role_4": {"num": 1, "tasks": ["n4"], "init_positions": [(1, 6)]},
                    "role_5": {"num": 1, "tasks": ["n5"], "init_positions": [(2, 0)]},
                },
                "obstacles": {"static": _WORKFLOW_OBSTACLES, "dynamic": []},
                "task_cells": _WORKFLOW_TASK_CELLS,
                "regions": _WORKFLOW_REGIONS,
                "workflow": {
                    "nodes": [
                        {"id": "n1a", "role": "role_1", "event": "cross_stage_1a"},
                        {"id": "n2a", "role": "role_2", "event": "cross_stage_2a"},
                        {"id": "n3", "role": "role_3", "event": "cross_stage_3"},
                        {"id": "n1b", "role": "role_1", "event": "cross_stage_1b"},
                        {"id": "n2b", "role": "role_2", "event": "cross_stage_2b"},
                        {"id": "n4", "role": "role_4", "event": "cross_stage_4"},
                        {"id": "n5", "role": "role_5", "event": "cross_stage_5"},
                    ],
                    "edges": [
                        ["n1a", "n2a"],
                        ["n1a", "n3"],
                        ["n2a", "n1b"],
                        ["n3", "n1b"],
                        ["n1b", "n2b"],
                        ["n2b", "n4"],
                        ["n4", "n5"],
                    ],
                },
                "task_dependencies": {
                    "n1a": [],
                    "n2a": ["n1a"],
                    "n3": ["n1a"],
                    "n1b": ["n2a", "n3"],
                    "n2b": ["n1b"],
                    "n4": ["n2b"],
                    "n5": ["n4"],
                },
            },
        },
    }
)


def get_spec_data(name: str) -> dict:
    key = name.lower()
    if key not in BENCHMARK_SPECS:
        raise KeyError(f"Unknown benchmark split: {name}")
    return deepcopy(BENCHMARK_SPECS[key])
