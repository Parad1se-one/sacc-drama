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


def get_spec_data(name: str) -> dict:
    key = name.lower()
    if key not in BENCHMARK_SPECS:
        raise KeyError(f"Unknown benchmark split: {name}")
    return deepcopy(BENCHMARK_SPECS[key])
