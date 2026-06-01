"""Difficulty loader backed by the standalone DRAMA benchmark package.

Public benchmark splits in this submission are:
- ``easy``
- ``hard``
"""

from DRAMA import get_benchmark, get_config, get_metadata, list_benchmarks

DIFFICULTY_CONFIGS = {name: name for name in list_benchmarks()}


def load_config(difficulty: str):
    difficulty = difficulty.lower()
    if difficulty not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Unknown difficulty: {difficulty}, choices: {list(DIFFICULTY_CONFIGS.keys())}")
    return get_config(difficulty), get_metadata(difficulty)


def get_all_difficulties():
    info = []
    for difficulty in DIFFICULTY_CONFIGS.keys():
        benchmark = get_benchmark(difficulty)
        meta = benchmark.meta
        info.append(
            {
                "level": difficulty,
                "description": meta["description"],
                "total_grids": benchmark.task_cell_count,
                "expected_steps": meta["expected_steps"],
                "complexity": meta["complexity"],
            }
        )
    return info


def print_difficulty_info():
    print("\n" + "=" * 60)
    print("  DRAMA benchmark splits")
    print("=" * 60)
    for info in get_all_difficulties():
        print(f"\n[{{info['level'].upper()}}]")
        print(f"  description: {{info['description']}}")
        print(f"  task cells:  {{info['total_grids']}}")
        print(f"  steps:       {{info['expected_steps']}}")
        print(f"  complexity:  {{info['complexity']}}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_difficulty_info()
