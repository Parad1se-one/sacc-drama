from __future__ import annotations

import argparse
import json
import random

from .benchmark import export_manifest, get_benchmark, list_benchmarks, make_env


def cmd_list(_args):
    for name in list_benchmarks():
        spec = get_benchmark(name)
        print(
            f"{name}: grid={spec.grid_shape[0]}x{spec.grid_shape[1]}, "
            f"agents={spec.num_agents}, tasks={spec.task_cell_count}, "
            f"obstacles={spec.obstacle_count}, regions={spec.region_count}"
        )


def cmd_describe(args):
    spec = get_benchmark(args.name)
    print(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False))


def cmd_sample(args):
    rng = random.Random(args.seed)
    env = make_env(args.name, seed=args.seed)
    for step in range(args.steps):
        actions = []
        for agent_idx in range(env.num_agents):
            valid = [index for index, enabled in enumerate(env.get_action_mask(agent_idx)) if enabled]
            actions.append(rng.choice(valid))
        rewards, infos = env.step(actions)
        print(
            json.dumps(
                {
                    "step": step + 1,
                    "actions": actions,
                    "rewards": rewards,
                    "completion_ratio": env.get_completion_ratio(),
                    "done": env.is_all_done(),
                    "infos": infos,
                },
                ensure_ascii=False,
            )
        )
        if env.is_all_done():
            break
    print(env.render(mode="text"))


def cmd_export(args):
    path = export_manifest(args.output)
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRAMA benchmark registry and utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List all benchmark splits")
    list_parser.set_defaults(func=cmd_list)

    describe_parser = subparsers.add_parser("describe", help="Describe one benchmark split")
    describe_parser.add_argument("name", choices=list_benchmarks())
    describe_parser.set_defaults(func=cmd_describe)

    sample_parser = subparsers.add_parser("sample", help="Run a short random rollout")
    sample_parser.add_argument("name", choices=list_benchmarks())
    sample_parser.add_argument("--steps", type=int, default=5)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.set_defaults(func=cmd_sample)

    export_parser = subparsers.add_parser("export", help="Export all specs as JSON manifest")
    export_parser.add_argument("--output", default="DRAMA/benchmark_manifest.json")
    export_parser.set_defaults(func=cmd_export)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
