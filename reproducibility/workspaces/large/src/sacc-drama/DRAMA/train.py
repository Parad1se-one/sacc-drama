#!/usr/bin/env python3
"""
DRAMA 统一训练入口

用法:
    python train.py --algo rmix --difficulty easy --seed 42
    python train.py --algo qmix --difficulty hard --episodes 5000
    python train.py --algo mappo --difficulty easy --seed 123
    python train.py --algo vdn --difficulty hard
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from configs.difficulty_loader import load_config
from envs.grid_env import GridAreaEnv
from reward_machines.reward_machine_enhanced import EnhancedRewardMachine
from utils.utils import set_seed

ALGOS_WITH_RM = {"rmix", "vdn", "mappo", "ippo"}
ALGOS_WITHOUT_RM = {"qmix"}
ALL_ALGOS = ALGOS_WITH_RM | ALGOS_WITHOUT_RM


def make_env_and_rm(difficulty: str, algo: str):
    config, meta = load_config(difficulty)
    env = GridAreaEnv(config=config)
    rm = None
    if algo in ALGOS_WITH_RM:
        rm = EnhancedRewardMachine(
            config["task_dependencies"],
            num_agents=config["num_robots"],
            regions=config.get("regions"),
        )
    return env, rm, config, meta


def train_rmix(env, rm, args):
    from marl.algos.qmix import rmix
    checkpoint_meta = {
        "difficulty": args.difficulty,
        "seed": args.seed,
    }
    rmix(env, rm,
         episodes=args.episodes, max_steps=args.max_steps,
         batch_size=64, buffer_capacity=args.buffer_size,
         eps_start=1.0, eps_end=0.05, eps_decay=args.eps_decay,
         lr=args.lr, gamma=args.gamma,
         log_dir="runs", exp_name=args.exp_name,
         save_interval=args.save_interval,
         variant=args.variant,
         state_mode=args.state_mode,
         use_action_mask=not args.disable_action_mask if args.disable_action_mask is not None else None,
         use_double_q=args.double_q if args.double_q is not None else None,
         eval_interval=args.eval_interval,
         eval_episodes=args.eval_episodes,
         checkpoint_meta=checkpoint_meta)


def train_qmix(env, _rm, args):
    from marl.algos.qmix import qmix
    checkpoint_meta = {
        "difficulty": args.difficulty,
        "seed": args.seed,
    }
    qmix(env,
         episodes=args.episodes, max_steps=args.max_steps,
         batch_size=64, buffer_capacity=args.buffer_size,
         eps_start=1.0, eps_end=0.05, eps_decay=args.eps_decay,
         lr=args.lr, gamma=args.gamma,
         log_dir="runs", exp_name=args.exp_name,
         save_interval=args.save_interval,
         state_mode=args.state_mode or "legacy",
         use_action_mask=not args.disable_action_mask if args.disable_action_mask is not None else True,
         use_double_q=args.double_q if args.double_q is not None else False,
         eval_interval=args.eval_interval,
         eval_episodes=args.eval_episodes,
         checkpoint_meta=checkpoint_meta)


def train_vdn(env, rm, args):
    from marl.algos.dqn_vdn import dqn_vdn
    dqn_vdn(env, rm,
            episodes=args.episodes, max_steps=args.max_steps,
            batch_size=64, buffer_capacity=args.buffer_size,
            lr=args.lr, gamma=args.gamma,
            log_dir="runs", exp_name=args.exp_name,
            save_interval=args.save_interval)


def train_mappo(env, rm, args):
    from marl.algos.mappo import mappo
    mappo(env, rm,
          episodes=args.episodes,
          rollout_len=args.max_steps,
          lr=args.lr, gamma=args.gamma,
          log_dir="runs", exp_name=args.exp_name,
          save_interval=args.save_interval)


def train_ippo(env, rm, args):
    from marl.algos.ippo import ippo
    ippo(env, rm,
         episodes=args.episodes,
         rollout_len=args.max_steps,
         lr=args.lr, gamma=args.gamma,
         log_dir="runs", exp_name=args.exp_name,
         save_interval=args.save_interval)


TRAIN_FN = {
    "rmix": train_rmix,
    "qmix": train_qmix,
    "vdn": train_vdn,
    "mappo": train_mappo,
    "ippo": train_ippo,
}


def main():
    parser = argparse.ArgumentParser(description="DRAMA 统一训练入口")
    parser.add_argument("--algo", type=str, required=True, choices=sorted(ALL_ALGOS))
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--eps-decay", type=float, default=50000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None, choices=["legacy", "rmix_v2", "rmix_v2_medium"])
    parser.add_argument("--state-mode", type=str, default=None, choices=["legacy", "rich", "rmix_v2"])
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--disable-action-mask", action="store_true", default=None)
    parser.add_argument("--double-q", action="store_true", default=None)
    args = parser.parse_args()

    if args.difficulty == "hard" and args.episodes == 3000:
        args.episodes = 10000
    if args.difficulty == "hard" and args.eps_decay == 50000:
        args.eps_decay = 200000
    if args.difficulty == "hard" and args.max_steps == 300:
        args.max_steps = 500

    if args.exp_name is None:
        args.exp_name = f"{args.algo}_{args.difficulty}_seed{args.seed}"

    if args.algo == "rmix":
        if args.variant is None:
            args.variant = "rmix_v2"
        if args.state_mode is None:
            args.state_mode = "legacy" if args.variant == "legacy" else "rmix_v2"
    elif args.algo == "qmix":
        args.variant = None
        if args.state_mode is None:
            args.state_mode = "legacy"

    set_seed(args.seed)

    print("=" * 60)
    print(f"  DRAMA Training")
    print(f"  Algorithm:  {args.algo}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Seed:       {args.seed}")
    print(f"  Exp Name:   {args.exp_name}")
    if args.algo == "rmix":
        print(f"  Variant:    {args.variant}")
    print(f"  State Mode: {args.state_mode}")
    print(f"  Device:     {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)

    env, rm, config, meta = make_env_and_rm(args.difficulty, args.algo)
    print(f"  Grid: {config['grid_width']}x{config['grid_height']}, "
          f"Units: {meta['total_grids']}, Agents: {config['num_robots']}")
    print("-" * 60)

    start = time.time()
    TRAIN_FN[args.algo](env, rm, args)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
