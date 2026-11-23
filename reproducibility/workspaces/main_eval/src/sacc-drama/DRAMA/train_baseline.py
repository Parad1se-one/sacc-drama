#!/usr/bin/env python3
"""Plain-baseline training entrypoint for DRAMA without Reward Machine extras."""

import argparse
import time

import torch

from configs.difficulty_loader import load_config
from DRAMA.starts import default_min_pairwise_manhattan, sample_start_positions
from envs.grid_env import GridAreaEnv
from reward_machines.reward_machine_enhanced import EnhancedRewardMachine
from utils.utils import set_seed


BASELINE_ALGOS = {"qmix", "vdn", "qtran", "ippo", "mappo", "happo", "happo_rm"}


def resolved_checkpoint_meta(args):
    """Record the resolved protocol without changing the optimizer path."""

    return {
        "difficulty": args.difficulty,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "gamma": args.gamma,
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": args.mini_batch_size,
        "clip_eps": args.clip_eps,
        "lam": args.lam,
        "grad_clip": args.grad_clip,
        "value_noise_std": args.value_noise_std,
        "entropy_coef": args.entropy_coef,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "eval_policy_mode": args.eval_policy_mode,
        "eval_temperature": args.eval_temperature,
        "eval_seed": args.eval_seed if args.eval_seed is not None else args.seed,
        "training_start_protocol": args.train_start_protocol,
        "selection_start_protocol": (
            "independent_start_set" if args.validation_start_set else "legacy_repeated_benchmark_fixed"
        ),
        "validation_start_set": args.validation_start_set,
    }



class RandomStartGridAreaEnv(GridAreaEnv):
    """Use the SACC start distribution without changing baseline updates."""

    def __init__(self, config, *, difficulty: str, base_seed: int):
        self._start_difficulty = difficulty
        self._start_base_seed = int(base_seed)
        self._start_reset_count = 0
        super().__init__(config=config)

    def reset(self, config=None, *, seed=None):
        resolved_seed = self._start_base_seed + self._start_reset_count if seed is None else int(seed)
        super().reset(config=config, seed=None)
        positions = sample_start_positions(
            config or self._config,
            seed=resolved_seed,
            exclude_task_cells=True,
            min_pairwise_manhattan=default_min_pairwise_manhattan(self._start_difficulty),
        )
        for robot, position in zip(self.robots, positions, strict=True):
            robot["pos"] = position
        self._update_passability()
        self._start_reset_count += 1
        return self._get_state()


def make_env(difficulty: str, *, train_start_protocol: str = "fixed", seed: int = 42):
    config, meta = load_config(difficulty)
    env = (
        RandomStartGridAreaEnv(config, difficulty=difficulty, base_seed=seed)
        if train_start_protocol == "random"
        else GridAreaEnv(config=config)
    )
    return env, config, meta



def build_reward_machine(config):
    return EnhancedRewardMachine(
        config["task_dependencies"],
        num_agents=config["num_robots"],
        regions=config.get("regions"),
    )



def train_qmix(env, args):
    from marl.algos.qmix import qmix

    checkpoint_meta = resolved_checkpoint_meta(args)
    qmix(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=64,
        buffer_capacity=args.buffer_size,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=args.eps_decay,
        lr=args.lr,
        gamma=args.gamma,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        state_mode=args.state_mode or "legacy",
        use_action_mask=not args.disable_action_mask if args.disable_action_mask is not None else True,
        use_double_q=False,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=checkpoint_meta,
    )



def train_vdn(env, args):
    from marl.algos.vdn_baseline import vdn_baseline

    vdn_baseline(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=64,
        buffer_capacity=args.buffer_size,
        lr=args.lr,
        gamma=args.gamma,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        eps_decay=args.eps_decay,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
    )



def train_qtran(env, args):
    from marl.algos.qtran_baseline import qtran_baseline

    qtran_baseline(
        env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        lr=args.lr,
        gamma=args.gamma,
        eps_decay=args.eps_decay,
        lambda_opt=args.lambda_opt,
        lambda_nopt=args.lambda_nopt,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
    )



def train_ippo(env, args):
    from marl.algos.ippo_baseline import ippo_baseline

    ippo_baseline(
        env,
        episodes=args.episodes,
        rollout_len=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        grad_clip=args.grad_clip,
        value_noise_std=args.value_noise_std,
        entropy_coef=args.entropy_coef,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_rate,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
    )



def train_mappo(env, args):
    from marl.algos.mappo_baseline import mappo_baseline

    mappo_baseline(
        env,
        episodes=args.episodes,
        rollout_len=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        grad_clip=args.grad_clip,
        value_noise_std=args.value_noise_std,
        entropy_coef=args.entropy_coef,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
    )



def train_happo(env, args):
    from marl.algos.happo_baseline import happo_baseline

    happo_baseline(
        env,
        episodes=args.episodes,
        rollout_len=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        grad_clip=args.grad_clip,
        value_noise_std=args.value_noise_std,
        entropy_coef=args.entropy_coef,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
        use_action_mask=not args.disable_action_mask if args.disable_action_mask is not None else True,
    )



def train_happo_rm(env, args):
    from marl.algos.happo_baseline import happo_rm

    rm = build_reward_machine(env._config)
    happo_rm(
        env,
        rm,
        episodes=args.episodes,
        rollout_len=args.max_steps,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        grad_clip=args.grad_clip,
        value_noise_std=args.value_noise_std,
        entropy_coef=args.entropy_coef,
        log_dir="runs",
        exp_name=args.exp_name,
        save_interval=args.save_interval,
        difficulty=args.difficulty,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_meta=resolved_checkpoint_meta(args),
        variant="rm_happo",
        state_mode=args.state_mode or "rmix_v2",
        reward_profile="stabilized",
        use_action_mask=not args.disable_action_mask if args.disable_action_mask is not None else True,
    )


TRAIN_FN = {
    "qmix": train_qmix,
    "vdn": train_vdn,
    "qtran": train_qtran,
    "ippo": train_ippo,
    "mappo": train_mappo,
    "happo": train_happo,
    "happo_rm": train_happo_rm,
}


def resolve_difficulty_defaults(args):
    """Fill omitted budgets without overriding values frozen by a run plan."""

    if args.episodes is None:
        args.episodes = 10000 if args.difficulty == "hard" else 3000
    if args.eps_decay is None:
        args.eps_decay = 200000 if args.difficulty == "hard" else 50000
    if args.max_steps is None:
        args.max_steps = 500 if args.difficulty == "hard" else 300
    return args



def main():
    parser = argparse.ArgumentParser(description="DRAMA plain-baseline training")
    parser.add_argument("--algo", type=str, required=True, choices=sorted(BASELINE_ALGOS))
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-start-protocol", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--validation-start-set", type=str, default=None)
    parser.add_argument("--eval-policy-mode", choices=["greedy", "tempered"], default="greedy")
    parser.add_argument("--eval-temperature", type=float, default=0.35)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eps-decay", type=float, default=None)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--state-mode", type=str, default=None, choices=["legacy", "rich", "rmix_v2"])
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--disable-action-mask", action="store_true", default=None)

    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=None)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--value-noise-std", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-rate", type=float, default=0.998)
    parser.add_argument("--lambda-opt", type=float, default=0.1)
    parser.add_argument("--lambda-nopt", type=float, default=0.1)
    args = parser.parse_args()

    # Difficulty-specific defaults apply only when the caller omits a value.
    # Earlier code rewrote an explicit Hard ``--episodes 3000`` request to
    # 10000, making a frozen plan execute a different budget from the one it
    # recorded.  Rebuttal experiments require explicit CLI values to be
    # authoritative.
    resolve_difficulty_defaults(args)

    if args.exp_name is None:
        args.exp_name = f"{args.algo}_{args.difficulty}_seed{args.seed}"

    if args.mini_batch_size is None:
        if args.algo in {"ippo", "mappo", "happo", "happo_rm"}:
            args.mini_batch_size = args.max_steps
        else:
            args.mini_batch_size = args.batch_size

    if args.algo == "ippo" and parser.get_default("ppo_epochs") == args.ppo_epochs:
        args.ppo_epochs = 4
    if args.algo == "qmix":
        args.state_mode = args.state_mode or "legacy"
    if args.algo == "happo_rm":
        args.state_mode = args.state_mode or "rmix_v2"

    set_seed(args.seed)

    print("=" * 60)
    print("  DRAMA Plain Baseline Training")
    print(f"  Algorithm:  {args.algo}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Seed:       {args.seed}")
    print(f"  Exp Name:   {args.exp_name}")
    print(f"  Device:     {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if args.state_mode is not None:
        print(f"  State Mode: {args.state_mode}")
    print("=" * 60)

    env, config, meta = make_env(
        args.difficulty, train_start_protocol=args.train_start_protocol, seed=args.seed
    )
    print(
        f"  Grid: {config['grid_width']}x{config['grid_height']}, "
        f"Units: {meta['total_grids']}, Agents: {config['num_robots']}"
    )
    print("-" * 60)

    start = time.time()
    TRAIN_FN[args.algo](env, args)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
