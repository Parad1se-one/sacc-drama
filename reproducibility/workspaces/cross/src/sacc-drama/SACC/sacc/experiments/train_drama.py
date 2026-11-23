"""CLI entrypoint for SACC training on DRAMA."""

from __future__ import annotations

import argparse
import json

from sacc.algorithms.sacc import SACCTrainer, TrainConfig
from sacc.experiments.common import resolve_max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SACC on the DRAMA benchmark")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "hard", "large", "linear6", "cross5"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--prev-ratio-clip-min", type=float, default=None)
    parser.add_argument("--prev-ratio-clip-max", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-final", type=float, default=None)
    parser.add_argument("--entropy-anneal-fraction", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=100_042)
    parser.add_argument("--validation-start-set", type=str, default=None)
    parser.add_argument("--train-start-protocol", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--disable-actor-rm", action="store_true")
    parser.add_argument("--disable-critic-rm", action="store_true")
    parser.add_argument("--rm-mode", choices=["local", "team"], default="local")
    parser.add_argument("--env-reward-scale", type=float, default=1.0)
    parser.add_argument("--rm-reward-scale", type=float, default=1.0)
    parser.add_argument("--team-progress-bonus-coeff", type=float, default=0.25)
    parser.add_argument("--team-progress-bonus-final", type=float, default=None)
    parser.add_argument("--team-progress-anneal-fraction", type=float, default=1.0)
    parser.add_argument("--train-temperature-start", type=float, default=1.0)
    parser.add_argument("--train-temperature-final", type=float, default=1.0)
    parser.add_argument("--train-temperature-anneal-fraction", type=float, default=1.0)
    parser.add_argument("--eval-policy-mode", default="greedy", choices=["greedy", "tempered"])
    parser.add_argument("--eval-temperature", type=float, default=0.35)
    parser.add_argument("--early-stop-patience-evals", type=int, default=0)
    parser.add_argument("--early-stop-min-episode", type=int, default=0)
    parser.add_argument("--reward-stage-event", type=float, default=1.0)
    parser.add_argument("--reward-stage-done", type=float, default=1.5)
    parser.add_argument("--reward-all-done", type=float, default=4.0)
    parser.add_argument("--penalty-idle", type=float, default=-0.01)
    parser.add_argument("--penalty-no-effect", type=float, default=-0.10)
    parser.add_argument("--penalty-blocked", type=float, default=-0.05)
    parser.add_argument("--distance-shaping-coeff", type=float, default=0.03)
    parser.add_argument("--distance-shaping-final", type=float, default=None)
    parser.add_argument("--distance-shaping-anneal-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--exp-name", type=str, default="sacc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        difficulty=args.difficulty,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=resolve_max_steps(args.difficulty, args.max_steps),
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        prev_ratio_clip_min=args.prev_ratio_clip_min,
        prev_ratio_clip_max=args.prev_ratio_clip_max,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        lam=args.lam,
        grad_clip=args.grad_clip,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_fraction=args.entropy_anneal_fraction,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        validation_start_set=args.validation_start_set,
        train_start_protocol=args.train_start_protocol,
        save_interval=args.save_interval,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        disable_actor_rm=args.disable_actor_rm,
        disable_critic_rm=args.disable_critic_rm,
        rm_mode=args.rm_mode,
        env_reward_scale=args.env_reward_scale,
        rm_reward_scale=args.rm_reward_scale,
        team_progress_bonus_coeff=args.team_progress_bonus_coeff,
        team_progress_bonus_final=args.team_progress_bonus_final,
        team_progress_anneal_fraction=args.team_progress_anneal_fraction,
        train_temperature_start=args.train_temperature_start,
        train_temperature_final=args.train_temperature_final,
        train_temperature_anneal_fraction=args.train_temperature_anneal_fraction,
        distance_shaping_final=args.distance_shaping_final,
        distance_shaping_anneal_fraction=args.distance_shaping_anneal_fraction,
        eval_policy_mode=args.eval_policy_mode,
        eval_temperature=args.eval_temperature,
        early_stop_patience_evals=args.early_stop_patience_evals,
        early_stop_min_episode=args.early_stop_min_episode,
        device=args.device,
        exp_name=args.exp_name,
        rm_kwargs={
            "reward_stage_event": args.reward_stage_event,
            "reward_stage_done": args.reward_stage_done,
            "reward_all_done": args.reward_all_done,
            "penalty_idle": args.penalty_idle,
            "penalty_no_effect": args.penalty_no_effect,
            "penalty_blocked": args.penalty_blocked,
            "distance_shaping_coeff": args.distance_shaping_coeff,
        },
    )
    run_dir = SACCTrainer(config).train()
    print(json.dumps({"run_dir": str(run_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
