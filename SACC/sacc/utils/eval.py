"""Evaluation and checkpoint loading for SACC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch

from sacc.algorithms.networks import CategoricalActor, CentralizedCritic
from sacc.envs.drama import DramaRMEnv
from sacc.experiments.common import resolve_device, resolve_max_steps, write_json

SUCCESS_EPS = 1e-6


def _episode_summary(ep_idx: int, reward: float, completion: float, steps: int) -> dict:
    success = 1.0 if completion >= 1.0 - SUCCESS_EPS else 0.0
    return {
        "episode": int(ep_idx),
        "reward": float(reward),
        "completion_ratio": float(completion),
        "steps": int(steps),
        "success": success,
    }


def _aggregate_episode_summaries(episodes: list[dict]) -> dict:
    if not episodes:
        return {
            "mean_completion_ratio": 0.0,
            "std_completion_ratio": 0.0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "std_steps": 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
        }
    completions = [ep["completion_ratio"] for ep in episodes]
    rewards = [ep["reward"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    successes = [ep["success"] for ep in episodes]
    return {
        "mean_completion_ratio": float(mean(completions)),
        "std_completion_ratio": float(pstdev(completions) if len(completions) > 1 else 0.0),
        "success_rate": float(mean(successes)),
        "mean_steps": float(mean(steps)),
        "std_steps": float(pstdev(steps) if len(steps) > 1 else 0.0),
        "mean_reward": float(mean(rewards)),
        "std_reward": float(pstdev(rewards) if len(rewards) > 1 else 0.0),
    }


def _masked_argmax(probs: torch.Tensor, action_mask: torch.Tensor) -> int:
    masked_probs = probs.masked_fill(~action_mask, -1.0)
    if torch.all(~action_mask):
        return int(probs.shape[-1] - 1)
    return int(torch.argmax(masked_probs, dim=-1).item())


@torch.no_grad()
def _masked_tempered_sample(
    probs: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> int:
    if torch.all(~action_mask):
        return int(probs.shape[-1] - 1)
    if temperature <= 1e-6:
        return _masked_argmax(probs, action_mask)

    masked_probs = probs.masked_fill(~action_mask, 0.0)
    logits = torch.log(masked_probs.clamp_min(1e-8)) / float(temperature)
    logits = logits.masked_fill(~action_mask, float("-inf"))
    tempered_probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
    action = torch.multinomial(tempered_probs, num_samples=1, replacement=True, generator=generator)
    return int(action.item())


@torch.no_grad()
def make_actor_selector(
    actors: list[CategoricalActor],
    device: torch.device,
    *,
    policy_mode: str = "greedy",
    temperature: float = 0.35,
    rng_seed: int | None = None,
):
    generator = None
    if policy_mode == "tempered":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(0 if rng_seed is None else rng_seed))

    def _select(obs_list, action_masks):
        actions = []
        for agent_idx, actor in enumerate(actors):
            obs_t = torch.tensor(obs_list[agent_idx], dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_masks[agent_idx], dtype=torch.bool, device=device).unsqueeze(0)
            probs = actor(obs_t)
            if policy_mode == "tempered":
                actions.append(
                    _masked_tempered_sample(
                        probs,
                        mask_t,
                        temperature=temperature,
                        generator=generator,
                    )
                )
            else:
                actions.append(_masked_argmax(probs, mask_t))
        return actions

    return _select


@torch.no_grad()
def evaluate_policy(
    *,
    actors: list[CategoricalActor],
    difficulty: str,
    episodes: int,
    seed: int,
    max_steps: int | None,
    device: torch.device,
    metadata: dict | None = None,
    checkpoint_path: str | None = None,
    json_out: str | None = None,
    rm_kwargs: dict | None = None,
    disable_actor_rm: bool = False,
    disable_critic_rm: bool = False,
    policy_mode: str = "greedy",
    temperature: float = 0.35,
    env_reward_scale: float = 1.0,
    rm_reward_scale: float = 1.0,
    team_progress_bonus_coeff: float = 0.25,
) -> dict:
    difficulty = difficulty.lower()
    max_steps = resolve_max_steps(difficulty, max_steps)
    action_selector = make_actor_selector(
        actors,
        device,
        policy_mode=policy_mode,
        temperature=temperature,
        rng_seed=seed,
    )
    results: list[dict] = []

    for ep in range(episodes):
        env = DramaRMEnv(
            difficulty=difficulty,
            seed=seed + ep,
            env_reward_scale=env_reward_scale,
            rm_reward_scale=rm_reward_scale,
            team_progress_bonus_coeff=team_progress_bonus_coeff,
            rm_kwargs=rm_kwargs,
        )
        env.reset(seed=seed + ep)
        ep_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            obs_list = env.build_obs_list(include_rm=not disable_actor_rm)
            action_masks = env.get_action_masks()
            actions = action_selector(obs_list, action_masks)
            step_out = env.step(actions)
            ep_reward += float(sum(step_out["rewards"]))
            steps += 1
            done = bool(step_out["done"])

        results.append(_episode_summary(ep + 1, ep_reward, env.get_completion_ratio(), steps))

    summary = _aggregate_episode_summaries(results)
    payload = {
        "algo": "sacc",
        "difficulty": difficulty,
        "checkpoint_path": checkpoint_path,
        "max_steps": max_steps,
        "seed": seed,
        "policy_mode": policy_mode,
        "temperature": float(temperature),
        "episodes": results,
        "summary": summary,
        "metadata": metadata or {},
    }
    if json_out:
        write_json(Path(json_out), payload)
    return payload


def load_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    meta = dict(payload["meta"])
    obs_dim = int(meta["obs_dim"])
    action_n = int(meta["action_n"])
    state_dim = int(meta["state_dim"])
    actor_hidden_dim = int(meta.get("actor_hidden_dim", 128))
    critic_hidden_dim = int(meta.get("critic_hidden_dim", 256))
    num_agents = int(meta["num_agents"])

    actors = [CategoricalActor(obs_dim, action_n, hidden_dim=actor_hidden_dim).to(device) for _ in range(num_agents)]
    for actor, state_dict in zip(actors, payload["actors"]):
        actor.load_state_dict(state_dict)
        actor.eval()

    critics = [CentralizedCritic(state_dim, hidden_dim=critic_hidden_dim).to(device) for _ in range(num_agents)]
    for critic, state_dict in zip(critics, payload.get("critics", [])):
        critic.load_state_dict(state_dict)
        critic.eval()

    return actors, critics, meta


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int,
    seed: int,
    difficulty: str | None = None,
    max_steps: int | None = None,
    device: str = "auto",
    json_out: str | None = None,
    policy_mode: str | None = None,
    temperature: float | None = None,
) -> dict:
    resolved_device = resolve_device(device)
    actors, _, meta = load_checkpoint_bundle(checkpoint_path, resolved_device)
    difficulty = (difficulty or meta.get("difficulty", "easy")).lower()
    policy_mode = policy_mode or meta.get("eval_policy_mode", "greedy")
    temperature = float(temperature if temperature is not None else meta.get("eval_temperature", 0.35))
    return evaluate_policy(
        actors=actors,
        difficulty=difficulty,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps or meta.get("max_steps"),
        device=resolved_device,
        metadata=meta,
        checkpoint_path=str(checkpoint_path),
        json_out=json_out,
        rm_kwargs=meta.get("rm_kwargs"),
        disable_actor_rm=bool(meta.get("disable_actor_rm", False)),
        disable_critic_rm=bool(meta.get("disable_critic_rm", False)),
        policy_mode=policy_mode,
        temperature=temperature,
        env_reward_scale=float(meta.get("env_reward_scale", 1.0)),
        rm_reward_scale=float(meta.get("rm_reward_scale", 1.0)),
        team_progress_bonus_coeff=float(meta.get("team_progress_bonus_coeff", 0.25)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation for SACC checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--policy-mode", choices=["greedy", "tempered"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
        device=args.device,
        json_out=args.json_out,
        policy_mode=args.policy_mode,
        temperature=args.temperature,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
