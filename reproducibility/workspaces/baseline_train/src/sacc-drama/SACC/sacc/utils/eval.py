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

from DRAMA import generate_disjoint_start_sets, load_start_set

SUCCESS_EPS = 1e-6


def _episode_summary(
    ep_idx: int,
    objective_return: float,
    environment_return: float,
    completion: float,
    steps: int,
    start_positions,
    diagnostics: dict | None = None,
) -> dict:
    success = 1.0 if completion >= 1.0 - SUCCESS_EPS else 0.0
    payload = {
        "episode": int(ep_idx),
        "objective_return": float(objective_return),
        "environment_return": float(environment_return),
        # Backward-compatible alias. Cross-method tables must not use this field.
        "reward": float(objective_return),
        "completion_ratio": float(completion),
        "steps": int(steps),
        "success": success,
        "start_positions": [list(map(int, pos)) for pos in start_positions],
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def _aggregate_episode_summaries(episodes: list[dict]) -> dict:
    if not episodes:
        return {
            "mean_completion_ratio": 0.0,
            "std_completion_ratio": 0.0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
            "std_steps": 0.0,
            "mean_objective_return": 0.0,
            "std_objective_return": 0.0,
            "mean_environment_return": 0.0,
            "std_environment_return": 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
        }
    completions = [ep["completion_ratio"] for ep in episodes]
    objective_returns = [ep["objective_return"] for ep in episodes]
    environment_returns = [ep["environment_return"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    successes = [ep["success"] for ep in episodes]
    return {
        "mean_completion_ratio": float(mean(completions)),
        "std_completion_ratio": float(pstdev(completions) if len(completions) > 1 else 0.0),
        "success_rate": float(mean(successes)),
        "mean_steps": float(mean(steps)),
        "std_steps": float(pstdev(steps) if len(steps) > 1 else 0.0),
        "mean_objective_return": float(mean(objective_returns)),
        "std_objective_return": float(pstdev(objective_returns) if len(objective_returns) > 1 else 0.0),
        "mean_environment_return": float(mean(environment_returns)),
        "std_environment_return": float(
            pstdev(environment_returns) if len(environment_returns) > 1 else 0.0
        ),
        # Backward-compatible aliases for trainer dashboards only.
        "mean_reward": float(mean(objective_returns)),
        "std_reward": float(pstdev(objective_returns) if len(objective_returns) > 1 else 0.0),
    }


def _aggregate_episode_diagnostics(episodes: list[dict], *, num_agents: int) -> dict:
    if not episodes:
        return {}
    stay_fractions = []
    blocked_fractions = []
    missed_perform_rates = []
    max_no_progress_streaks = []
    progress_steps = []
    terminal_keys = ("to_distribute", "to_vibrate", "to_level", "to_cover", "completed")
    terminal_values = {key: [] for key in terminal_keys}
    for episode in episodes:
        diagnostics = episode["diagnostics"]
        denominator = max(int(episode["steps"]) * int(num_agents), 1)
        stay_fractions.append(float(diagnostics["status_counts"].get("stayed", 0)) / denominator)
        blocked_fractions.append(float(diagnostics["status_counts"].get("blocked", 0)) / denominator)
        opportunities = int(diagnostics["perform_opportunities"])
        missed_perform_rates.append(
            float(diagnostics["missed_perform_opportunities"]) / opportunities
            if opportunities
            else 0.0
        )
        max_no_progress_streaks.append(float(diagnostics["max_no_progress_streak"]))
        progress_steps.append(float(diagnostics["progress_steps"]))
        for key in terminal_keys:
            terminal_values[key].append(float(diagnostics["terminal_stage_counts"][key]))
    return {
        "mean_stay_fraction": float(mean(stay_fractions)),
        "mean_blocked_fraction": float(mean(blocked_fractions)),
        "mean_missed_perform_rate": float(mean(missed_perform_rates)),
        "mean_max_no_progress_streak": float(mean(max_no_progress_streaks)),
        "mean_progress_steps": float(mean(progress_steps)),
        "mean_terminal_stage_counts": {
            key: float(mean(values)) for key, values in terminal_values.items()
        },
    }


def _resolve_evaluation_starts(
    *,
    difficulty: str,
    episodes: int | None,
    seed: int,
    start_protocol: str,
    start_set_path: str | None,
) -> tuple[list[tuple[tuple[int, int], ...] | None], dict]:
    if start_set_path:
        payload = load_start_set(start_set_path, expected_difficulty=difficulty)
        starts = list(payload["starts"])
        if episodes is not None and int(episodes) != len(starts):
            raise ValueError(
                f"Requested {episodes} episodes but start set {payload['start_set_id']} contains {len(starts)}"
            )
        sampling = str(payload["generation"].get("sampling", ""))
        versioned_protocol = (
            "versioned_fixed" if sampling == "benchmark_fixed_role_order" else "versioned_random"
        )
        return starts, {
            "start_protocol": versioned_protocol,
            "start_set_path": str(Path(start_set_path).resolve()),
            "start_set_id": payload["start_set_id"],
            "start_split": payload["split"],
            "unique_start_count": len(set(starts)),
        }

    count = 20 if episodes is None else int(episodes)
    if count <= 0:
        raise ValueError("Evaluation episodes must be positive")
    protocol = start_protocol.lower()
    if protocol == "fixed":
        return [None] * count, {
            "start_protocol": "fixed",
            "start_set_path": None,
            "start_set_id": None,
            "start_split": None,
            "unique_start_count": 1,
        }
    if protocol == "random":
        payload = generate_disjoint_start_sets(
            difficulty,
            {"ad_hoc_evaluation": count},
            seed=seed,
        )["ad_hoc_evaluation"]
        return list(payload["starts"]), {
            "start_protocol": "ad_hoc_random",
            "start_set_path": None,
            "start_set_id": payload["start_set_id"],
            "start_split": payload["split"],
            "unique_start_count": count,
        }
    raise ValueError(f"Unsupported evaluation start protocol: {start_protocol}")


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
    episodes: int | None,
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
    start_protocol: str = "fixed",
    start_set_path: str | None = None,
    actor_observation_mode: str = "sacc",
    algo: str = "sacc",
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
    episode_starts, start_metadata = _resolve_evaluation_starts(
        difficulty=difficulty,
        episodes=episodes,
        seed=seed,
        start_protocol=start_protocol,
        start_set_path=start_set_path,
    )
    results: list[dict] = []
    actor_observation_mode = actor_observation_mode.lower()
    if actor_observation_mode not in {"sacc", "base"}:
        raise ValueError(f"Unsupported actor observation mode: {actor_observation_mode}")

    for ep, start_positions in enumerate(episode_starts):
        env = DramaRMEnv(
            difficulty=difficulty,
            seed=seed + ep,
            env_reward_scale=env_reward_scale,
            rm_reward_scale=rm_reward_scale,
            team_progress_bonus_coeff=team_progress_bonus_coeff,
            rm_kwargs=rm_kwargs,
        )
        env.reset(seed=seed + ep, start_positions=start_positions)
        objective_return = 0.0
        environment_return = 0.0
        steps = 0
        done = False
        status_counts: dict[str, int] = {}
        action_counts = [0] * env.action_n
        perform_opportunities = 0
        missed_perform_opportunities = 0
        progress_steps = 0
        first_progress_step = None
        no_progress_streak = 0
        max_no_progress_streak = 0

        while not done and steps < max_steps:
            if actor_observation_mode == "base":
                obs_list = env.build_base_obs_list()
            else:
                obs_list = env.build_obs_list(include_rm=not disable_actor_rm)
            action_masks = env.get_action_masks()
            actions = action_selector(obs_list, action_masks)
            for agent_idx, action in enumerate(actions):
                action_counts[int(action)] += 1
                if bool(action_masks[agent_idx][4]):
                    perform_opportunities += 1
                    if int(action) != 4:
                        missed_perform_opportunities += 1
            completion_before = env.get_completion_ratio()
            step_out = env.step(actions)
            objective_return += float(sum(step_out["rewards"]))
            environment_return += float(sum(step_out["env_rewards"]))
            steps += 1
            done = bool(step_out["done"])
            for info in step_out["infos"]:
                status = str(info.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
            if env.get_completion_ratio() > completion_before + SUCCESS_EPS:
                progress_steps += 1
                if first_progress_step is None:
                    first_progress_step = steps
                no_progress_streak = 0
            else:
                no_progress_streak += 1
                max_no_progress_streak = max(max_no_progress_streak, no_progress_streak)

        terminal_stage_counts = {
            "to_distribute": len(env.env.floors_to_distribute),
            "to_vibrate": len(env.env.floors_to_vibrate),
            "to_level": len(env.env.floors_to_level),
            "to_cover": len(env.env.floors_to_cover),
            "completed": len(env.env.floors_completed),
        }
        diagnostics = {
            "status_counts": status_counts,
            "action_counts": action_counts,
            "perform_opportunities": perform_opportunities,
            "missed_perform_opportunities": missed_perform_opportunities,
            "progress_steps": progress_steps,
            "first_progress_step": first_progress_step,
            "max_no_progress_streak": max_no_progress_streak,
            "terminal_stage_counts": terminal_stage_counts,
        }

        results.append(
            _episode_summary(
                ep + 1,
                objective_return,
                environment_return,
                env.get_completion_ratio(),
                steps,
                env.get_start_positions(),
                diagnostics,
            )
        )

    summary = _aggregate_episode_summaries(results)
    diagnostic_summary = _aggregate_episode_diagnostics(results, num_agents=len(actors))
    payload = {
        "algo": str(algo).lower(),
        "difficulty": difficulty,
        "checkpoint_path": checkpoint_path,
        "max_steps": max_steps,
        "seed": seed,
        "policy_mode": policy_mode,
        "temperature": float(temperature),
        "actor_observation_mode": actor_observation_mode,
        "episodes": results,
        "summary": summary,
        "diagnostic_summary": diagnostic_summary,
        "metadata": metadata or {},
        "evaluation_reward_definitions": {
            "objective_return": "sum of the method-specific training objective rewards across agents",
            "environment_return": "sum of raw DRAMA environment rewards across agents",
        },
        **start_metadata,
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


def load_actor_checkpoint_bundle(checkpoint_path: str | Path, device: torch.device):
    """Load actors from either a SACC or legacy HAPPO/IPPO checkpoint."""

    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dicts = list(payload["actors"])
    if not state_dicts:
        raise ValueError("Checkpoint contains no actor state dictionaries")
    meta = dict(payload.get("meta", {}))

    first_state = state_dicts[0]
    try:
        inferred_obs_dim = int(first_state["policy.0.weight"].shape[1])
        inferred_hidden_dim = int(first_state["policy.0.weight"].shape[0])
        inferred_action_n = int(first_state["policy.4.weight"].shape[0])
    except KeyError as exc:
        raise ValueError(f"Unsupported actor checkpoint architecture; missing {exc.args[0]}") from exc

    obs_dim = int(meta.get("obs_dim", inferred_obs_dim))
    action_n = int(meta.get("action_n", inferred_action_n))
    actor_hidden_dim = int(meta.get("actor_hidden_dim", inferred_hidden_dim))
    if (obs_dim, action_n, actor_hidden_dim) != (
        inferred_obs_dim,
        inferred_action_n,
        inferred_hidden_dim,
    ):
        raise ValueError("Checkpoint actor metadata does not match its tensor shapes")

    num_agents = int(meta.get("num_agents", len(state_dicts)))
    if num_agents != len(state_dicts):
        raise ValueError(
            f"Checkpoint metadata declares {num_agents} agents but contains {len(state_dicts)} actors"
        )
    actors = [
        CategoricalActor(obs_dim, action_n, hidden_dim=actor_hidden_dim).to(device)
        for _ in range(num_agents)
    ]
    for actor, state_dict in zip(actors, state_dicts):
        actor.load_state_dict(state_dict)
        actor.eval()

    meta.setdefault("obs_dim", obs_dim)
    meta.setdefault("action_n", action_n)
    meta.setdefault("actor_hidden_dim", actor_hidden_dim)
    meta.setdefault("num_agents", num_agents)
    return actors, meta


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int | None,
    seed: int,
    difficulty: str | None = None,
    max_steps: int | None = None,
    device: str = "auto",
    json_out: str | None = None,
    policy_mode: str | None = None,
    temperature: float | None = None,
    start_protocol: str = "fixed",
    start_set_path: str | None = None,
    actor_observation_mode: str | None = None,
    algo: str | None = None,
) -> dict:
    resolved_device = resolve_device(device)
    actors, meta = load_actor_checkpoint_bundle(checkpoint_path, resolved_device)
    difficulty = (difficulty or meta.get("difficulty", "easy")).lower()
    policy_mode = policy_mode or meta.get("eval_policy_mode", "greedy")
    temperature = float(temperature if temperature is not None else meta.get("eval_temperature", 0.35))
    probe_env = DramaRMEnv(difficulty=difficulty, seed=seed)
    obs_dim = int(meta["obs_dim"])
    if actor_observation_mode is None:
        if obs_dim == probe_env.base_obs_dim:
            actor_observation_mode = "base"
        elif obs_dim == probe_env.obs_dim:
            actor_observation_mode = "sacc"
        else:
            raise ValueError(
                f"Actor observation dimension {obs_dim} matches neither base "
                f"({probe_env.base_obs_dim}) nor SACC ({probe_env.obs_dim}) observations"
            )
    resolved_algo = str(algo or meta.get("algo") or meta.get("algorithm") or "sacc").lower()
    is_base_policy = actor_observation_mode.lower() == "base"
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
        rm_reward_scale=0.0 if is_base_policy else float(meta.get("rm_reward_scale", 1.0)),
        team_progress_bonus_coeff=(
            0.0 if is_base_policy else float(meta.get("team_progress_bonus_coeff", 0.25))
        ),
        start_protocol=start_protocol,
        start_set_path=start_set_path,
        actor_observation_mode=actor_observation_mode,
        algo=resolved_algo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation for SACC checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--policy-mode", choices=["greedy", "tempered"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--start-protocol", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--start-set", type=str, default=None)
    parser.add_argument("--actor-observation-mode", choices=["sacc", "base"], default=None)
    parser.add_argument("--algo", type=str, default=None)
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
        start_protocol=args.start_protocol,
        start_set_path=args.start_set,
        actor_observation_mode=args.actor_observation_mode,
        algo=args.algo,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
