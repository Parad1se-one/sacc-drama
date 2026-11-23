import json
from pathlib import Path
from statistics import mean, pstdev

import torch

from configs.difficulty_loader import load_config
from DRAMA.starts import load_start_set
from envs.grid_env import GridAreaEnv
from reward_machines.reward_machine_enhanced import EnhancedRewardMachine
from utils.utils import build_obs_list_for_agents, device, get_action_masks


SUCCESS_EPS = 1e-6


def resolve_max_steps(difficulty, explicit=None):
    if explicit is not None:
        return explicit
    return 500 if difficulty == "hard" else 300


def _episode_summary(ep_idx, reward, completion, steps):
    success = 1.0 if completion >= 1.0 - SUCCESS_EPS else 0.0
    return {
        "episode": int(ep_idx),
        "reward": float(reward),
        "completion_ratio": float(completion),
        "steps": int(steps),
        "success": success,
    }


def _aggregate_episode_summaries(episodes):
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


def save_payload(payload, json_out):
    out_path = Path(json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_reward_machine(config, rm_kwargs=None):
    params = dict(rm_kwargs or {})
    return EnhancedRewardMachine(
        config["task_dependencies"],
        num_agents=config["num_robots"],
        regions=config.get("regions"),
        **params,
    )


@torch.no_grad()
def evaluate_policy(
    *,
    action_selector,
    difficulty,
    episodes,
    seed,
    algo,
    max_steps=None,
    metadata=None,
    json_out=None,
    uses_rm=False,
    rm_kwargs=None,
    start_set_path=None,
):
    config, _ = load_config(difficulty)
    max_steps = resolve_max_steps(difficulty, explicit=max_steps)
    start_set = load_start_set(start_set_path, expected_difficulty=difficulty) if start_set_path else None
    if start_set is not None and episodes != start_set["count"]:
        raise ValueError("Selection episodes must match the frozen start-set count")

    results = []

    for ep in range(episodes):
        env = GridAreaEnv(config=config)
        env.reset(seed=None if start_set is not None else seed + ep)
        if start_set is not None:
            positions = start_set["starts"][ep]
            for robot, position in zip(env.robots, positions, strict=True):
                robot["pos"] = position
            env._update_passability()
        rm = build_reward_machine(config, rm_kwargs=rm_kwargs) if uses_rm else None
        if rm is not None:
            rm.reset(env=env)
        ep_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)
            action_masks = get_action_masks(env)
            actions = action_selector(obs_list, action_masks)
            rewards_env, infos = env.step(actions)

            if rm is not None:
                events = [info.get("status") for info in infos]
                rm_rewards, _, _ = rm.step(events, env=env)
                rewards = [
                    (rewards_env[i] if i < len(rewards_env) else 0.0)
                    + (rm_rewards[i] if i < len(rm_rewards) else 0.0)
                    for i in range(len(actions))
                ]
            else:
                rewards = rewards_env

            ep_reward += float(sum(rewards))
            steps += 1
            done = env.is_all_done()

        completion = env.get_completion_ratio()
        episode_result = _episode_summary(ep + 1, ep_reward, completion, steps)
        if start_set is not None:
            episode_result["start_positions"] = [list(pos) for pos in start_set["starts"][ep]]
        results.append(episode_result)

    summary = _aggregate_episode_summaries(results)
    payload = {
        "algo": algo,
        "difficulty": difficulty,
        "checkpoint_path": None,
        "max_steps": max_steps,
        "seed": seed,
        "episodes": results,
        "summary": summary,
        "metadata": metadata or {},
    }

    if start_set is not None:
        payload["selection_start_set"] = {
            "path": str(start_set_path),
            "start_set_id": start_set["start_set_id"],
        }

    if json_out:
        save_payload(payload, json_out)

    return payload


def make_greedy_value_selector(agents, use_action_mask=True):
    @torch.no_grad()
    def _select(obs_list, action_masks):
        actions = []
        for i, agent in enumerate(agents):
            obs_t = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            q_values = agent.net(obs_t)
            if use_action_mask:
                mask_t = torch.tensor(action_masks[i], dtype=torch.bool, device=device).unsqueeze(0)
                q_values = q_values.masked_fill(~mask_t, -1e9)
            actions.append(int(torch.argmax(q_values, dim=1).item()))
        return actions

    return _select


def make_greedy_actor_selector(actors, use_action_mask=True):
    @torch.no_grad()
    def _select(obs_list, action_masks):
        actions = []
        for i, actor in enumerate(actors):
            obs_t = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            probs = actor(obs_t)
            if use_action_mask:
                mask_t = torch.tensor(action_masks[i], dtype=torch.bool, device=device).unsqueeze(0)
                probs = probs.masked_fill(~mask_t, 0.0)
            actions.append(int(torch.argmax(probs, dim=1).item()))
        return actions

    return _select


def make_actor_selector(
    actors, *, policy_mode="greedy", temperature=0.35, rng_seed=0, use_action_mask=True
):
    if policy_mode == "greedy":
        return make_greedy_actor_selector(actors, use_action_mask=use_action_mask)
    if policy_mode != "tempered" or temperature <= 0:
        raise ValueError("Expected tempered policy with positive temperature")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(rng_seed))

    @torch.no_grad()
    def _select(obs_list, action_masks):
        actions = []
        for i, actor in enumerate(actors):
            obs_t = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
            probs = actor(obs_t)
            mask_t = (
                torch.tensor(action_masks[i], dtype=torch.bool, device=device).unsqueeze(0)
                if use_action_mask else torch.ones_like(probs, dtype=torch.bool)
            )
            if torch.all(~mask_t):
                actions.append(int(probs.shape[-1] - 1))
                continue
            masked_probs = probs.masked_fill(~mask_t, 0.0)
            logits = torch.log(masked_probs.clamp_min(1e-8)) / float(temperature)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            tempered_probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
            action = torch.multinomial(
                tempered_probs, num_samples=1, replacement=True, generator=generator
            )
            actions.append(int(action.item()))
        return actions

    return _select


def log_eval_scalars(writer, summary, step):
    writer.add_scalar("eval/mean_completion_ratio", summary["mean_completion_ratio"], step)
    writer.add_scalar("eval/success_rate", summary["success_rate"], step)
    writer.add_scalar("eval/mean_steps", summary["mean_steps"], step)
    writer.add_scalar("eval/mean_reward", summary["mean_reward"], step)
