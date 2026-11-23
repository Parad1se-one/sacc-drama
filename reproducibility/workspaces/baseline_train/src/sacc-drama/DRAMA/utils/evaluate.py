import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from tqdm import tqdm

from configs.difficulty_loader import load_config
from DRAMA.starts import load_start_set
from envs.grid_env import GridAreaEnv
from marl.agents import QMIXAgent
from reward_machines.reward_machine_enhanced import EnhancedRewardMachine
from utils.utils import build_obs_list_for_agents, build_state, device, get_action_masks


def build_reward_machine(config, rm_kwargs=None):
    params = dict(rm_kwargs or {})
    return EnhancedRewardMachine(
        config["task_dependencies"],
        num_agents=config["num_robots"],
        regions=config.get("regions"),
        **params,
    )


def build_eval_env(config, uses_rm=False, rm_kwargs=None):
    env = GridAreaEnv(config=config)
    rm = build_reward_machine(config, rm_kwargs=rm_kwargs) if uses_rm else None
    return env, rm


def _resolve_max_steps(difficulty, explicit=None):
    if explicit is not None:
        return explicit
    return 500 if difficulty == "hard" else 300


def _episode_summary(ep_idx, reward, completion, steps):
    success = 1.0 if completion >= 0.999999 else 0.0
    return {
        "episode": ep_idx,
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


@torch.no_grad()
def evaluate_decentralized_policy(
    agents,
    config,
    *,
    uses_rm,
    episodes,
    difficulty,
    max_steps=None,
    render=False,
    seed=42,
    rm_kwargs=None,
    algo="qmix",
    checkpoint_path=None,
    metadata=None,
    json_out=None,
    start_set_path=None,
):
    max_steps = _resolve_max_steps(difficulty, explicit=max_steps)
    start_set = load_start_set(start_set_path, expected_difficulty=difficulty) if start_set_path else None
    if start_set is not None and episodes != start_set["count"]:
        raise ValueError("Evaluation episodes must match the frozen start-set count")
    results = []

    for ep in tqdm(range(episodes), desc=f"Evaluating {algo}"):
        env, rm = build_eval_env(config, uses_rm=uses_rm, rm_kwargs=rm_kwargs)
        env.reset(seed=None if start_set is not None else seed + ep)
        if start_set is not None:
            positions = start_set["starts"][ep]
            for robot, position in zip(env.robots, positions, strict=True):
                robot["pos"] = position
            env._update_passability()
        if rm is not None:
            rm.reset(env=env)

        obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)
        ep_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action_masks = get_action_masks(env)
            actions_int = [
                agents[i].act(obs_list[i], eps=0.0, action_mask=action_masks[i])
                for i in range(len(agents))
            ]
            rewards_env, infos = env.step(actions_int)

            if rm is not None:
                events = [info.get("status") for info in infos]
                rm_rewards, _, _ = rm.step(events, env=env)
                total_rewards = [
                    (rewards_env[i] if i < len(rewards_env) else 0.0)
                    + (rm_rewards[i] if i < len(rm_rewards) else 0.0)
                    for i in range(len(agents))
                ]
            else:
                total_rewards = [rewards_env[i] if i < len(rewards_env) else 0.0 for i in range(len(agents))]

            ep_reward += sum(total_rewards)
            steps += 1
            done = env.is_all_done()

            if render:
                env.render()

            if not done and steps < max_steps:
                obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)

        completion = env.get_completion_ratio()
        episode_result = _episode_summary(ep + 1, ep_reward, completion, steps)
        if start_set is not None:
            episode_result["start_positions"] = [list(pos) for pos in start_set["starts"][ep]]
        results.append(episode_result)

    summary = _aggregate_episode_summaries(results)
    payload = {
        "algo": algo,
        "difficulty": difficulty,
        "checkpoint_path": checkpoint_path,
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
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


@torch.no_grad()
def load_qmix_family_checkpoint(checkpoint_path, *, algo, difficulty):
    ckpt = torch.load(checkpoint_path, map_location=device)
    meta = dict(ckpt.get("meta", {}))
    num_agents = ckpt["num_agents"]
    obs_dim = ckpt["obs_dim"]
    action_n = ckpt["action_n"]

    agents = [QMIXAgent(obs_dim, action_n) for _ in range(num_agents)]
    for i in range(num_agents):
        agents[i].net.load_state_dict(ckpt[f"agent_{i}"])
        agents[i].target.load_state_dict(ckpt[f"agent_{i}"])
        agents[i].net.eval()
        agents[i].target.eval()

    uses_rm = meta.get("uses_rm", algo == "rmix")
    state_mode = meta.get("state_mode", "legacy")
    rm_kwargs = meta.get("rm_kwargs", {})
    checkpoint_difficulty = meta.get("difficulty", difficulty)

    config, _ = load_config(checkpoint_difficulty)
    env, rm = build_eval_env(config, uses_rm=uses_rm, rm_kwargs=rm_kwargs)
    obs_dim_actual = len(build_obs_list_for_agents(env, rm if uses_rm else None)[0])
    state_dim_actual = len(build_state(env=env, rm=rm if state_mode == "rmix_v2" else None, state_mode=state_mode))

    if obs_dim_actual != obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim mismatch: ckpt={obs_dim}, actual={obs_dim_actual}, algo={algo}, difficulty={checkpoint_difficulty}"
        )
    if "state_dim" in ckpt and ckpt["state_dim"] != state_dim_actual:
        raise ValueError(
            f"Checkpoint state_dim mismatch: ckpt={ckpt['state_dim']}, actual={state_dim_actual}, state_mode={state_mode}"
        )

    meta.setdefault("algo", algo)
    meta.setdefault("difficulty", checkpoint_difficulty)
    meta.setdefault("uses_rm", uses_rm)
    meta.setdefault("state_mode", state_mode)
    meta.setdefault("rm_kwargs", rm_kwargs)
    meta.setdefault("max_steps", _resolve_max_steps(checkpoint_difficulty, explicit=meta.get("max_steps")))
    return agents, ckpt, meta, config


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    *,
    algo,
    difficulty,
    episodes,
    seed,
    json_out=None,
    render=False,
    max_steps=None,
):
    agents, _, meta, config = load_qmix_family_checkpoint(
        checkpoint_path,
        algo=algo,
        difficulty=difficulty,
    )
    max_steps = _resolve_max_steps(difficulty, explicit=max_steps or meta.get("max_steps"))
    return evaluate_decentralized_policy(
        agents,
        config,
        uses_rm=meta["uses_rm"],
        episodes=episodes,
        difficulty=difficulty,
        max_steps=max_steps,
        render=render,
        seed=seed,
        rm_kwargs=meta.get("rm_kwargs", {}),
        algo=algo,
        checkpoint_path=str(checkpoint_path),
        metadata=meta,
        json_out=json_out,
    )


def main():
    parser = argparse.ArgumentParser(description="Greedy evaluation for RMix/QMIX checkpoints")
    parser.add_argument("--algo", type=str, required=True, choices=["rmix", "qmix"])
    parser.add_argument("--difficulty", type=str, required=True, choices=["easy", "hard"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    payload = evaluate_checkpoint(
        args.checkpoint,
        algo=args.algo,
        difficulty=args.difficulty,
        episodes=args.episodes,
        seed=args.seed,
        json_out=args.json_out,
        render=args.render,
        max_steps=args.max_steps,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
