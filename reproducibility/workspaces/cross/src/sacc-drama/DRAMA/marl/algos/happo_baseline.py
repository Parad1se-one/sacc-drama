import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marl.agents import MAPPOActor, MAPPOCritic
from utils.baseline_eval import evaluate_policy, log_eval_scalars, make_greedy_actor_selector
from utils.utils import build_obs_list_for_agents, build_state, compute_gae, device, get_action_masks


LEGACY_RM_KWARGS = {
    "reward_grid_advance": 10.0,
    "reward_region_phase": 50.0,
    "reward_region_done": 100.0,
    "reward_all_done": 500.0,
    "penalty_idle": -0.05,
    "penalty_invalid": -0.5,
    "penalty_blocked": -1.0,
    "distance_shaping_coeff": 0.1,
}

STABILIZED_RM_KWARGS = {
    "reward_grid_advance": 1.0,
    "reward_region_phase": 5.0,
    "reward_region_done": 10.0,
    "reward_all_done": 50.0,
    "penalty_idle": -0.005,
    "penalty_invalid": -0.05,
    "penalty_blocked": -0.1,
    "penalty_oscillation": -0.02,
    "distance_shaping_coeff": 0.05,
}


def _save_checkpoint(path, actors, critics, meta):
    payload = {
        "actors": [actor.state_dict() for actor in actors],
        "meta": dict(meta),
    }
    if isinstance(critics, list):
        payload["critics"] = [critic.state_dict() for critic in critics]
    else:
        payload["critic"] = critics.state_dict()
    torch.save(payload, path)



def _all_true_masks(num_agents, action_n):
    return [np.ones(action_n, dtype=bool) for _ in range(num_agents)]



def _get_action_masks(env, use_action_mask, action_n):
    if use_action_mask:
        return get_action_masks(env)
    return _all_true_masks(len(env.robots), action_n)



def _apply_rm_profile(rm, reward_profile):
    if rm is None:
        return {}
    if reward_profile == "legacy":
        params = LEGACY_RM_KWARGS
    elif reward_profile == "stabilized":
        params = STABILIZED_RM_KWARGS
    else:
        raise ValueError(f"Unknown reward_profile: {reward_profile}")

    for key, value in params.items():
        setattr(rm, key, value)
    return deepcopy(params)



def _masked_categorical(probs, action_mask):
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=probs.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    masked_probs = probs.masked_fill(~mask, 0.0)
    denom = masked_probs.sum(dim=-1, keepdim=True)
    # Fall back to STAY when a mask is unexpectedly empty.
    fallback = torch.zeros_like(masked_probs)
    fallback[..., -1] = 1.0
    masked_probs = torch.where(denom > 1e-8, masked_probs / denom.clamp_min(1e-8), fallback)
    return Categorical(masked_probs)



def _state_vector(env, rm, state_mode):
    return build_state(
        env=env,
        rm=rm if state_mode == "rmix_v2" else None,
        state_mode=state_mode,
    )


def _stack_critic_values(critics, state_tensor):
    return torch.stack([critic(state_tensor).squeeze() for critic in critics], dim=0)


def _compute_gae_per_agent(rewards_agent, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Per-agent GAE for a list of centralized critics.

    Args:
        rewards_agent: (T, N)
        values: (T, N)
        next_values: (N,)
        dones: (T,)
    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    timesteps, num_agents = rewards_agent.shape
    advantages = torch.zeros((timesteps, num_agents), dtype=torch.float32, device=rewards_agent.device)
    returns = torch.zeros((timesteps, num_agents), dtype=torch.float32, device=rewards_agent.device)

    next_val = next_values.clone().detach().to(rewards_agent.device)
    gae = torch.zeros(num_agents, dtype=torch.float32, device=rewards_agent.device)
    for t in reversed(range(timesteps)):
        mask = 1.0 - dones[t]
        delta = rewards_agent[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_val = values[t]

    return advantages, returns



def _train_happo_family(
    env,
    rm,
    *,
    uses_rm,
    algo_name,
    variant,
    episodes,
    rollout_len,
    ppo_epochs,
    mini_batch_size,
    clip_eps,
    lr,
    gamma,
    lam,
    grad_clip,
    advantage_norm,
    value_noise_std,
    entropy_coef,
    log_dir,
    exp_name,
    save_interval,
    difficulty,
    eval_interval,
    eval_episodes,
    checkpoint_meta,
    state_mode,
    reward_profile,
    use_action_mask,
    per_agent_critic,
):
    if uses_rm and rm is None:
        raise ValueError("RM-enabled HAPPO requires a reward machine instance")

    ckpt_dir = os.path.join("checkpoints", algo_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_save_path = os.path.join(log_dir, algo_name, f"{exp_name}_{timestamp}")
    os.makedirs(runs_save_path, exist_ok=True)
    eval_dir = os.path.join(runs_save_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_save_path)

    rm_kwargs = _apply_rm_profile(rm, reward_profile) if uses_rm else {}

    env.reset()
    if rm is not None:
        rm.reset(env=env)

    num_agents = len(env.robots)
    obs_dim = len(build_obs_list_for_agents(env, rm if uses_rm else None)[0])
    state_dim = len(_state_vector(env, rm if uses_rm else None, state_mode))
    action_n = env.NUM_ACTIONS

    actors = [MAPPOActor(obs_dim, action_n).to(device) for _ in range(num_agents)]
    critics = (
        [MAPPOCritic(state_dim).to(device) for _ in range(num_agents)]
        if per_agent_critic
        else MAPPOCritic(state_dim).to(device)
    )
    actor_optims = [optim.Adam(actor.parameters(), lr=lr) for actor in actors]
    critic_optims = (
        [optim.Adam(critic.parameters(), lr=lr) for critic in critics]
        if per_agent_critic
        else optim.Adam(critics.parameters(), lr=lr)
    )

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "variant": variant,
            "difficulty": difficulty,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_n": action_n,
            "uses_rm": uses_rm,
            "state_mode": state_mode,
            "rm_kwargs": rm_kwargs,
            "max_steps": rollout_len,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "use_action_mask": use_action_mask,
            "entropy_coef": entropy_coef,
            "per_agent_critic": per_agent_critic,
            "critic_type": "per_agent_centralized" if per_agent_critic else "shared_centralized",
        }
    )

    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc=f"Training {algo_name.upper()} ({variant})"):
        rollout = {
            "obs": [],
            "states": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "action_masks": [],
        }

        env.reset()
        if rm is not None:
            rm.reset(env=env)

        done = False
        ep_reward = 0.0
        steps = 0
        first_done_steps = None

        for _ in range(rollout_len):
            obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)
            state = _state_vector(env, rm if uses_rm else None, state_mode)
            action_masks = _get_action_masks(env, use_action_mask, action_n)
            obs_tensor_list = [
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                for obs in obs_list
            ]
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value = (
                    _stack_critic_values(critics, state_tensor).detach()
                    if per_agent_critic
                    else critics(state_tensor).squeeze().detach()
                )

            actions = []
            logps = []
            for i in range(num_agents):
                probs = actors[i](obs_tensor_list[i])
                dist = _masked_categorical(probs, action_masks[i])
                sample = dist.sample()
                actions.append(int(sample.item()))
                logps.append(float(dist.log_prob(sample).item()))

            rewards_env, infos = env.step(actions)
            if rm is not None:
                events = [info.get("status", None) for info in infos]
                rm_rewards, _, _ = rm.step(events, env=env)
                per_agent_rewards = [
                    (rewards_env[i] if i < len(rewards_env) else 0.0)
                    + (rm_rewards[i] if i < len(rm_rewards) else 0.0)
                    for i in range(num_agents)
                ]
            else:
                per_agent_rewards = [rewards_env[i] if i < len(rewards_env) else 0.0 for i in range(num_agents)]

            rollout["obs"].append(obs_list)
            rollout["states"].append(state)
            rollout["actions"].append(actions)
            rollout["logprobs"].append(logps)
            rollout["values"].append(
                value.detach().cpu().tolist() if per_agent_critic else float(value.item())
            )
            rollout["rewards"].append(per_agent_rewards)
            rollout["dones"].append(0.0)
            rollout["action_masks"].append(action_masks)

            ep_reward += float(sum(per_agent_rewards))
            steps += 1

            if env.is_all_done():
                rollout["dones"][-1] = 1.0
                done = True
                if first_done_steps is None:
                    first_done_steps = steps

            if done:
                break

        bootstrap_state = _state_vector(env, rm if uses_rm else None, state_mode)

        rewards_agent = torch.tensor(rollout["rewards"], dtype=torch.float32, device=device)
        dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=device)
        values = torch.tensor(rollout["values"], dtype=torch.float32, device=device)
        if value_noise_std > 0:
            values = values + torch.randn_like(values) * value_noise_std

        if rollout["dones"][-1] == 1.0:
            next_value = (
                torch.zeros(num_agents, dtype=torch.float32, device=device)
                if per_agent_critic
                else torch.tensor(0.0, dtype=torch.float32, device=device)
            )
        else:
            last_state = torch.tensor(bootstrap_state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                next_value = (
                    _stack_critic_values(critics, last_state).detach()
                    if per_agent_critic
                    else critics(last_state).squeeze().detach()
                )

        if per_agent_critic:
            advantages, returns = _compute_gae_per_agent(
                rewards_agent,
                values,
                next_value,
                dones,
                gamma=gamma,
                lam=lam,
            )
            critic_returns = returns
            if advantage_norm:
                adv_mean = advantages.mean(dim=0, keepdim=True)
                adv_std = advantages.std(dim=0, unbiased=False, keepdim=True)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        elif uses_rm:
            advantages, returns = compute_gae(
                rewards_agent,
                values,
                next_value,
                dones,
                gamma=gamma,
                lam=lam,
            )
            critic_returns = returns.mean(dim=1)
            if advantage_norm:
                adv_mean = advantages.mean(dim=0, keepdim=True)
                adv_std = advantages.std(dim=0, unbiased=False, keepdim=True)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            team_rewards = rewards_agent.sum(dim=1)
            advantages, returns = compute_gae(
                team_rewards.unsqueeze(1),
                values,
                next_value,
                dones,
                gamma=gamma,
                lam=lam,
            )
            advantages = advantages.squeeze(1)
            returns = returns.squeeze(1)
            critic_returns = returns
            if advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_logprobs = torch.tensor(rollout["logprobs"], dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(rollout["actions"], dtype=torch.long, device=device)
        action_masks_tensor = torch.tensor(
            np.array(rollout["action_masks"], dtype=bool),
            dtype=torch.bool,
            device=device,
        )
        states_tensor = torch.tensor(
            np.array(rollout["states"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        timesteps = len(rollout["rewards"])
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        for _ in range(ppo_epochs):
            idxs = np.arange(timesteps)
            np.random.shuffle(idxs)
            for start in range(0, timesteps, mini_batch_size):
                mb_idx = idxs[start:start + mini_batch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states_tensor[mb_idx]
                mb_returns = critic_returns[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logp = old_logprobs[mb_idx]
                mb_masks = action_masks_tensor[mb_idx]

                if per_agent_critic:
                    critic_loss_total = 0.0
                    for i in range(num_agents):
                        values_pred_i = critics[i](mb_states).squeeze()
                        critic_loss_i = F.mse_loss(values_pred_i, mb_returns[:, i].detach())
                        critic_optims[i].zero_grad()
                        critic_loss_i.backward()
                        torch.nn.utils.clip_grad_norm_(critics[i].parameters(), grad_clip)
                        critic_optims[i].step()
                        critic_loss_total += float(critic_loss_i.item())
                    last_critic_loss = critic_loss_total / max(num_agents, 1)
                else:
                    values_pred = critics(mb_states).squeeze()
                    critic_loss = F.mse_loss(values_pred, mb_returns.detach())
                    critic_optims.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critics.parameters(), grad_clip)
                    critic_optims.step()
                    last_critic_loss = float(critic_loss.item())

                joint_ratio = torch.ones(len(mb_idx), dtype=torch.float32, device=device)
                actor_loss_total = 0.0
                for i in range(num_agents):
                    obs_i = np.array([rollout["obs"][t][i] for t in mb_idx], dtype=np.float32)
                    obs_i = torch.tensor(obs_i, dtype=torch.float32, device=device)
                    probs = actors[i](obs_i)
                    dist = _masked_categorical(probs, mb_masks[:, i, :])
                    logp_now = dist.log_prob(mb_actions[:, i])
                    old_lp = mb_old_logp[:, i]
                    adv_i = (mb_adv[:, i] if per_agent_critic or uses_rm else mb_adv) * joint_ratio
                    ratio = torch.exp(logp_now - old_lp)
                    obj1 = ratio * adv_i
                    obj2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
                    actor_loss = -torch.min(obj1, obj2).mean() - entropy_coef * dist.entropy().mean()

                    actor_optims[i].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors[i].parameters(), grad_clip)
                    actor_optims[i].step()
                    actor_loss_total += float(actor_loss.item())

                    with torch.no_grad():
                        probs_updated = actors[i](obs_i)
                        dist_updated = _masked_categorical(probs_updated, mb_masks[:, i, :])
                        logp_updated = dist_updated.log_prob(mb_actions[:, i])
                        joint_ratio = joint_ratio * torch.exp(logp_updated - old_lp)

                last_actor_loss = actor_loss_total

        completion = env.get_completion_ratio()
        writer.add_scalar("episode_reward", ep_reward, ep + 1)
        writer.add_scalar("completion_ratio", completion, ep + 1)
        writer.add_scalar("train/episode_reward", ep_reward, ep + 1)
        writer.add_scalar("train/completion_ratio", completion, ep + 1)
        writer.add_scalar("train/episode_length", steps, ep + 1)
        writer.add_scalar("train/success", 1.0 if done else 0.0, ep + 1)
        writer.add_scalar("train/actor_loss", last_actor_loss, ep + 1)
        writer.add_scalar("train/critic_loss", last_critic_loss, ep + 1)
        writer.add_scalar("train/value_mean", values.mean().item(), ep + 1)
        writer.add_scalar("train/adv_mean", advantages.mean().item(), ep + 1)
        if first_done_steps is not None:
            writer.add_scalar("train/steps_to_complete", first_done_steps, ep + 1)

        if (ep + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_ep{ep+1}.pth")
            _save_checkpoint(ckpt_path, actors, critics, metadata)
            print(f"[Checkpoint] Saved {algo_name.upper()} at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = evaluate_policy(
                action_selector=make_greedy_actor_selector(actors, use_action_mask=use_action_mask),
                difficulty=difficulty,
                episodes=eval_episodes,
                seed=metadata.get("seed", 42),
                algo=algo_name,
                max_steps=rollout_len,
                metadata=metadata,
                json_out=os.path.join(eval_dir, f"eval_ep{ep+1}.json"),
                uses_rm=uses_rm,
                rm_kwargs=rm_kwargs,
            )
            eval_summary = eval_payload["summary"]
            log_eval_scalars(writer, eval_summary, ep + 1)

            better = False
            if best_eval is None:
                better = True
            elif eval_summary["mean_completion_ratio"] > best_eval["mean_completion_ratio"]:
                better = True
            elif (
                eval_summary["mean_completion_ratio"] == best_eval["mean_completion_ratio"]
                and eval_summary["success_rate"] > best_eval["success_rate"]
            ):
                better = True

            if better:
                best_eval = dict(eval_summary)
                best_meta = dict(metadata)
                best_meta["best_eval"] = best_eval
                _save_checkpoint(best_ckpt_path, actors, critics, best_meta)
                print(
                    f"[Best] Saved {algo_name.upper()} best checkpoint at ep {ep+1} | "
                    f"completion={best_eval['mean_completion_ratio']:.3f} "
                    f"success={best_eval['success_rate']:.3f}",
                    flush=True,
                )

        if (ep + 1) % 10 == 0:
            print(
                f"\nEp {ep+1}/{episodes} | completion={completion:.3f} | "
                f"steps={steps} | reward={ep_reward:.2f} | done={done}",
                flush=True,
            )

    final_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_final.pth")
    final_meta = dict(metadata)
    if best_eval is not None:
        final_meta["best_eval"] = best_eval
        final_meta["best_by_completion_path"] = best_ckpt_path
    _save_checkpoint(final_path, actors, critics, final_meta)
    writer.close()
    return actors, critics



def happo_baseline(
    env,
    *,
    episodes=3000,
    rollout_len=300,
    ppo_epochs=8,
    mini_batch_size=64,
    clip_eps=0.2,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    grad_clip=0.5,
    advantage_norm=True,
    value_noise_std=0.0,
    entropy_coef=0.01,
    log_dir="runs",
    exp_name="happo",
    save_interval=500,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
    use_action_mask=True,
):
    return _train_happo_family(
        env,
        rm=None,
        uses_rm=False,
        algo_name="happo",
        variant="baseline",
        episodes=episodes,
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        clip_eps=clip_eps,
        lr=lr,
        gamma=gamma,
        lam=lam,
        grad_clip=grad_clip,
        advantage_norm=advantage_norm,
        value_noise_std=value_noise_std,
        entropy_coef=entropy_coef,
        log_dir=log_dir,
        exp_name=exp_name,
        save_interval=save_interval,
        difficulty=difficulty,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        checkpoint_meta=checkpoint_meta,
        state_mode="legacy",
        reward_profile="legacy",
        use_action_mask=use_action_mask,
        per_agent_critic=False,
    )



def happo_rm(
    env,
    rm,
    *,
    episodes=3000,
    rollout_len=300,
    ppo_epochs=8,
    mini_batch_size=64,
    clip_eps=0.2,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    grad_clip=0.5,
    advantage_norm=True,
    value_noise_std=0.0,
    entropy_coef=0.01,
    log_dir="runs",
    exp_name="happo_rm",
    save_interval=500,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
    variant="rm_happo",
    state_mode="rmix_v2",
    reward_profile="stabilized",
    use_action_mask=True,
):
    return _train_happo_family(
        env,
        rm=rm,
        uses_rm=True,
        algo_name="happo_rm",
        variant=variant,
        episodes=episodes,
        rollout_len=rollout_len,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        clip_eps=clip_eps,
        lr=lr,
        gamma=gamma,
        lam=lam,
        grad_clip=grad_clip,
        advantage_norm=advantage_norm,
        value_noise_std=value_noise_std,
        entropy_coef=entropy_coef,
        log_dir=log_dir,
        exp_name=exp_name,
        save_interval=save_interval,
        difficulty=difficulty,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        checkpoint_meta=checkpoint_meta,
        state_mode=state_mode,
        reward_profile=reward_profile,
        use_action_mask=use_action_mask,
        per_agent_critic=True,
    )
