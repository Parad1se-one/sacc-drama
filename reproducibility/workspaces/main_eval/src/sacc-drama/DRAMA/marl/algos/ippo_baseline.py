import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marl.agents import MAPPOActor, MAPPOCritic
from utils.baseline_eval import evaluate_policy, log_eval_scalars, make_actor_selector
from utils.utils import build_obs_list_for_agents, compute_gae, device, get_action_masks



def _save_checkpoint(path, actors, critics, meta):
    payload = {
        "actors": [actor.state_dict() for actor in actors],
        "critics": [critic.state_dict() for critic in critics],
        "meta": dict(meta),
    }
    torch.save(payload, path)


def _masked_categorical(probs, action_mask):
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=probs.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    masked_probs = probs.masked_fill(~mask, 0.0)
    denom = masked_probs.sum(dim=-1, keepdim=True)
    fallback = torch.full_like(masked_probs, 1.0 / masked_probs.shape[-1])
    masked_probs = torch.where(denom > 1e-8, masked_probs / denom.clamp_min(1e-8), fallback)
    return Categorical(masked_probs)



def ippo_baseline(
    env,
    *,
    episodes=3000,
    rollout_len=300,
    ppo_epochs=4,
    mini_batch_size=300,
    clip_eps=0.2,
    lr=1e-4,
    gamma=0.99,
    lam=0.95,
    grad_clip=0.5,
    advantage_norm=True,
    value_noise_std=0.0,
    epsilon_start=0.3,
    epsilon_end=0.05,
    epsilon_decay=0.998,
    entropy_coef=0.01,
    log_dir="runs",
    exp_name="ippo",
    save_interval=500,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    algo_name = "ippo"
    ckpt_dir = os.path.join("checkpoints", algo_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_save_path = os.path.join(log_dir, algo_name, f"{exp_name}_{timestamp}")
    os.makedirs(runs_save_path, exist_ok=True)
    eval_dir = os.path.join(runs_save_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_save_path)

    env.reset()
    num_agents = len(env.robots)
    obs_dim = len(build_obs_list_for_agents(env)[0])
    action_n = env.NUM_ACTIONS

    actors = [MAPPOActor(obs_dim, action_n).to(device) for _ in range(num_agents)]
    critics = [MAPPOCritic(obs_dim).to(device) for _ in range(num_agents)]
    actor_optims = [optim.Adam(actor.parameters(), lr=lr) for actor in actors]
    critic_optims = [optim.Adam(critic.parameters(), lr=lr) for critic in critics]

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "difficulty": difficulty,
            "obs_dim": obs_dim,
            "action_n": action_n,
            "uses_rm": False,
            "max_steps": rollout_len,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "use_action_mask": True,
            "entropy_coef": entropy_coef,
        }
    )

    epsilon = epsilon_start
    total_steps = 0
    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc="Training IPPO"):
        rollout = [
            {
                "obs": [],
                "actions": [],
                "logprobs": [],
                "values": [],
                "rewards": [],
                "dones": [],
                "action_masks": [],
            }
            for _ in range(num_agents)
        ]

        env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        first_done_steps = None

        for _ in range(rollout_len):
            obs_list = build_obs_list_for_agents(env)
            action_masks = get_action_masks(env)
            actions = np.zeros(num_agents, dtype=np.int32)
            logps = []
            values = []
            per_agent_rewards = []

            for i in range(num_agents):
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    value = critics[i](obs_tensor).squeeze().detach()
                    if value_noise_std > 0:
                        value = value + torch.randn_like(value) * value_noise_std

                probs = actors[i](obs_tensor)
                dist = _masked_categorical(probs, action_masks[i])
                sample = dist.sample()
                action = int(sample.item())
                logp = float(dist.log_prob(sample).item())

                actions[i] = action
                logps.append(logp)
                values.append(float(value.item()))

            rewards_env, _infos = env.step(actions)
            for i in range(num_agents):
                per_agent_rewards.append(rewards_env[i] if i < len(rewards_env) else 0.0)

            done = env.is_all_done()
            if done and first_done_steps is None:
                first_done_steps = steps + 1

            for i in range(num_agents):
                rollout[i]["obs"].append(obs_list[i])
                rollout[i]["actions"].append(actions[i])
                rollout[i]["logprobs"].append(logps[i])
                rollout[i]["values"].append(values[i])
                rollout[i]["rewards"].append(per_agent_rewards[i])
                rollout[i]["dones"].append(1.0 if done else 0.0)
                rollout[i]["action_masks"].append(action_masks[i])

            ep_reward += float(sum(per_agent_rewards))
            steps += 1
            total_steps += 1

            if done:
                break

        bootstrap_obs_list = build_obs_list_for_agents(env)

        for i in range(num_agents):
            obs_arr = torch.tensor(np.array(rollout[i]["obs"]), dtype=torch.float32, device=device)
            actions_arr = torch.tensor(rollout[i]["actions"], dtype=torch.long, device=device)
            logprobs_arr = torch.tensor(rollout[i]["logprobs"], dtype=torch.float32, device=device)
            values_arr = torch.tensor(rollout[i]["values"], dtype=torch.float32, device=device)
            rewards_arr = torch.tensor(rollout[i]["rewards"], dtype=torch.float32, device=device)
            dones_arr = torch.tensor(rollout[i]["dones"], dtype=torch.float32, device=device)
            masks_arr = torch.tensor(np.array(rollout[i]["action_masks"], dtype=bool), dtype=torch.bool, device=device)

            if rollout[i]["dones"][-1] == 1.0:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
            else:
                next_obs = torch.tensor(bootstrap_obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    next_value = critics[i](next_obs).squeeze().detach()

            advantages, returns = compute_gae(
                rewards_arr.unsqueeze(1), values_arr, next_value, dones_arr, gamma=gamma, lam=lam
            )
            advantages = advantages.squeeze(1)
            returns = returns.squeeze(1)

            if advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            last_actor_loss = 0.0
            last_critic_loss = 0.0
            T = len(rewards_arr)
            idxs = np.arange(T)

            for _ in range(ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, T, mini_batch_size):
                    mb_idx = idxs[start:start + mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs = obs_arr[mb_idx]
                    mb_actions = actions_arr[mb_idx]
                    mb_old_logp = logprobs_arr[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]
                    mb_masks = masks_arr[mb_idx]

                    values_pred = critics[i](mb_obs).squeeze()
                    critic_loss = F.mse_loss(values_pred, mb_ret)
                    critic_optims[i].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critics[i].parameters(), grad_clip)
                    critic_optims[i].step()

                    probs = actors[i](mb_obs)
                    dist = _masked_categorical(probs, mb_masks)
                    logp_now = dist.log_prob(mb_actions)
                    ratio = torch.exp(logp_now - mb_old_logp)
                    obj1 = ratio * mb_adv
                    obj2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                    entropy = dist.entropy().mean()
                    actor_loss = -torch.min(obj1, obj2).mean() - entropy_coef * entropy
                    actor_optims[i].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors[i].parameters(), grad_clip)
                    actor_optims[i].step()

                    last_actor_loss = float(actor_loss.item())
                    last_critic_loss = float(critic_loss.item())

            writer.add_scalar(f"agent{i}/actor_loss", last_actor_loss, ep + 1)
            writer.add_scalar(f"agent{i}/critic_loss", last_critic_loss, ep + 1)
            writer.add_scalar(f"agent{i}/average_return", returns.mean().item(), ep + 1)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("train/epsilon", epsilon, ep + 1)

        completion = env.get_completion_ratio()
        writer.add_scalar("episode_reward", ep_reward, ep + 1)
        writer.add_scalar("completion_ratio", completion, ep + 1)
        writer.add_scalar("train/episode_reward", ep_reward, ep + 1)
        writer.add_scalar("train/completion_ratio", completion, ep + 1)
        writer.add_scalar("train/episode_length", steps, ep + 1)
        writer.add_scalar("train/success", 1.0 if done else 0.0, ep + 1)
        if first_done_steps is not None:
            writer.add_scalar("train/steps_to_complete", first_done_steps, ep + 1)

        if (ep + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_ep{ep+1}.pth")
            _save_checkpoint(ckpt_path, actors, critics, metadata)
            print(f"[Checkpoint] Saved IPPO at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = evaluate_policy(
                action_selector=make_actor_selector(
                    actors,
                    policy_mode=metadata.get("eval_policy_mode", "greedy"),
                    temperature=metadata.get("eval_temperature", 0.35),
                    rng_seed=metadata.get("eval_seed", metadata.get("seed", 42)),
                ),
                difficulty=difficulty,
                episodes=eval_episodes,
                seed=metadata.get("eval_seed", metadata.get("seed", 42)),
                algo=algo_name,
                max_steps=rollout_len,
                metadata=metadata,
                start_set_path=metadata.get("validation_start_set"),
                json_out=os.path.join(eval_dir, f"eval_ep{ep+1}.json"),
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
                    f"[Best] Saved IPPO best checkpoint at ep {ep+1} | "
                    f"completion={best_eval['mean_completion_ratio']:.3f} success={best_eval['success_rate']:.3f}",
                    flush=True,
                )

        if (ep + 1) % 10 == 0:
            print(
                f"\nEp {ep+1}/{episodes} | completion={completion:.3f} | steps={steps} | reward={ep_reward:.2f} | done={done}",
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
