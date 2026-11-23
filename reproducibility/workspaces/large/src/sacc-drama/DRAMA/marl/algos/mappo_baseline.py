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
from utils.baseline_eval import evaluate_policy, log_eval_scalars, make_greedy_actor_selector
from utils.utils import build_obs_list_for_agents, build_state, compute_gae, device, get_action_masks



def _save_checkpoint(path, actors, critic, meta):
    payload = {
        "actors": [actor.state_dict() for actor in actors],
        "critic": critic.state_dict(),
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



def mappo_baseline(
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
    exp_name="mappo",
    save_interval=500,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    algo_name = "mappo"
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
    state_dim = len(build_state(env=env))
    action_n = env.NUM_ACTIONS

    actors = [MAPPOActor(obs_dim, action_n).to(device) for _ in range(num_agents)]
    critic = MAPPOCritic(state_dim).to(device)

    actor_params = []
    for actor in actors:
        actor_params.extend(list(actor.parameters()))
    actor_optim = optim.Adam(actor_params, lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "difficulty": difficulty,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_n": action_n,
            "uses_rm": False,
            "max_steps": rollout_len,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "use_action_mask": True,
            "entropy_coef": entropy_coef,
        }
    )

    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc="Training MAPPO"):
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
        done = False
        ep_reward = 0.0
        steps = 0
        first_done_steps = None

        for _ in range(rollout_len):
            obs_list = build_obs_list_for_agents(env)
            state = build_state(env=env)
            action_masks = get_action_masks(env)

            obs_tensor_list = [
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                for obs in obs_list
            ]
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                value = critic(state_tensor).squeeze().detach()

            actions = []
            logps = []
            for i in range(num_agents):
                probs = actors[i](obs_tensor_list[i])
                dist = _masked_categorical(probs, action_masks[i])
                sample = dist.sample()
                actions.append(int(sample.item()))
                logps.append(float(dist.log_prob(sample).item()))

            rewards_env, _infos = env.step(actions)
            per_agent_rewards = [rewards_env[i] if i < len(rewards_env) else 0.0 for i in range(num_agents)]

            rollout["obs"].append(obs_list)
            rollout["states"].append(state)
            rollout["actions"].append(actions)
            rollout["logprobs"].append(logps)
            rollout["values"].append(float(value.item()))
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

        bootstrap_state = build_state(env=env)

        rewards_agent = torch.tensor(rollout["rewards"], dtype=torch.float32, device=device)
        dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=device)
        values = torch.tensor(rollout["values"], dtype=torch.float32, device=device)
        if value_noise_std > 0:
            values = values + torch.randn_like(values) * value_noise_std

        if rollout["dones"][-1] == 1.0:
            next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
        else:
            last_state = torch.tensor(bootstrap_state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                next_value = critic(last_state).squeeze().detach()

        team_rewards = rewards_agent.sum(dim=1)
        advantages, returns = compute_gae(team_rewards.unsqueeze(1), values, next_value, dones, gamma=gamma, lam=lam)
        advantages = advantages.squeeze(1)
        returns = returns.squeeze(1)

        if advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        old_logprobs = torch.tensor(rollout["logprobs"], dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(rollout["actions"], dtype=torch.long, device=device)
        action_masks_tensor = torch.tensor(np.array(rollout["action_masks"], dtype=bool), dtype=torch.bool, device=device)
        states_tensor = torch.tensor(np.array(rollout["states"], dtype=np.float32), dtype=torch.float32, device=device)

        T = len(rollout["rewards"])
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        for _ in range(ppo_epochs):
            idxs = np.arange(T)
            np.random.shuffle(idxs)
            for start in range(0, T, mini_batch_size):
                mb_idx = idxs[start:start + mini_batch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states_tensor[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logp = old_logprobs[mb_idx]
                mb_masks = action_masks_tensor[mb_idx]

                values_pred = critic(mb_states).squeeze()
                critic_loss = F.mse_loss(values_pred, mb_returns.detach())
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                critic_optim.step()
                last_critic_loss = float(critic_loss.item())

                actor_loss_total = 0.0
                for i in range(num_agents):
                    obs_i = np.array([rollout["obs"][t][i] for t in mb_idx], dtype=np.float32)
                    obs_i = torch.tensor(obs_i, dtype=torch.float32, device=device)
                    probs = actors[i](obs_i)
                    dist = _masked_categorical(probs, mb_masks[:, i, :])
                    logp_now = dist.log_prob(mb_actions[:, i])
                    old_lp = mb_old_logp[:, i]
                    ratio = torch.exp(logp_now - old_lp)
                    adv_i = mb_adv
                    obj1 = ratio * adv_i
                    obj2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_i
                    actor_loss_total = actor_loss_total + (-torch.min(obj1, obj2).mean() - entropy_coef * dist.entropy().mean())

                actor_optim.zero_grad()
                actor_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, grad_clip)
                actor_optim.step()
                last_actor_loss = float(actor_loss_total.item())

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
            _save_checkpoint(ckpt_path, actors, critic, metadata)
            print(f"[Checkpoint] Saved MAPPO at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = evaluate_policy(
                action_selector=make_greedy_actor_selector(actors),
                difficulty=difficulty,
                episodes=eval_episodes,
                seed=metadata.get("seed", 42),
                algo=algo_name,
                max_steps=rollout_len,
                metadata=metadata,
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
                _save_checkpoint(best_ckpt_path, actors, critic, best_meta)
                print(
                    f"[Best] Saved MAPPO best checkpoint at ep {ep+1} | "
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
    _save_checkpoint(final_path, actors, critic, final_meta)
    writer.close()
    return actors, critic
