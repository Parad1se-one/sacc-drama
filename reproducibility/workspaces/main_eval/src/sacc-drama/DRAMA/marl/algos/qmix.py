import math
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marl.agents import QMIXAgent
from marl.buffer import ReplayBuffer
from marl.mixer import Mixer
from utils.evaluate import evaluate_decentralized_policy
from utils.utils import build_obs_list_for_agents, build_state, device, get_action_masks


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

MEDIUM_ADAPTED_RM_KWARGS = {
    "reward_grid_advance": 1.0,
    "reward_region_phase": 3.0,
    "reward_region_done": 6.0,
    "reward_all_done": 30.0,
    "penalty_idle": -0.002,
    "penalty_invalid": -0.05,
    "penalty_blocked": -0.1,
    "penalty_oscillation": -0.01,
    "distance_shaping_coeff": 0.02,
}


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
    elif reward_profile == "medium_adapted":
        params = MEDIUM_ADAPTED_RM_KWARGS
    else:
        params = STABILIZED_RM_KWARGS
    for key, value in params.items():
        setattr(rm, key, value)
    return deepcopy(params)


def _checkpoint_payload(agents, mixer, obs_dim, state_dim, action_n, meta):
    payload = {
        "num_agents": len(agents),
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_n": action_n,
        "mixer": mixer.state_dict(),
        "meta": deepcopy(meta),
    }
    for i, ag in enumerate(agents):
        payload[f"agent_{i}"] = ag.net.state_dict()
    return payload


def _save_checkpoint(path, agents, mixer, obs_dim, state_dim, action_n, meta):
    torch.save(_checkpoint_payload(agents, mixer, obs_dim, state_dim, action_n, meta), path)


def _masked_next_q_values(agents, next_obs_batch, next_action_mask_batch, use_double_q):
    next_q_list = []
    for i, agent in enumerate(agents):
        mask = next_action_mask_batch[:, i, :]
        target_q = agent.target_q_values(next_obs_batch[i])
        if use_double_q:
            online_q = agent.q_values(next_obs_batch[i]).masked_fill(~mask, -1e9)
            next_actions = online_q.argmax(dim=1, keepdim=True)
            next_q = target_q.gather(1, next_actions).squeeze(1)
        else:
            next_q = target_q.masked_fill(~mask, -1e9).max(dim=1).values
        next_q_list.append(next_q)
    return torch.stack(next_q_list, dim=1)


@torch.no_grad()
def _run_periodic_eval(
    agents,
    env,
    *,
    uses_rm,
    rm_kwargs,
    difficulty,
    algo_name,
    eval_episodes,
    max_steps,
    seed,
    json_out,
    metadata,
):
    return evaluate_decentralized_policy(
        agents,
        env._config,
        uses_rm=uses_rm,
        episodes=eval_episodes,
        difficulty=difficulty,
        max_steps=max_steps,
        render=False,
        seed=seed,
        rm_kwargs=rm_kwargs,
        algo=algo_name,
        checkpoint_path=None,
        metadata=metadata,
        json_out=json_out,
        start_set_path=metadata.get("validation_start_set"),
    )


def _train_qmix_family(
    env,
    rm,
    *,
    uses_rm,
    algo_name,
    variant,
    episodes,
    batch_size,
    buffer_capacity,
    eps_start,
    eps_end,
    eps_decay,
    lr,
    gamma,
    log_dir,
    exp_name,
    save_interval,
    max_steps,
    state_mode,
    use_action_mask,
    use_double_q,
    global_reward_mode,
    reward_profile,
    eval_interval,
    eval_episodes,
    checkpoint_meta,
):
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

    obs_list0 = build_obs_list_for_agents(env, rm if uses_rm else None)
    obs_dim = len(obs_list0[0])
    state0 = build_state(env=env, rm=rm if state_mode == "rmix_v2" else None, state_mode=state_mode)
    state_dim = len(state0)
    num_agents = len(env.robots)
    action_n = env.NUM_ACTIONS

    agents = [QMIXAgent(obs_dim, action_n, lr=lr) for _ in range(num_agents)]
    mixer = Mixer(n_agents=num_agents, state_dim=state_dim).to(device)
    target_mixer = Mixer(n_agents=num_agents, state_dim=state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    params = []
    for agent in agents:
        params.extend(list(agent.net.parameters()))
    params.extend(list(mixer.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "variant": variant,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_n": action_n,
            "uses_rm": uses_rm,
            "state_mode": state_mode,
            "use_action_mask": use_action_mask,
            "use_double_q": use_double_q,
            "global_reward_mode": global_reward_mode,
            "rm_kwargs": rm_kwargs,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
        }
    )

    global_step = 0
    update_count = 0
    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc=f"Training {algo_name.upper()} ({variant})"):
        env.reset()
        if rm is not None:
            rm.reset(env=env)

        obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)
        state = build_state(env=env, rm=rm if state_mode == "rmix_v2" else None, state_mode=state_mode)
        ep_reward = 0.0
        steps = 0
        done = False

        for t in range(max_steps):
            eps = max(eps_end, eps_start - (global_step / max(eps_decay, 1)) * (eps_start - eps_end))
            if use_action_mask:
                action_masks = get_action_masks(env)
            else:
                action_masks = _all_true_masks(len(env.robots), action_n)
            actions_int = [
                agents[i].act(obs_list[i], eps=eps, action_mask=action_masks[i])
                for i in range(num_agents)
            ]
            rewards_env, infos = env.step(actions_int)
            next_obs_list = build_obs_list_for_agents(env, rm if uses_rm else None)
            next_state = build_state(env=env, rm=rm if state_mode == "rmix_v2" else None, state_mode=state_mode)
            if use_action_mask:
                next_action_masks = get_action_masks(env)
            else:
                next_action_masks = _all_true_masks(len(env.robots), action_n)

            if rm is not None:
                events = [info.get("status", None) for info in infos]
                rm_rewards, _, _ = rm.step(events, env=env)
                total_rewards = [
                    (rewards_env[i] if i < len(rewards_env) else 0.0)
                    + (rm_rewards[i] if i < len(rm_rewards) else 0.0)
                    for i in range(num_agents)
                ]
            else:
                total_rewards = [rewards_env[i] if i < len(rewards_env) else 0.0 for i in range(num_agents)]

            done = env.is_all_done()
            buffer.push(
                obs_list,
                actions_int,
                total_rewards,
                next_obs_list,
                done,
                state=state,
                next_state=next_state,
                action_masks=np.asarray(action_masks, dtype=bool),
                next_action_masks=np.asarray(next_action_masks, dtype=bool),
            )

            obs_list = next_obs_list
            state = next_state
            ep_reward += sum(total_rewards) * math.pow(gamma, t)
            steps += 1
            global_step += 1

            if len(buffer) >= batch_size:
                (
                    obs_batch,
                    next_obs_batch,
                    actions_batch,
                    rewards_batch,
                    done_batch,
                    state_batch,
                    next_state_batch,
                    _action_mask_batch,
                    next_action_mask_batch,
                ) = buffer.sample_torch(batch_size, device, return_state=True, return_masks=True)

                q_taken_list = []
                for i, agent in enumerate(agents):
                    q_vals = agent.q_values(obs_batch[i])
                    q_taken = q_vals.gather(1, actions_batch[:, i].unsqueeze(1)).squeeze(1)
                    q_taken_list.append(q_taken)
                q_taken_stack = torch.stack(q_taken_list, dim=1)
                q_tot = mixer(q_taken_stack, state_batch)

                with torch.no_grad():
                    q_next_stack = _masked_next_q_values(
                        agents,
                        next_obs_batch,
                        next_action_mask_batch,
                        use_double_q=use_double_q,
                    )
                    q_tot_next = target_mixer(q_next_stack, next_state_batch)

                if global_reward_mode == "mean":
                    r_global = rewards_batch.mean(dim=1)
                else:
                    r_global = rewards_batch.sum(dim=1)
                td_target = r_global + (1 - done_batch) * gamma * q_tot_next
                loss = nn.MSELoss()(q_tot, td_target.detach())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()

                update_count += 1
                writer.add_scalar("train/loss", loss.item(), update_count)
                writer.add_scalar("train/epsilon", eps, global_step)

                if global_step % 1000 == 0:
                    for agent in agents:
                        agent.update_target()
                    target_mixer.load_state_dict(mixer.state_dict())

            if done:
                break

        completion = env.get_completion_ratio()
        writer.add_scalar("episode_reward", ep_reward, ep + 1)
        writer.add_scalar("completion_ratio", completion, ep + 1)
        writer.add_scalar("episode_steps", steps, ep + 1)
        writer.add_scalar("train/episode_reward", ep_reward, ep + 1)
        writer.add_scalar("train/completion_ratio", completion, ep + 1)
        writer.add_scalar("train/episode_steps", steps, ep + 1)

        if (ep + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_ep{ep+1}.pth")
            _save_checkpoint(ckpt_path, agents, mixer, obs_dim, state_dim, action_n, metadata)
            print(f"[Checkpoint] Saved {algo_name} at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = _run_periodic_eval(
                agents,
                env,
                uses_rm=uses_rm,
                rm_kwargs=rm_kwargs,
                difficulty=metadata.get("difficulty", "easy"),
                algo_name=algo_name,
                eval_episodes=eval_episodes,
                max_steps=max_steps,
                seed=metadata.get("eval_seed", metadata.get("seed", 42)),
                json_out=os.path.join(eval_dir, f"eval_ep{ep+1}.json"),
                metadata=metadata,
            )
            eval_summary = eval_payload["summary"]
            writer.add_scalar("eval/mean_completion_ratio", eval_summary["mean_completion_ratio"], ep + 1)
            writer.add_scalar("eval/success_rate", eval_summary["success_rate"], ep + 1)
            writer.add_scalar("eval/mean_reward", eval_summary["mean_reward"], ep + 1)
            writer.add_scalar("eval/mean_steps", eval_summary["mean_steps"], ep + 1)

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
                _save_checkpoint(best_ckpt_path, agents, mixer, obs_dim, state_dim, action_n, best_meta)
                print(
                    f"[Best] Saved {algo_name} best checkpoint at ep {ep+1} | "
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
    _save_checkpoint(final_path, agents, mixer, obs_dim, state_dim, action_n, final_meta)
    writer.close()
    return agents, mixer


# Keep backward-compatible alias
qmix_rm = rmix = None


def rmix(
    env,
    rm,
    episodes=1000,
    batch_size=64,
    buffer_capacity=5000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=50000,
    lr=5e-4,
    gamma=0.99,
    log_dir="runs",
    exp_name="rmix",
    save_interval=100,
    max_steps=400,
    variant="rmix_v2",
    state_mode=None,
    use_action_mask=None,
    use_double_q=None,
    global_reward_mode=None,
    reward_profile=None,
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    if variant == "legacy":
        state_mode = state_mode or "legacy"
        use_action_mask = False if use_action_mask is None else use_action_mask
        use_double_q = False if use_double_q is None else use_double_q
        global_reward_mode = global_reward_mode or "sum"
        reward_profile = reward_profile or "legacy"
    elif variant == "rmix_v2_medium":
        state_mode = state_mode or "rmix_v2"
        use_action_mask = True if use_action_mask is None else use_action_mask
        use_double_q = True if use_double_q is None else use_double_q
        global_reward_mode = global_reward_mode or "mean"
        reward_profile = reward_profile or "medium_adapted"
    else:
        state_mode = state_mode or "rmix_v2"
        use_action_mask = True if use_action_mask is None else use_action_mask
        use_double_q = True if use_double_q is None else use_double_q
        global_reward_mode = global_reward_mode or "sum"
        reward_profile = reward_profile or "stabilized"

    return _train_qmix_family(
        env,
        rm,
        uses_rm=True,
        algo_name="rmix",
        variant=variant,
        episodes=episodes,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        lr=lr,
        gamma=gamma,
        log_dir=log_dir,
        exp_name=exp_name,
        save_interval=save_interval,
        max_steps=max_steps,
        state_mode=state_mode,
        use_action_mask=use_action_mask,
        use_double_q=use_double_q,
        global_reward_mode=global_reward_mode,
        reward_profile=reward_profile,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        checkpoint_meta=checkpoint_meta,
    )


qmix_rm = rmix


def qmix(
    env,
    episodes=1000,
    batch_size=64,
    buffer_capacity=5000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=50000,
    lr=5e-4,
    gamma=0.99,
    log_dir="runs",
    exp_name="qmix",
    save_interval=100,
    max_steps=400,
    state_mode="legacy",
    use_action_mask=True,
    use_double_q=False,
    global_reward_mode="sum",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    return _train_qmix_family(
        env,
        rm=None,
        uses_rm=False,
        algo_name="qmix",
        variant="baseline",
        episodes=episodes,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        lr=lr,
        gamma=gamma,
        log_dir=log_dir,
        exp_name=exp_name,
        save_interval=save_interval,
        max_steps=max_steps,
        state_mode=state_mode,
        use_action_mask=use_action_mask,
        use_double_q=use_double_q,
        global_reward_mode=global_reward_mode,
        reward_profile="legacy",
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        checkpoint_meta=checkpoint_meta,
    )
