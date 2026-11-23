import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marl.agents import DQNAgent
from marl.buffer import ReplayBuffer
from utils.baseline_eval import evaluate_policy, log_eval_scalars, make_greedy_value_selector
from utils.utils import build_obs_list_for_agents, device, get_action_masks



def _save_checkpoint(path, agents, meta):
    payload = {
        "num_agents": len(agents),
        "obs_dim": agents[0].obs_dim,
        "action_n": agents[0].action_dim,
        "meta": dict(meta),
    }
    for i, agent in enumerate(agents):
        payload[f"agent_{i}"] = agent.net.state_dict()
    torch.save(payload, path)



def vdn_baseline(
    env,
    *,
    episodes=3000,
    batch_size=64,
    buffer_capacity=50000,
    lr=5e-4,
    gamma=0.99,
    log_dir="runs",
    exp_name="vdn",
    save_interval=500,
    warmup_steps=5000,
    grad_clip=10.0,
    max_steps=300,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=50000,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    algo_name = "vdn"
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

    agents = [DQNAgent(obs_dim, action_n, lr=lr) for _ in range(num_agents)]
    buffer = ReplayBuffer(capacity=buffer_capacity)

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "difficulty": difficulty,
            "obs_dim": obs_dim,
            "action_n": action_n,
            "uses_rm": False,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
        }
    )

    global_step = 0
    update_count = 0
    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc="Training VDN"):
        env.reset()
        obs_list = build_obs_list_for_agents(env)
        ep_reward = 0.0
        steps = 0
        done = False
        first_done_steps = None

        for _ in range(max_steps):
            eps = max(eps_end, eps_start - (global_step / max(eps_decay, 1)) * (eps_start - eps_end))
            action_masks = get_action_masks(env)
            actions_int = [
                agents[i].act(obs_list[i], eps, action_mask=action_masks[i])
                for i in range(num_agents)
            ]
            rewards_env, _infos = env.step(actions_int)
            next_obs_list = build_obs_list_for_agents(env)
            next_action_masks = get_action_masks(env)
            total_rewards = [rewards_env[i] if i < len(rewards_env) else 0.0 for i in range(num_agents)]
            done = env.is_all_done()

            buffer.push(
                obs_list,
                actions_int,
                total_rewards,
                next_obs_list,
                done,
                action_masks=action_masks,
                next_action_masks=next_action_masks,
            )
            obs_list = next_obs_list
            ep_reward += float(sum(total_rewards))
            steps += 1
            global_step += 1

            if done and first_done_steps is None:
                first_done_steps = steps

            if len(buffer) >= batch_size and global_step > warmup_steps:
                (
                    obs_batch,
                    next_obs_batch,
                    actions_batch,
                    rewards_batch,
                    done_batch,
                    _action_mask_batch,
                    next_action_mask_batch,
                ) = buffer.sample_torch(batch_size, device, return_masks=True)

                joint_next_max = torch.zeros(batch_size, device=device)
                for i in range(num_agents):
                    q_next = agents[i].target(next_obs_batch[i])
                    q_next = q_next.masked_fill(~next_action_mask_batch[:, i, :], -1e9)
                    joint_next_max += q_next.max(dim=1).values

                r_global = rewards_batch.sum(dim=1)
                td_target = r_global + (1 - done_batch) * gamma * joint_next_max

                q_taken_list = []
                for i in range(num_agents):
                    q_vals = agents[i].net(obs_batch[i])
                    acts_i = actions_batch[:, i].unsqueeze(1)
                    q_taken_list.append(q_vals.gather(1, acts_i).squeeze(1))

                q_tot = sum(q_taken_list)
                loss = nn.MSELoss()(q_tot, td_target.detach())

                for agent in agents:
                    agent.opt.zero_grad()
                loss.backward()
                for agent in agents:
                    nn_utils.clip_grad_norm_(agent.net.parameters(), grad_clip)
                    agent.opt.step()

                update_count += 1
                writer.add_scalar("train/loss", loss.item(), update_count)
                writer.add_scalar("train/epsilon", eps, global_step)

                if global_step % 1000 == 0:
                    for agent in agents:
                        agent.update_target()

            if done:
                break

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
            _save_checkpoint(ckpt_path, agents, metadata)
            print(f"[Checkpoint] Saved VDN at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = evaluate_policy(
                action_selector=make_greedy_value_selector(agents),
                difficulty=difficulty,
                episodes=eval_episodes,
                seed=metadata.get("eval_seed", metadata.get("seed", 42)),
                algo=algo_name,
                max_steps=max_steps,
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
                _save_checkpoint(best_ckpt_path, agents, best_meta)
                print(
                    f"[Best] Saved VDN best checkpoint at ep {ep+1} | "
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
    _save_checkpoint(final_path, agents, final_meta)
    writer.close()
    return agents
