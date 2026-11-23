import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from marl.agents import QTRANAgent, QTRANBase
from marl.buffer import ReplayBuffer
from utils.baseline_eval import evaluate_policy, log_eval_scalars, make_greedy_value_selector
from utils.utils import build_obs_list_for_agents, build_state, device, get_action_masks



def _save_checkpoint(path, agents, qtran_net, meta, obs_dim, state_dim, action_n):
    payload = {
        "num_agents": len(agents),
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "action_n": action_n,
        "qtran_net": qtran_net.state_dict(),
        "meta": dict(meta),
    }
    for i, agent in enumerate(agents):
        payload[f"agent_{i}"] = agent.net.state_dict()
    torch.save(payload, path)


def _sample_valid_action(mask):
    valid = torch.nonzero(torch.as_tensor(mask, dtype=torch.bool), as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0
    return int(valid[torch.randint(valid.numel(), (1,))].item())


def _masked_argmax(q_values, mask):
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device).unsqueeze(0)
    return int(q_values.masked_fill(~mask_t, -1e9).argmax(dim=1).item())



def qtran_baseline(
    env,
    *,
    episodes=3000,
    batch_size=128,
    buffer_capacity=50000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=50000,
    lr=5e-4,
    gamma=0.99,
    lambda_opt=0.1,
    lambda_nopt=0.1,
    log_dir="runs",
    exp_name="qtran",
    save_interval=500,
    max_steps=300,
    warmup_steps=1000,
    difficulty="easy",
    eval_interval=100,
    eval_episodes=10,
    checkpoint_meta=None,
):
    algo_name = "qtran"
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

    agents = [QTRANAgent(obs_dim, action_n, device=device) for _ in range(num_agents)]
    qtran_net = QTRANBase(num_agents=num_agents, state_dim=state_dim, action_dim=action_n).to(device)
    qtran_target = QTRANBase(num_agents=num_agents, state_dim=state_dim, action_dim=action_n).to(device)
    qtran_target.load_state_dict(qtran_net.state_dict())

    params = []
    for agent in agents:
        params.extend(list(agent.net.parameters()))
    params.extend(list(qtran_net.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    metadata = dict(checkpoint_meta or {})
    metadata.update(
        {
            "algo": algo_name,
            "difficulty": difficulty,
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "action_n": action_n,
            "uses_rm": False,
            "max_steps": max_steps,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "lambda_opt": lambda_opt,
            "lambda_nopt": lambda_nopt,
            "use_action_mask": True,
        }
    )

    global_step = 0
    update_count = 0
    best_eval = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_{timestamp}_best_by_completion.pth")

    for ep in tqdm(range(episodes), desc="Training QTRAN"):
        env.reset()
        obs_list = build_obs_list_for_agents(env)
        state = build_state(env=env)
        ep_reward = 0.0
        steps = 0
        done = False
        first_done_steps = None

        for _ in range(max_steps):
            eps = max(eps_end, eps_start - (global_step / max(eps_decay, 1)) * (eps_start - eps_end))
            action_masks = get_action_masks(env)
            actions_int = []
            for i, agent in enumerate(agents):
                if torch.rand(1).item() < eps:
                    action = _sample_valid_action(action_masks[i])
                else:
                    obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action = _masked_argmax(agent(obs_tensor), action_masks[i])
                actions_int.append(action)

            rewards_env, _infos = env.step(actions_int)
            next_obs_list = build_obs_list_for_agents(env)
            next_state = build_state(env=env)
            next_action_masks = get_action_masks(env)
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
                action_masks=action_masks,
                next_action_masks=next_action_masks,
            )

            obs_list = next_obs_list
            state = next_state
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
                    state_batch,
                    next_state_batch,
                    action_mask_batch,
                    next_action_mask_batch,
                ) = buffer.sample_torch(batch_size, device, return_state=True, return_masks=True)

                q_taken_list = []
                for i, agent in enumerate(agents):
                    q_values = agent(obs_batch[i])
                    q_taken = q_values.gather(1, actions_batch[:, i].unsqueeze(1)).squeeze(1)
                    q_taken_list.append(q_taken)
                q_taken_stack = torch.stack(q_taken_list, dim=1)

                actions_onehot = F.one_hot(actions_batch, num_classes=action_n).float().reshape(batch_size, -1)
                q_joint_taken, v_taken = qtran_net(state_batch, actions_onehot)

                masked_online_q = [
                    agent(obs_batch[i]).masked_fill(~action_mask_batch[:, i, :], -1e9)
                    for i, agent in enumerate(agents)
                ]
                greedy_actions = torch.stack(
                    [torch.argmax(masked_online_q[i], dim=1) for i in range(num_agents)], dim=1
                )
                greedy_actions_onehot = F.one_hot(greedy_actions, num_classes=action_n).float().reshape(batch_size, -1)
                q_joint_greedy, v_greedy = qtran_net(state_batch, greedy_actions_onehot)
                q_greedy_taken = torch.stack(
                    [masked_online_q[i].gather(1, greedy_actions[:, i].unsqueeze(1)).squeeze(1) for i in range(num_agents)],
                    dim=1,
                )

                with torch.no_grad():
                    next_q_values = [
                        agent.target(next_obs_batch[i]).masked_fill(~next_action_mask_batch[:, i, :], -1e9)
                        for i, agent in enumerate(agents)
                    ]
                    next_actions_max = torch.stack(
                        [torch.argmax(next_q_values[i], dim=1) for i in range(num_agents)], dim=1
                    )
                    next_actions_onehot = F.one_hot(next_actions_max, num_classes=action_n).float().reshape(batch_size, -1)
                    q_joint_next, _ = qtran_target(next_state_batch, next_actions_onehot)
                    td_target = rewards_batch.sum(dim=1, keepdim=True) + (1 - done_batch.unsqueeze(1)) * gamma * q_joint_next

                q_ind_taken = q_taken_stack.sum(dim=1, keepdim=True)
                q_ind_greedy = q_greedy_taken.sum(dim=1, keepdim=True)
                loss_td = nn.MSELoss()(q_joint_taken, td_target.detach())
                loss_opt = nn.MSELoss()(q_joint_greedy - v_greedy, q_ind_greedy)
                nopt_violation = q_ind_taken - (q_joint_taken - v_taken)
                loss_nopt = torch.mean(torch.clamp(nopt_violation, min=0.0) ** 2)
                loss = loss_td + lambda_opt * loss_opt + lambda_nopt * loss_nopt

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()

                update_count += 1
                writer.add_scalar("train/loss_td", loss_td.item(), update_count)
                writer.add_scalar("train/loss_opt", loss_opt.item(), update_count)
                writer.add_scalar("train/loss_nopt", loss_nopt.item(), update_count)
                writer.add_scalar("train/epsilon", eps, global_step)

                if global_step % 1000 == 0:
                    qtran_target.load_state_dict(qtran_net.state_dict())
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
            _save_checkpoint(ckpt_path, agents, qtran_net, metadata, obs_dim, state_dim, action_n)
            print(f"[Checkpoint] Saved QTRAN at episode {ep+1}", flush=True)

        if eval_interval and (ep + 1) % eval_interval == 0:
            eval_payload = evaluate_policy(
                action_selector=make_greedy_value_selector(agents),
                difficulty=difficulty,
                episodes=eval_episodes,
                seed=metadata.get("seed", 42),
                algo=algo_name,
                max_steps=max_steps,
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
                _save_checkpoint(best_ckpt_path, agents, qtran_net, best_meta, obs_dim, state_dim, action_n)
                print(
                    f"[Best] Saved QTRAN best checkpoint at ep {ep+1} | "
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
    _save_checkpoint(final_path, agents, qtran_net, final_meta, obs_dim, state_dim, action_n)
    writer.close()
    return agents, qtran_net
