import os
import math
import torch
import numpy as np
from marl.agents import MAPPOActor, MAPPOCritic
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from datetime import datetime
from tqdm import tqdm

def ippo(env, rm,
         episodes=2000,
         rollout_len=500,
         ppo_epochs=4,
         mini_batch_size=500,
         clip_eps=0.2,
         lr=1e-4,
         gamma=0.99,
         lam=0.95,
         grad_clip=0.5,
         advantage_norm=True,
         value_noise_std=0.5,
         epsilon_start=0.3,
         epsilon_end=0.05,
         epsilon_decay=0.998,
         log_dir="runs",
         exp_name="ippo",
         save_interval=200,
         use_rm_obs=True):

    os.makedirs("checkpoints/ippo", exist_ok=True)
    runs_save_path = os.path.join(log_dir, "ippo", f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=runs_save_path)

    env.reset()
    rm.reset(env=env)
    num_agents = len(env.robots)

    # 构建观测（可选包含RM状态）
    obs0 = build_obs_list_for_agents(env, rm=rm if use_rm_obs else None)
    obs_dim = len(obs0[0])
    action_n = 6

    print(f"[IPPO] Obs dim: {obs_dim}, Use RM obs: {use_rm_obs}")

    # 每个智能体独立actor和critic
    actors = [MAPPOActor(obs_dim, action_n).to(device) for _ in range(num_agents)]
    critics = [MAPPOCritic(obs_dim).to(device) for _ in range(num_agents)]  # critic输入obs而不是全局state
    actor_optims = [optim.Adam(a.parameters(), lr=lr) for a in actors]
    critic_optims = [optim.Adam(c.parameters(), lr=lr) for c in critics]

    total_steps = 0
    warmup_steps = np.zeros(num_agents)  # 每个智能体的warm-up步数
    epsilon = epsilon_start  # ε-greedy exploration


    for ep in tqdm(range(episodes), desc="Training IPPO"):
        # 每个智能体独立存储
        rollout = [{
            "obs": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": []
        } for _ in range(num_agents)]

        env.reset()
        rm.reset(env=env)
        done = False
        ep_reward = 0.0
        steps = 0
        final_reward = np.zeros(num_agents, dtype=np.int32)
        first_done_steps = None

        for t in range(rollout_len):
            obs_list = build_obs_list_for_agents(env, rm=rm if use_rm_obs else None)
            actions = np.zeros(num_agents, dtype=np.int32) + 5
            logps = []
            values = []
            per_agent_rewards = []

            # 每个智能体独立决策
            for i in range(num_agents):
                obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    value = critics[i](obs_tensor).squeeze().detach()
                    if value_noise_std > 0:
                        noise = torch.randn_like(value) * value_noise_std
                        value += noise

                # ε-greedy exploration
                if np.random.rand() < epsilon:
                    # Random action
                    actions[i] = np.random.randint(0, action_n)
                    # Still need logprob for the random action
                    probs = actors[i](obs_tensor)
                    dist = Categorical(probs)
                    logps.append(float(dist.log_prob(torch.tensor(actions[i], device=device)).item()))
                else:
                    # Policy action
                    probs = actors[i](obs_tensor)
                    dist = Categorical(probs)
                    a = dist.sample()
                    actions[i] = int(a.item())
                    logps.append(float(dist.log_prob(a).item()))

                values.append(float(value.item()))

            # Environment Step (修复：移到循环外)
            rewards_env, infos = env.step(actions)
            events = [info.get("status", None) for info in infos]
            rm_rewards, _, _ = rm.step(events, env=env)

            # 检查全部完成，触发全局完成奖励
            all_done_now = env.is_all_done()
            global_bonus = 0.0
            if all_done_now:
                global_bonus = rm.trigger_all_done_reward()

            # Calculate Rewards
            for i in range(num_agents):
                r = (rewards_env[i] if i < len(rewards_env) else 0.0) + \
                    (rm_rewards[i] if i < len(rm_rewards) else 0.0) + \
                    global_bonus
                per_agent_rewards.append(r)

            done = all_done_now
            if done and first_done_steps is None:
                first_done_steps = steps + 1

            total_steps += 1

            # 存储每个智能体自己的数据
            for i in range(num_agents):
                rollout[i]["obs"].append(obs_list[i])
                rollout[i]["actions"].append(actions[i])
                rollout[i]["logprobs"].append(logps[i])
                rollout[i]["values"].append(values[i])
                rollout[i]["rewards"].append(per_agent_rewards[i])
                rollout[i]["dones"].append(1.0 if done else 0.0)

            ep_reward += sum(per_agent_rewards)
            steps += 1

            if done:
                break

        # 每个智能体独立PPO更新
        for i in range(num_agents):
            obs_arr = torch.tensor(np.array(rollout[i]["obs"]), dtype=torch.float32, device=device)  # (T, obs_dim)
            actions_arr = torch.tensor(rollout[i]["actions"], dtype=torch.long, device=device)       # (T,)
            logprobs_arr = torch.tensor(rollout[i]["logprobs"], dtype=torch.float32, device=device)  # (T,)
            values_arr = torch.tensor(rollout[i]["values"], dtype=torch.float32, device=device)      # (T,)
            rewards_arr = torch.tensor(rollout[i]["rewards"], dtype=torch.float32, device=device)    # (T,)
            dones_arr = torch.tensor(rollout[i]["dones"], dtype=torch.float32, device=device)        # (T,)

            # 下一个状态的value
            if rollout[i]["dones"][-1] == 1.0:
                next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
            else:
                next_obs = torch.tensor(rollout[i]["obs"][-1], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    next_value = critics[i](next_obs).squeeze().detach()

            advantages, returns = compute_gae(
                rewards_arr.unsqueeze(1), values_arr, next_value, dones_arr, gamma=gamma, lam=lam
            )
            advantages = advantages.squeeze(1)
            returns = returns.squeeze(1)

            if advantage_norm:
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False) + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            T = len(rewards_arr)
            idxs = np.arange(T)
            for _ in range(ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, T, mini_batch_size):
                    mb_idx = idxs[start:start+mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    # --- warm-up: 动态调整学习率 ---
                    cur_lr = get_warmup_lr(lr, total_steps, warmup_steps[i])
                    for param_group in actor_optims[i].param_groups:
                        param_group['lr'] = cur_lr
                    for param_group in critic_optims[i].param_groups:
                        param_group['lr'] = cur_lr

                    mb_obs = obs_arr[mb_idx]
                    mb_actions = actions_arr[mb_idx]
                    mb_old_logp = logprobs_arr[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]

                    # critic update
                    values_pred = critics[i](mb_obs).squeeze()
                    critic_loss = F.mse_loss(values_pred, mb_ret)
                    critic_optims[i].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critics[i].parameters(), grad_clip)
                    critic_optims[i].step()

                    # actor update
                    probs = actors[i](mb_obs)
                    dist = Categorical(probs)
                    logp_now = dist.log_prob(mb_actions)
                    ratio = torch.exp(logp_now - mb_old_logp)
                    obj1 = ratio * mb_adv
                    obj2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                    actor_loss = -torch.min(obj1, obj2).mean()
                    actor_optims[i].zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actors[i].parameters(), grad_clip)
                    actor_optims[i].step()

            writer.add_scalar(f"agent{i}/actor_loss", actor_loss, ep+1)
            writer.add_scalar(f"agent{i}/critic_loss", critic_loss, ep+1)
            writer.add_scalar(f"agent{i}/average_return", returns.mean().item(), ep+1)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        writer.add_scalar("train/epsilon", epsilon, ep+1)

        # 三个核心指标
        completion_ratio = env.get_completion_ratio()
        writer.add_scalar("episode_reward", ep_reward, ep+1)
        writer.add_scalar("train/episode_length", steps, ep+1)
        writer.add_scalar("completion_ratio", completion_ratio, ep+1)
        writer.add_scalar("train/success", 1.0 if done else 0.0, ep+1)
        if first_done_steps is not None:
            writer.add_scalar("train/steps_to_complete", first_done_steps, ep+1)

        if (ep+1) % 10 == 0:
            print(f"\nEp {ep+1}/{episodes} | completion={completion_ratio:.3f} | steps={steps} | reward={ep_reward:.2f} | done={done}", flush=True)

        if (ep+1) % save_interval == 0:
            ckpt = {
                "actors": [a.state_dict() for a in actors],
                "critics": [c.state_dict() for c in critics],
                "episode": ep+1
            }
            torch.save(ckpt, f"checkpoints/ippo/ippo_ep{ep+1}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth")

    writer.close()
    return actors, critics

# def ippo(env, rm,
#             episodes=500,
#             rollout_len=200,
#             ppo_epochs=8,
#             mini_batch_size=64,
#             clip_eps=0.2,
#             lr=3e-4,
#             gamma=0.99,
#             lam=0.95,
#             grad_clip=0.5,
#             advantage_norm=True,
#             value_noise_std=0.0,
#             log_dir="runs",
#             exp_name="ippo",
#             save_interval=200):

#     """
#     Independent PPO (IPPO) implementation for GridAreaEnv (integer actions 0..5).
#     每个 agent 有独立的 actor + critic(local observation -> value)。
#     使用 per-agent GAE 与 per-agent PPO 更新。
#     """
#     os.makedirs("checkpoints/ippo", exist_ok=True)
#     writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))

#     env.reset()
#     rm.reset()

#     num_agents = len(env.robots)
#     obs0 = build_obs_list_from_env(env)
#     obs_dim = len(obs0[0])
#     action_n = 6  # 0..5

#     # 为每个 agent 创建独立 actor + critic + optimizer
#     actors = [MAPPOActor(obs_dim, action_n).to(device) for _ in range(num_agents)]
#     critics = [MAPPOCritic(obs_dim).to(device) for _ in range(num_agents)]
#     actor_opts = [optim.Adam(a.parameters(), lr=lr) for a in actors]
#     critic_opts = [optim.Adam(c.parameters(), lr=lr) for c in critics]

#     total_steps = 0

#     for ep in tqdm(range(episodes), desc="Training IPPO"):
#         env.reset(); rm.reset()
#         done = False
#         ep_reward = 0.0
#         steps = 0

#         # Rollout 存储：为每个 agent 分别存储
#         rollout = {
#             "obs": [[] for _ in range(num_agents)],      # 每个元素为 list length T of obs arrays
#             "actions": [[] for _ in range(num_agents)],
#             "logprobs": [[] for _ in range(num_agents)],
#             "values": [[] for _ in range(num_agents)],
#             "rewards": [[] for _ in range(num_agents)],
#             "dones": []
#         }

#         for t in range(rollout_len):
#             obs_list = build_obs_list_from_env(env)
#             # 逐 agent 采样
#             actions = []
#             logps = []
#             values = []
#             for i in range(num_agents):
#                 obs_i = torch.tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)  # (1,obs_dim)
#                 with torch.no_grad():
#                     probs = actors[i](obs_i)   # (1, action_n)
#                     dist = Categorical(probs)
#                     a = dist.sample()
#                     lp = dist.log_prob(a)
#                     v = critics[i](obs_i).squeeze(0)  # scalar
#                 actions.append(int(a.item()))
#                 logps.append(float(lp.item()))
#                 if value_noise_std > 0:
#                     noise = torch.randn_like(v) * value_noise_std
#                     v = v + noise
#                 values.append(float(v.item()))
            
#                 # 存入个体 buffer
#                 rollout["obs"][i].append(obs_list[i])
#                 rollout["actions"][i].append(int(a.item()))
#                 rollout["logprobs"][i].append(float(lp.item()))
#                 rollout["values"][i].append(float(v.item()))

#             # env step（直接传 int 动作列表）
#             rewards_env, infos = env.step(actions)
#             events = [info.get("status", None) for info in infos]
#             rm_rewards, _ = rm.step(events)
#             per_agent_rewards = [(rewards_env[i] if i < len(rewards_env) else 0.0) + (rm_rewards[i] if i < len(rm_rewards) else 0.0)
#                                  for i in range(num_agents)]
#             for i in range(num_agents):
#                 rollout["rewards"][i].append(float(per_agent_rewards[i]))

#             rollout["dones"].append(0.0)
#             ep_reward += sum(per_agent_rewards)
#             steps += 1
#             total_steps += 1

#             # 终止检查（与 env._get_state 里字段一致）
#             st = env._get_state()
#             finish_cond = (
#                 len(st.get("walls_to_distribute", [])) == 0 and
#                 len(st.get("walls_to_vibrate", [])) == 0 and
#                 len(st.get("floors_to_distribute", [])) == 0 and
#                 len(st.get("floors_to_vibrate", [])) == 0 and
#                 len(st.get("floors_to_level", [])) == 0 and
#                 len(st.get("floors_to_cover", [])) == 0
#             )
#             if finish_cond:
#                 rollout["dones"][-1] = 1.0
#                 # simple finishing bonus 平均分给 agents
#                 bonus = 100.0 / num_agents
#                 for i in range(num_agents):
#                     rollout["rewards"][i][-1] += bonus
#                 done = True

#             if done:
#                 break

#         # 每个 agent 单独进行 GAE 与 PPO 更新
#         for i in range(num_agents):
#             # 转为 tensor
#             obs_i = torch.tensor(np.array(rollout["obs"][i], dtype=np.float32), device=device)   # (T, obs_dim)
#             actions_i = torch.tensor(np.array(rollout["actions"][i], dtype=np.int64), device=device)  # (T,)
#             old_logp_i = torch.tensor(np.array(rollout["logprobs"][i], dtype=np.float32), device=device)  # (T,)
#             values_i = torch.tensor(np.array(rollout["values"][i], dtype=np.float32), device=device)  # (T,)
#             rewards_i = torch.tensor(np.array(rollout["rewards"][i], dtype=np.float32), device=device)  # (T,)
#             dones_t = torch.tensor(np.array(rollout["dones"], dtype=np.float32), device=device)  # (T,)
            
#             T = rewards_i.shape[0]
#             # next_value
#             if rollout["dones"][-1] == 1.0:
#                 next_value = torch.tensor(0.0, dtype=torch.float32, device=device)
#             else:
#                 last_obs = torch.tensor(rollout["obs"][i][-1], dtype=torch.float32, device=device).unsqueeze(0)
#                 with torch.no_grad():
#                     next_value = critics[i](last_obs).squeeze(0)

#             # GAE (per-agent)
#             advantages_i, returns_i = compute_gae(rewards_i, values_i, next_value, dones_t, gamma=gamma, lam=lam)
#             if advantage_norm:
#                 adv_mean = advantages_i.mean()
#                 adv_std = advantages_i.std(unbiased=False) + 1e-8
#                 advantages_i = (advantages_i - adv_mean) / adv_std

#             # PPO 更新多轮
#             for epoch in range(ppo_epochs):
#                 # 随机打乱并分 mini-batch
#                 indices = np.arange(T)
#                 np.random.shuffle(indices)
#                 for start in range(0, T, mini_batch_size):
#                     mb_idx = indices[start:start+mini_batch_size]
#                     if len(mb_idx) == 0: continue

#                     mb_obs = obs_i[mb_idx]             # (mb, obs_dim)
#                     mb_actions = actions_i[mb_idx]     # (mb,)
#                     mb_old_logp = old_logp_i[mb_idx]   # (mb,)
#                     mb_adv = advantages_i[mb_idx]      # (mb,)
#                     mb_ret = returns_i[mb_idx]         # (mb,)

#                     # Critic update
#                     vals_pred = critics[i](mb_obs)    # (mb,)
#                     critic_loss = F.mse_loss(vals_pred, mb_ret.detach())
#                     critic_opts[i].zero_grad()
#                     critic_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(critics[i].parameters(), grad_clip)
#                     critic_opts[i].step()
#                     last_critic_loss = float(critic_loss.item())

#                     # Actor update
#                     probs = actors[i](mb_obs)        # (mb, action_n)
#                     dist = Categorical(probs)
#                     logp_now = dist.log_prob(mb_actions)   # (mb,)
#                     ratio = torch.exp(logp_now - mb_old_logp)
#                     obj1 = ratio * mb_adv
#                     obj2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
#                     actor_loss = -torch.min(obj1, obj2).mean()

#                     actor_opts[i].zero_grad()
#                     actor_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(actors[i].parameters(), grad_clip)
#                     actor_opts[i].step()

#                     writer.add_scalar(f"train/actor{i}_loss", actor_loss, ep+1)
#                     writer.add_scalar(f"train/critic{i}_loss", critic_loss, ep+1)

#         # logging
#         writer.add_scalar("train/episode_reward", ep_reward, ep+1)
#         writer.add_scalar("train/episode_length", steps, ep+1)
        

#         if (ep+1) % 10 == 0:
#             print(f"Ep {ep+1}/{episodes}, reward={ep_reward:.2f}, steps={steps}, T_rollout={T}", flush=True)

#         if (ep+1) % save_interval == 0:
#             ckpt = {
#                 "actors": [a.state_dict() for a in actors],
#                 "critics": [c.state_dict() for c in critics],
#                 "episode": ep+1
#             }
#             torch.save(ckpt, f"checkpoints/ippo/ippo_ep{ep+1}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth")

#     writer.close()
#     return actors, critics