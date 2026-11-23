#!/usr/bin/env python3
"""
Multi-Agent Q-Learning with Reward Machine (MAQL-RM)
结合奖励塑形和Reward Machine的多智能体Q学习算法

核心特性：
1. 独立Q学习：每个agent维护独立的Q表
2. Reward Machine：基于任务进度的密集奖励
3. 奖励塑形：距离奖励 + 任务完成奖励
4. ε-greedy探索：从高探索率逐渐衰减
"""

import os
import numpy as np
import pickle
from collections import defaultdict, deque
import random as pyrandom
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import build_obs_list_for_agents


class MAQLAgent:
    """单个Q学习智能体"""

    def __init__(self, agent_id, action_n=6, lr=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.action_n = action_n
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q表：使用defaultdict自动初始化
        self.q_table = defaultdict(lambda: np.zeros(action_n))

    def get_state_key(self, obs):
        """将观测转换为状态键（离散化）- 优化版"""
        pos = tuple(obs[:2].astype(int))

        # 其他agent位置（用于避免碰撞）
        other_agents = obs[2:8].reshape(3, 2)  # 3个其他agent
        nearby_count = sum(1 for p in other_agents
                          if abs(p[0]-pos[0]) + abs(p[1]-pos[1]) <= 2)

        # 任务地图（更细粒度：0/1/2+）
        map_start = 8
        map_data = obs[map_start:]
        grid_size = 36

        if len(map_data) >= 5 * grid_size:
            task_maps = map_data[:5*grid_size].reshape(5, grid_size)
            task_counts = tuple(min(2, int(m.sum())) for m in task_maps)
        else:
            task_counts = tuple(min(2, int(map_data[i*30:(i+1)*30].sum()))
                               for i in range(min(5, len(map_data)//30)))

        return (pos, nearby_count, task_counts)

    def act(self, obs, epsilon=None):
        """ε-greedy策略选择动作"""
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(self.action_n)

        state_key = self.get_state_key(obs)
        q_values = self.q_table[state_key]
        return np.argmax(q_values)

    def update(self, obs, action, reward, next_obs, done):
        """Q学习更新"""
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)

        # Q-learning更新公式
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * next_max_q

        # 更新Q值
        self.q_table[state_key][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self, ep, episodes):
        """分段ε衰减：前50%快速探索，后50%慢速衰减保持稳定"""
        progress = ep / episodes
        if progress < 0.3:
            # 前30%：快速衰减到0.3
            target = 0.3
            self.epsilon = max(target, self.epsilon * 0.99)
        elif progress < 0.7:
            # 30%-70%：慢速衰减到epsilon_end，保持探索
            self.epsilon = max(self.epsilon_end, self.epsilon * 0.9995)
        else:
            # 后30%：固定在epsilon_end
            self.epsilon = self.epsilon_end


def compute_distance_reward(env, agent_idx):
    """计算距离奖励+碰撞惩罚"""
    robot = env.robots[agent_idx]
    pos = robot["pos"]
    tasks = robot["tasks"]

    # 找到该agent能执行的待执行任务位置
    target_positions = []
    if "distribute_floor" in tasks:
        target_positions.extend(list(env.floors_to_distribute))
    if "vibrate_floor" in tasks:
        target_positions.extend(list(env.floors_to_vibrate))
    if "level_floor" in tasks:
        target_positions.extend(list(env.floors_to_level))
    if "cover_surface" in tasks:
        target_positions.extend(list(env.floors_to_cover))

    if not target_positions:
        return 0.0

    # 距离奖励
    min_distance = min(abs(pos[0] - t[0]) + abs(pos[1] - t[1])
                      for t in target_positions)
    max_distance = env.width + env.height
    distance_reward = -min_distance / max_distance

    # 碰撞惩罚：检查是否与其他agent太近
    collision_penalty = 0.0
    for j, other_robot in enumerate(env.robots):
        if j != agent_idx:
            other_pos = other_robot["pos"]
            dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
            if dist <= 1:  # 相邻或重叠
                collision_penalty -= 0.5

    return distance_reward + collision_penalty


def maql_rm(env, rm,
            episodes=2000,
            max_steps=200,
            lr=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            distance_reward_weight=0.3,
            replay_buffer_size=5000,
            replay_batch=32,
            replay_freq=10,
            log_dir="runs",
            exp_name="maql_rm",
            save_interval=200,
            use_rm_obs=True):
    """
    Multi-Agent Q-Learning with Reward Machine

    Args:
        env: 环境实例
        rm: Reward Machine实例
        episodes: 训练轮数
        max_steps: 每个episode最大步数
        lr: 学习率
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减
        distance_reward_weight: 距离奖励权重
        log_dir: TensorBoard日志目录
        exp_name: 实验名称
        save_interval: 模型保存间隔
        use_rm_obs: 是否使用RM观测

    Returns:
        agents: 训练好的智能体列表
    """

    # 创建保存目录
    os.makedirs("checkpoints/maql_rm", exist_ok=True)
    runs_save_path = os.path.join(log_dir, "maql_rm",
                                  f"{exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=runs_save_path)

    # 初始化环境和智能体
    env.reset()
    rm.reset(env=env)
    num_agents = len(env.robots)
    action_n = 6  # UP/DOWN/LEFT/RIGHT/PERFORM/STAY

    # 创建智能体
    agents = [MAQLAgent(i, action_n=action_n, lr=lr, gamma=gamma,
                       epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                       epsilon_decay=epsilon_decay)
              for i in range(num_agents)]

    # 获取观测维度
    obs0 = build_obs_list_for_agents(env, rm=rm if use_rm_obs else None)
    obs_dim = len(obs0[0])

    print(f"\n{'='*70}")
    print(f"  MAQL-RM 训练（优化版）")
    print(f"{'='*70}")
    print(f"观测维度: {obs_dim}")
    print(f"使用RM观测: {use_rm_obs}")
    print(f"智能体数量: {num_agents}")
    print(f"动作空间: {action_n}")
    print(f"学习率: {lr}")
    print(f"初始ε: {epsilon_start} → 最终ε: {epsilon_end}")
    print(f"距离奖励权重: {distance_reward_weight}")
    print(f"经验回放: buffer={replay_buffer_size}, batch={replay_batch}, freq={replay_freq}")
    print(f"{'='*70}\n")

    # 经验回放缓冲区（每个agent独立）
    replay_buffers = [deque(maxlen=replay_buffer_size) for _ in range(num_agents)]
    global_step = 0

    # 训练循环
    for ep in tqdm(range(episodes), desc="Training MAQL-RM"):
        env.reset()
        rm.reset(env=env)
        obs_list = build_obs_list_for_agents(env, rm=rm if use_rm_obs else None)

        done = False
        ep_reward = 0.0
        steps = 0
        first_done_steps = None

        # Episode循环
        for step in range(max_steps):
            # 每个agent选择动作
            actions = [agents[i].act(obs_list[i]) for i in range(num_agents)]

            # 环境执行动作
            rewards_env, infos = env.step(actions)
            next_obs_list = build_obs_list_for_agents(env, rm=rm if use_rm_obs else None)

            # RM奖励
            events = [info.get("status", None) for info in infos]
            rm_rewards, _, _ = rm.step(events, env=env)

            # 检查全部完成
            all_done_now = env.is_all_done()
            global_bonus = 0.0
            if all_done_now:
                global_bonus = rm.trigger_all_done_reward()
                done = True
                if first_done_steps is None:
                    first_done_steps = steps + 1

            # 动态距离奖励权重：前期大，后期小
            dyn_weight = distance_reward_weight * (1.0 - 0.5 * ep / episodes)

            # 计算总奖励
            total_rewards = []
            for i in range(num_agents):
                r_env = rewards_env[i] if i < len(rewards_env) else 0.0
                r_rm = rm_rewards[i] if i < len(rm_rewards) else 0.0
                r_distance = compute_distance_reward(env, i) * dyn_weight
                r_total = r_env + r_rm + r_distance + global_bonus
                total_rewards.append(r_total)

            # 存入经验回放缓冲区 + 即时更新
            for i in range(num_agents):
                s_key = agents[i].get_state_key(obs_list[i])
                ns_key = agents[i].get_state_key(next_obs_list[i])
                replay_buffers[i].append((s_key, actions[i], total_rewards[i], ns_key, done))
                agents[i].update(obs_list[i], actions[i], total_rewards[i],
                                 next_obs_list[i], done)

            # 经验回放更新
            global_step += 1
            if global_step % replay_freq == 0:
                for i in range(num_agents):
                    if len(replay_buffers[i]) >= replay_batch:
                        batch = pyrandom.sample(replay_buffers[i], replay_batch)
                        for (s, a, r, ns, d) in batch:
                            cur_q = agents[i].q_table[s][a]
                            if d:
                                tgt = r
                            else:
                                tgt = r + agents[i].gamma * np.max(agents[i].q_table[ns])
                            agents[i].q_table[s][a] += agents[i].lr * (tgt - cur_q)

            # 更新状态
            obs_list = next_obs_list
            ep_reward += sum(total_rewards)
            steps += 1

            if done:
                break

        # 分段ε衰减
        for agent in agents:
            agent.decay_epsilon(ep, episodes)

        # 记录指标
        completion_ratio = env.get_completion_ratio()
        writer.add_scalar("episode_reward", ep_reward, ep+1)
        writer.add_scalar("train/episode_length", steps, ep+1)
        writer.add_scalar("completion_ratio", completion_ratio, ep+1)
        writer.add_scalar("train/success", 1.0 if done else 0.0, ep+1)
        writer.add_scalar("train/epsilon", agents[0].epsilon, ep+1)
        if first_done_steps is not None:
            writer.add_scalar("train/steps_to_complete", first_done_steps, ep+1)

        # 打印进度
        if (ep+1) % 10 == 0:
            print(f"\nEp {ep+1}/{episodes} | completion={completion_ratio:.3f} | "
                  f"steps={steps} | reward={ep_reward:.2f} | done={done} | "
                  f"ε={agents[0].epsilon:.3f}", flush=True)

        # 保存模型
        if (ep+1) % save_interval == 0:
            checkpoint = {
                "episode": ep+1,
                "agents": [{"q_table": dict(agent.q_table),
                           "epsilon": agent.epsilon}
                          for agent in agents],
                "config": {
                    "lr": lr,
                    "gamma": gamma,
                    "epsilon_decay": epsilon_decay,
                    "use_rm_obs": use_rm_obs
                }
            }
            save_path = f"checkpoints/maql_rm/checkpoint_ep{ep+1}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"[Checkpoint] Saved at episode {ep+1}", flush=True)

    # 保存最终模型
    final_checkpoint = {
        "episode": episodes,
        "agents": [{"q_table": dict(agent.q_table),
                   "epsilon": agent.epsilon}
                  for agent in agents],
        "config": {
            "lr": lr,
            "gamma": gamma,
            "epsilon_decay": epsilon_decay,
            "use_rm_obs": use_rm_obs
        }
    }
    final_path = f"checkpoints/maql_rm/final_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)

    writer.close()
    print(f"\n{'='*70}")
    print(f"  训练完成！")
    print(f"{'='*70}")
    print(f"最终模型保存至: {final_path}")

    return agents
