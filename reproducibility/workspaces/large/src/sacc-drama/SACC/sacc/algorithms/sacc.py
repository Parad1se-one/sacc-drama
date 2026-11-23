"""Standalone SACC trainer for DRAMA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sacc.algorithms.networks import CategoricalActor, CentralizedCritic
from sacc.envs.drama import DramaRMEnv
from sacc.experiments.common import append_jsonl, make_run_dir, resolve_device, set_seed, write_json
from sacc.utils.eval import evaluate_policy


@dataclass
class TrainConfig:
    difficulty: str = "easy"
    seed: int = 42
    episodes: int = 1000
    max_steps: int = 120
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    clip_eps: float = 0.2
    prev_ratio_clip_min: float | None = None
    prev_ratio_clip_max: float | None = None
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    grad_clip: float = 10.0
    entropy_coef: float = 0.01
    entropy_coef_final: float | None = None
    entropy_anneal_fraction: float = 1.0
    eval_interval: int = 50
    eval_episodes: int = 10
    eval_seed: int = 100_042
    validation_start_set: str | None = None
    train_start_protocol: str = "fixed"
    save_interval: int = 100
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 256
    disable_actor_rm: bool = False
    disable_critic_rm: bool = False
    advantage_norm: bool = True
    env_reward_scale: float = 1.0
    rm_reward_scale: float = 1.0
    team_progress_bonus_coeff: float = 0.25
    team_progress_bonus_final: float | None = None
    team_progress_anneal_fraction: float = 1.0
    train_temperature_start: float = 1.0
    train_temperature_final: float = 1.0
    train_temperature_anneal_fraction: float = 1.0
    distance_shaping_final: float | None = None
    distance_shaping_anneal_fraction: float = 1.0
    eval_policy_mode: str = "greedy"
    eval_temperature: float = 0.35
    early_stop_patience_evals: int = 0
    early_stop_min_episode: int = 0
    device: str = "auto"
    exp_name: str = "sacc"
    rm_kwargs: dict = field(default_factory=dict)


def _masked_categorical(
    probs: torch.Tensor,
    action_mask,
    device: torch.device,
    *,
    temperature: float = 1.0,
) -> Categorical:
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    masked_probs = probs.masked_fill(~mask, 0.0)
    denom = masked_probs.sum(dim=-1, keepdim=True)
    fallback = torch.zeros_like(masked_probs)
    fallback[..., -1] = 1.0
    masked_probs = torch.where(denom > 1e-8, masked_probs / denom.clamp_min(1e-8), fallback)
    if abs(float(temperature) - 1.0) > 1e-6:
        logits = torch.log(masked_probs.clamp_min(1e-8)) / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~mask, float("-inf"))
        masked_probs = torch.softmax(logits, dim=-1)
    return Categorical(masked_probs)


def _compute_gae_per_agent(
    rewards_agent: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


class SACCTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = resolve_device(config.device)
        set_seed(config.seed)

        self.env = DramaRMEnv(
            difficulty=config.difficulty,
            seed=config.seed,
            env_reward_scale=config.env_reward_scale,
            rm_reward_scale=config.rm_reward_scale,
            team_progress_bonus_coeff=config.team_progress_bonus_coeff,
            rm_kwargs=config.rm_kwargs,
            start_protocol=config.train_start_protocol,
        )
        self.num_agents = self.env.num_agents
        self.run_dir = make_run_dir(config.exp_name, config.difficulty)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self.metrics_path = self.run_dir / "training_metrics.jsonl"

        self.actors = [
            CategoricalActor(self.env.obs_dim, self.env.action_n, hidden_dim=config.actor_hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.critics = [
            CentralizedCritic(self.env.state_dim, hidden_dim=config.critic_hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.actor_optims = [optim.Adam(actor.parameters(), lr=config.actor_lr) for actor in self.actors]
        self.critic_optims = [optim.Adam(critic.parameters(), lr=config.critic_lr) for critic in self.critics]

        self.metadata = {
            "algo": "sacc",
            "variant": "per_agent_sacc",
            "difficulty": config.difficulty,
            "seed": config.seed,
            "obs_dim": self.env.obs_dim,
            "state_dim": self.env.state_dim,
            "action_n": self.env.action_n,
            "num_agents": self.num_agents,
            "uses_rm": True,
            "per_agent_rm": True,
            "critic_type": "per_agent_centralized",
            "actor_hidden_dim": config.actor_hidden_dim,
            "critic_hidden_dim": config.critic_hidden_dim,
            "disable_actor_rm": config.disable_actor_rm,
            "disable_critic_rm": config.disable_critic_rm,
            "max_steps": config.max_steps,
            "train_start_protocol": config.train_start_protocol,
            "eval_seed": config.eval_seed,
            "validation_start_set": config.validation_start_set,
            "prev_ratio_clip_min": config.prev_ratio_clip_min,
            "prev_ratio_clip_max": config.prev_ratio_clip_max,
            "rm_kwargs": dict(config.rm_kwargs),
            "env_reward_scale": config.env_reward_scale,
            "rm_reward_scale": config.rm_reward_scale,
            "team_progress_bonus_coeff": config.team_progress_bonus_coeff,
            "team_progress_bonus_final": config.team_progress_bonus_final,
            "train_temperature_start": config.train_temperature_start,
            "train_temperature_final": config.train_temperature_final,
            "entropy_coef_final": config.entropy_coef_final,
            "distance_shaping_final": config.distance_shaping_final,
            "eval_policy_mode": config.eval_policy_mode,
            "eval_temperature": config.eval_temperature,
            "device": str(self.device),
        }
        write_json(
            self.run_dir / "config.json",
            {
                "train_config": asdict(config),
                "resolved_device": str(self.device),
                "metadata": self.metadata,
            },
        )
        self.best_eval: dict | None = None
        self.no_improve_evals = 0
        self.current_entropy_coef = float(config.entropy_coef)
        self.current_train_temperature = float(config.train_temperature_start)

    @staticmethod
    def _schedule_value(start: float, end: float, episode: int, total_episodes: int, fraction: float) -> float:
        if total_episodes <= 1 or fraction <= 0.0:
            return float(end)
        schedule_episodes = max(int(round(total_episodes * fraction)), 1)
        progress = min(max((episode - 1) / max(schedule_episodes - 1, 1), 0.0), 1.0)
        return float(start + (end - start) * progress)

    def _apply_training_schedules(self, episode: int) -> None:
        self.current_entropy_coef = self._schedule_value(
            float(self.config.entropy_coef),
            float(self.config.entropy_coef if self.config.entropy_coef_final is None else self.config.entropy_coef_final),
            episode,
            self.config.episodes,
            float(self.config.entropy_anneal_fraction),
        )
        self.current_train_temperature = self._schedule_value(
            float(self.config.train_temperature_start),
            float(self.config.train_temperature_final),
            episode,
            self.config.episodes,
            float(self.config.train_temperature_anneal_fraction),
        )

        progress_bonus_coeff = self._schedule_value(
            float(self.config.team_progress_bonus_coeff),
            float(
                self.config.team_progress_bonus_coeff
                if self.config.team_progress_bonus_final is None
                else self.config.team_progress_bonus_final
            ),
            episode,
            self.config.episodes,
            float(self.config.team_progress_anneal_fraction),
        )
        distance_shaping_coeff = self._schedule_value(
            float(self.config.rm_kwargs.get("distance_shaping_coeff", 0.0)),
            float(
                self.config.rm_kwargs.get("distance_shaping_coeff", 0.0)
                if self.config.distance_shaping_final is None
                else self.config.distance_shaping_final
            ),
            episode,
            self.config.episodes,
            float(self.config.distance_shaping_anneal_fraction),
        )

        self.env.team_progress_bonus_coeff = float(progress_bonus_coeff)
        self.env.rm.config.distance_shaping_coeff = float(distance_shaping_coeff)

    def _save_checkpoint(self, path: Path, extra_meta: dict | None = None) -> Path:
        meta = dict(self.metadata)
        if extra_meta:
            meta.update(extra_meta)
        payload = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "meta": meta,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def _critic_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(state_tensor).view(-1)[0] for critic in self.critics], dim=0)

    def _run_eval(self, episode: int) -> dict:
        json_out = self.run_dir / "eval" / f"eval_ep{episode}.json"
        payload = evaluate_policy(
            actors=self.actors,
            difficulty=self.config.difficulty,
            episodes=self.config.eval_episodes,
            seed=self.config.eval_seed,
            max_steps=self.config.max_steps,
            device=self.device,
            metadata=self.metadata,
            json_out=str(json_out),
            rm_kwargs=self.config.rm_kwargs,
            policy_mode=self.config.eval_policy_mode,
            temperature=self.config.eval_temperature,
            env_reward_scale=self.config.env_reward_scale,
            rm_reward_scale=self.config.rm_reward_scale,
            team_progress_bonus_coeff=self.config.team_progress_bonus_coeff,
            start_protocol="fixed",
            start_set_path=self.config.validation_start_set,
        )
        summary = payload["summary"]
        self.writer.add_scalar("eval/mean_completion_ratio", summary["mean_completion_ratio"], episode)
        self.writer.add_scalar("eval/success_rate", summary["success_rate"], episode)
        self.writer.add_scalar("eval/mean_steps", summary["mean_steps"], episode)
        self.writer.add_scalar("eval/mean_reward", summary["mean_reward"], episode)

        better = False
        if self.best_eval is None:
            better = True
        elif summary["mean_completion_ratio"] > self.best_eval["mean_completion_ratio"]:
            better = True
        elif (
            summary["mean_completion_ratio"] == self.best_eval["mean_completion_ratio"]
            and summary["success_rate"] > self.best_eval["success_rate"]
        ):
            better = True
        elif (
            summary["mean_completion_ratio"] == self.best_eval["mean_completion_ratio"]
            and summary["success_rate"] == self.best_eval["success_rate"]
            and summary["mean_steps"] < self.best_eval["mean_steps"]
        ):
            better = True

        if better:
            self.best_eval = dict(summary)
            self._save_checkpoint(
                self.run_dir / "checkpoints" / "best_by_completion.pt",
                extra_meta={"best_eval": self.best_eval, "selected_at_episode": episode},
            )
        payload["improved_best"] = better
        return payload

    def train(self) -> Path:
        progress = tqdm(range(1, self.config.episodes + 1), desc="Training SACC")
        last_episode = 0
        last_eval_payload = None
        stopped_early = False
        for episode in progress:
            last_episode = episode
            self._apply_training_schedules(episode)
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

            self.env.reset(seed=self.config.seed + episode - 1)
            done = False
            ep_reward = 0.0
            ep_env_reward = 0.0
            ep_rm_reward = 0.0
            ep_progress_bonus = 0.0
            steps = 0
            first_done_steps = None

            for _ in range(self.config.max_steps):
                obs_list = self.env.build_obs_list(include_rm=not self.config.disable_actor_rm)
                state = self.env.build_state(include_rm=not self.config.disable_critic_rm)
                action_masks = self.env.get_action_masks()
                obs_tensor_list = [
                    torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) for obs in obs_list
                ]
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    values = self._critic_values(state_tensor)

                actions: list[int] = []
                logprobs: list[float] = []
                for agent_idx in range(self.num_agents):
                    probs = self.actors[agent_idx](obs_tensor_list[agent_idx])
                    dist = _masked_categorical(
                        probs,
                        action_masks[agent_idx],
                        self.device,
                        temperature=self.current_train_temperature,
                    )
                    sampled = dist.sample()
                    actions.append(int(sampled.item()))
                    logprobs.append(float(dist.log_prob(sampled).item()))

                step_out = self.env.step(actions)
                rewards = step_out["rewards"]

                rollout["obs"].append(obs_list)
                rollout["states"].append(state)
                rollout["actions"].append(actions)
                rollout["logprobs"].append(logprobs)
                rollout["values"].append(values.detach().cpu().tolist())
                rollout["rewards"].append(rewards)
                rollout["dones"].append(1.0 if step_out["done"] else 0.0)
                rollout["action_masks"].append(action_masks)

                ep_reward += float(sum(rewards))
                ep_env_reward += float(sum(step_out["env_rewards"]))
                ep_rm_reward += float(sum(step_out["rm_rewards"]))
                ep_progress_bonus += float(step_out["progress_bonus"]) * self.num_agents
                steps += 1

                if step_out["done"]:
                    done = True
                    if first_done_steps is None:
                        first_done_steps = steps
                    break

            rewards_agent = torch.tensor(rollout["rewards"], dtype=torch.float32, device=self.device)
            values = torch.tensor(rollout["values"], dtype=torch.float32, device=self.device)
            dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=self.device)
            bootstrap_state = self.env.build_state(include_rm=not self.config.disable_critic_rm)

            if done:
                next_values = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            else:
                last_state = torch.tensor(bootstrap_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    next_values = self._critic_values(last_state)

            advantages, returns = _compute_gae_per_agent(
                rewards_agent,
                values,
                next_values,
                dones,
                gamma=self.config.gamma,
                lam=self.config.lam,
            )
            if self.config.advantage_norm:
                adv_mean = advantages.mean(dim=0, keepdim=True)
                adv_std = advantages.std(dim=0, unbiased=False, keepdim=True)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            old_logprobs = torch.tensor(rollout["logprobs"], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(rollout["actions"], dtype=torch.long, device=self.device)
            action_masks_tensor = torch.tensor(
                np.array(rollout["action_masks"], dtype=bool),
                dtype=torch.bool,
                device=self.device,
            )
            states_tensor = torch.tensor(
                np.array(rollout["states"], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )

            timesteps = len(rollout["rewards"])
            last_actor_loss = 0.0
            last_critic_loss = 0.0
            for _ in range(self.config.ppo_epochs):
                indices = np.arange(timesteps)
                np.random.shuffle(indices)
                for start in range(0, timesteps, self.config.mini_batch_size):
                    mb_idx = indices[start : start + self.config.mini_batch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_states = states_tensor[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_logp = old_logprobs[mb_idx]
                    mb_masks = action_masks_tensor[mb_idx]

                    critic_loss_total = 0.0
                    for agent_idx in range(self.num_agents):
                        values_pred = self.critics[agent_idx](mb_states).squeeze(-1)
                        critic_loss = F.mse_loss(values_pred, mb_returns[:, agent_idx].detach())
                        self.critic_optims[agent_idx].zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), self.config.grad_clip)
                        self.critic_optims[agent_idx].step()
                        critic_loss_total += float(critic_loss.item())
                    last_critic_loss = critic_loss_total / max(self.num_agents, 1)

                    joint_ratio = torch.ones(len(mb_idx), dtype=torch.float32, device=self.device)
                    actor_loss_total = 0.0
                    for agent_idx in range(self.num_agents):
                        obs_i = np.array([rollout["obs"][t][agent_idx] for t in mb_idx], dtype=np.float32)
                        obs_i = torch.tensor(obs_i, dtype=torch.float32, device=self.device)
                        probs = self.actors[agent_idx](obs_i)
                        dist = _masked_categorical(
                            probs,
                            mb_masks[:, agent_idx, :],
                            self.device,
                            temperature=self.current_train_temperature,
                        )
                        logp_now = dist.log_prob(mb_actions[:, agent_idx])
                        old_logp = mb_old_logp[:, agent_idx]
                        adv_i = mb_adv[:, agent_idx] * joint_ratio
                        ratio = torch.exp(logp_now - old_logp)
                        obj1 = ratio * adv_i
                        obj2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_i
                        actor_loss = -torch.min(obj1, obj2).mean() - self.current_entropy_coef * dist.entropy().mean()

                        self.actor_optims[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.config.grad_clip)
                        self.actor_optims[agent_idx].step()
                        actor_loss_total += float(actor_loss.item())

                        with torch.no_grad():
                            updated_probs = self.actors[agent_idx](obs_i)
                            updated_dist = _masked_categorical(updated_probs, mb_masks[:, agent_idx, :], self.device)
                            updated_logp = updated_dist.log_prob(mb_actions[:, agent_idx])
                            joint_ratio = joint_ratio * torch.exp(updated_logp - old_logp)
                            if (
                                self.config.prev_ratio_clip_min is not None
                                or self.config.prev_ratio_clip_max is not None
                            ):
                                clip_min = (
                                    float(self.config.prev_ratio_clip_min)
                                    if self.config.prev_ratio_clip_min is not None
                                    else float("-inf")
                                )
                                clip_max = (
                                    float(self.config.prev_ratio_clip_max)
                                    if self.config.prev_ratio_clip_max is not None
                                    else float("inf")
                                )
                                joint_ratio = torch.clamp(joint_ratio, min=clip_min, max=clip_max)

                    last_actor_loss = actor_loss_total / max(self.num_agents, 1)

            completion = self.env.get_completion_ratio()
            metric_row = {
                "episode": episode,
                "reward": ep_reward,
                "env_reward": ep_env_reward,
                "rm_reward": ep_rm_reward,
                "progress_bonus": ep_progress_bonus,
                "completion_ratio": completion,
                "steps": steps,
                "success": 1.0 if done else 0.0,
                "actor_loss": last_actor_loss,
                "critic_loss": last_critic_loss,
                "train_temperature": self.current_train_temperature,
                "entropy_coef": self.current_entropy_coef,
                "team_progress_bonus_coeff": float(self.env.team_progress_bonus_coeff),
                "distance_shaping_coeff": float(self.env.rm.config.distance_shaping_coeff),
                "start_positions": [list(pos) for pos in self.env.get_start_positions()],
            }
            append_jsonl(self.metrics_path, metric_row)

            self.writer.add_scalar("train/reward", ep_reward, episode)
            self.writer.add_scalar("train/env_reward", ep_env_reward, episode)
            self.writer.add_scalar("train/rm_reward", ep_rm_reward, episode)
            self.writer.add_scalar("train/progress_bonus", ep_progress_bonus, episode)
            self.writer.add_scalar("train/completion_ratio", completion, episode)
            self.writer.add_scalar("train/episode_length", steps, episode)
            self.writer.add_scalar("train/success", 1.0 if done else 0.0, episode)
            self.writer.add_scalar("train/actor_loss", last_actor_loss, episode)
            self.writer.add_scalar("train/critic_loss", last_critic_loss, episode)
            self.writer.add_scalar("train/train_temperature", self.current_train_temperature, episode)
            self.writer.add_scalar("train/entropy_coef", self.current_entropy_coef, episode)
            self.writer.add_scalar("train/team_progress_bonus_coeff", self.env.team_progress_bonus_coeff, episode)
            self.writer.add_scalar("train/distance_shaping_coeff", self.env.rm.config.distance_shaping_coeff, episode)
            if first_done_steps is not None:
                self.writer.add_scalar("train/steps_to_complete", first_done_steps, episode)

            progress.set_postfix(
                completion=f"{completion:.3f}",
                success=int(done),
                reward=f"{ep_reward:.2f}",
            )

            if self.config.save_interval and episode % self.config.save_interval == 0:
                self._save_checkpoint(self.run_dir / "checkpoints" / f"episode_{episode}.pt")

            if self.config.eval_interval and episode % self.config.eval_interval == 0:
                last_eval_payload = self._run_eval(episode)
                improved = bool(last_eval_payload.get("improved_best", False))
                if episode < self.config.early_stop_min_episode:
                    self.no_improve_evals = 0
                elif improved:
                    self.no_improve_evals = 0
                else:
                    self.no_improve_evals += 1

                if (
                    self.config.early_stop_patience_evals > 0
                    and episode >= self.config.early_stop_min_episode
                    and self.no_improve_evals >= self.config.early_stop_patience_evals
                ):
                    progress.write(
                        f"[EarlyStop] stop at ep {episode} after {self.no_improve_evals} evals without best_by_completion improvement"
                    )
                    stopped_early = True
                    break

        final_checkpoint = self._save_checkpoint(self.run_dir / "checkpoints" / "final.pt")
        if last_eval_payload is not None and last_episode % max(self.config.eval_interval, 1) == 0:
            final_eval = last_eval_payload
        else:
            final_eval = self._run_eval(last_episode if last_episode > 0 else self.config.episodes)
        summary = {
            "run_dir": str(self.run_dir),
            "final_checkpoint": str(final_checkpoint),
            "best_eval": self.best_eval,
            "final_eval": final_eval["summary"],
            "stopped_early": bool(stopped_early),
            "stopped_at_episode": int(last_episode),
            "metadata": self.metadata,
        }
        write_json(self.run_dir / "summary.json", summary)
        self.writer.close()
        return self.run_dir
