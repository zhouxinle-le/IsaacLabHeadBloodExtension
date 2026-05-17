from __future__ import annotations

import copy
import itertools
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO as SkrlPPO
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.runner.torch import Runner as SkrlRunner


def _cfg_get(cfg: object, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std() + 1e-8)


class SafePPO(SkrlPPO):
    """skrl PPO-Lagrangian with an explicit cost-value model."""

    def __init__(self, *args, cfg: dict | None = None, **kwargs):
        self._safe_cfg = (cfg or {}).get("safety", {})
        self._safety_enabled = bool(_cfg_get(self._safe_cfg, "enabled", True))
        self._safety_cost_limit = float(_cfg_get(self._safe_cfg, "cost_limit", 0.02))
        self._safety_lambda_lr = float(_cfg_get(self._safe_cfg, "lambda_lr", 0.5))
        self._safety_lambda_max = float(_cfg_get(self._safe_cfg, "lambda_max", 30.0))
        self._cost_value_loss_scale = float(_cfg_get(self._safe_cfg, "cost_value_loss_coef", 1.0))
        self._normalize_cost_advantage = bool(_cfg_get(self._safe_cfg, "normalize_cost_advantage", True))
        lambda_init = float(_cfg_get(self._safe_cfg, "lambda_init", 0.0))
        super().__init__(*args, cfg=cfg, **kwargs)

        self.cost_value = self.models.get("cost_value", None)
        if self.cost_value is None:
            raise KeyError("SafePPO requires a 'cost_value' model.")
        self.checkpoint_modules["cost_value"] = self.cost_value
        if config.torch.is_distributed:
            self.cost_value.broadcast_parameters()

        self._cost_lambda = torch.tensor(lambda_init, dtype=torch.float32, device=self.device)
        self._rebuild_optimizer()
        self._last_safety_logs = {
            "Safety-Lagrangian / Lambda": float(self._cost_lambda.item()),
            "Safety-Lagrangian / Cost limit": self._safety_cost_limit,
            "Safety-Lagrangian / Observed cost mean": 0.0,
            "Safety-Lagrangian / Lambda error": -self._safety_cost_limit,
        }

    def _rebuild_optimizer(self) -> None:
        params = [self.policy.parameters()]
        if self.value is not self.policy:
            params.append(self.value.parameters())
        if self.cost_value is not self.policy and self.cost_value is not self.value:
            params.append(self.cost_value.parameters())
        self.optimizer = torch.optim.Adam(itertools.chain(*params), lr=self._learning_rate)
        self.checkpoint_modules["optimizer"] = self.optimizer
        if self._learning_rate_scheduler is not None:
            self.scheduler = self._learning_rate_scheduler(
                self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
            )

    def init(self, trainer_cfg: Mapping[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="costs", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_advantages", size=1, dtype=torch.float32)
            self._tensors_names = [
                "states",
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "costs",
                "cost_values",
                "cost_returns",
                "cost_advantages",
            ]

    def _extract_cost(self, infos: Any, rewards: torch.Tensor) -> torch.Tensor:
        cost = None
        if isinstance(infos, Mapping):
            cost = infos.get("safety_cost")
        if cost is None:
            return torch.zeros_like(rewards)
        return cost.to(device=self.device, dtype=torch.float32).view_as(rewards)

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        Agent.record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is None:
            return
        self._current_next_states = next_states
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        costs = self._extract_cost(infos, rewards)
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)
            cost_values, _, _ = self.cost_value.act(
                {"states": self._state_preprocessor(states)}, role="cost_value"
            )

        if self._time_limit_bootstrap:
            rewards += self._discount_factor * values * truncated
            costs += self._discount_factor * cost_values * truncated

        log_prob = self._current_log_prob
        if log_prob is None:
            log_prob = torch.zeros_like(rewards)

        self.memory.add_samples(
            states=states,
            actions=actions.detach(),
            rewards=rewards.detach(),
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            log_prob=log_prob.detach(),
            values=values.detach(),
            costs=costs.detach(),
            cost_values=cost_values.detach(),
        )
        for memory in self.secondary_memories:
            memory.add_samples(
                states=states,
                actions=actions.detach(),
                rewards=rewards.detach(),
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob.detach(),
                values=values.detach(),
                costs=costs.detach(),
                cost_values=cost_values.detach(),
            )

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        last_values: torch.Tensor,
        normalize_advantage: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]
        for index in reversed(range(memory_size)):
            next_values = values[index + 1] if index < memory_size - 1 else last_values
            advantage = (
                rewards[index]
                - values[index]
                + self._discount_factor * not_dones[index] * (next_values + self._lambda * advantage)
            )
            advantages[index] = advantage
        returns = advantages + values
        if normalize_advantage:
            advantages = _normalize(advantages)
        return returns, advantages

    def _update_safety_lambda(self, observed_cost_mean: float) -> dict[str, float]:
        error = float(observed_cost_mean) - self._safety_cost_limit
        if self._safety_enabled:
            with torch.no_grad():
                self._cost_lambda.copy_(
                    torch.clamp(
                        self._cost_lambda + self._safety_lambda_lr * error,
                        min=0.0,
                        max=self._safety_lambda_max,
                    )
                )
        self._last_safety_logs = {
            "Safety-Lagrangian / Lambda": float(self._cost_lambda.item()),
            "Safety-Lagrangian / Cost limit": self._safety_cost_limit,
            "Safety-Lagrangian / Observed cost mean": float(observed_cost_mean),
            "Safety-Lagrangian / Lambda error": error,
        }
        return self._last_safety_logs

    def _update(self, timestep: int, timesteps: int) -> None:
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

            self.cost_value.train(False)
            last_cost_values, _, _ = self.cost_value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="cost_value"
            )
            self.cost_value.train(True)

        dones = self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated")
        values = self.memory.get_tensor_by_name("values")
        returns, advantages = self._compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=dones,
            values=values,
            last_values=last_values,
            normalize_advantage=True,
        )
        cost_values = self.memory.get_tensor_by_name("cost_values")
        cost_returns, cost_advantages = self._compute_gae(
            rewards=self.memory.get_tensor_by_name("costs"),
            dones=dones,
            values=cost_values,
            last_values=last_cost_values,
            normalize_advantage=self._normalize_cost_advantage,
        )
        self._update_safety_lambda(float(self.memory.get_tensor_by_name("costs").mean().detach().item()))

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
        self.memory.set_tensor_by_name("cost_values", cost_values)
        self.memory.set_tensor_by_name("cost_returns", cost_returns)
        self.memory.set_tensor_by_name("cost_advantages", cost_advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_cost_value_loss = 0.0

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                _sampled_costs,
                sampled_cost_values,
                sampled_cost_returns,
                sampled_cost_advantages,
            ) in sampled_batches:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    entropy_loss = (
                        -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                        if self._entropy_loss_scale
                        else 0
                    )

                    safe_advantages = sampled_advantages - self._cost_lambda.detach() * sampled_cost_advantages
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = safe_advantages * ratio
                    surrogate_clipped = safe_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    predicted_cost_values, _, _ = self.cost_value.act(
                        {"states": sampled_states}, role="cost_value"
                    )
                    if self._clip_predicted_values:
                        predicted_cost_values = sampled_cost_values + torch.clip(
                            predicted_cost_values - sampled_cost_values,
                            min=-self._value_clip,
                            max=self._value_clip,
                        )
                    cost_value_loss = self._cost_value_loss_scale * F.mse_loss(
                        sampled_cost_returns, predicted_cost_values
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + cost_value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                    if self.cost_value is not self.policy and self.cost_value is not self.value:
                        self.cost_value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    params = [self.policy.parameters()]
                    if self.policy is not self.value:
                        params.append(self.value.parameters())
                    if self.cost_value is not self.policy and self.cost_value is not self.value:
                        params.append(self.cost_value.parameters())
                    nn.utils.clip_grad_norm_(itertools.chain(*params), self._grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_cost_value_loss += cost_value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += float(entropy_loss.item())

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        num_updates = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data("Loss / Value loss", cumulative_value_loss / num_updates)
        self.track_data("Loss / Cost value loss", cumulative_cost_value_loss / num_updates)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / num_updates)
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        for key, value in self._last_safety_logs.items():
            self.track_data(key, value)


class SafeSkrlRunner(SkrlRunner):
    """skrl Runner with a local SafePPO component."""

    def __init__(self, env, cfg):
        cfg = copy.deepcopy(cfg)
        self._using_safe_ppo = cfg.get("agent", {}).get("class", "").lower() == "safeppo"
        if self._using_safe_ppo:
            cfg.setdefault("agent", {})["class"] = "PPO"
            cfg["agent"]["safety"] = cfg.get("safety", {})
        super().__init__(env, cfg)

    def _component(self, name: str):
        if self._using_safe_ppo and name.lower() == "ppo":
            return SafePPO
        if self._using_safe_ppo and name.lower() == "ppo_default_config":
            return PPO_DEFAULT_CONFIG
        return super()._component(name)
