from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO as RslPPO
from rsl_rl.modules import ActorCritic as RslActorCritic
from rsl_rl.runners import OnPolicyRunner as RslOnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_nn_activation


def _cfg_get(cfg: object, key: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / (values.std() + 1e-8)


class SafeActorCritic(RslActorCritic):
    """RSL-RL actor-critic with an additional cost-value critic."""

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        activation_fn = resolve_nn_activation(activation)
        layers: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_fn]
        for index in range(len(critic_hidden_dims)):
            if index == len(critic_hidden_dims) - 1:
                layers.append(nn.Linear(critic_hidden_dims[index], 1))
            else:
                layers.append(nn.Linear(critic_hidden_dims[index], critic_hidden_dims[index + 1]))
                layers.append(resolve_nn_activation(activation))
        self.cost_critic = nn.Sequential(*layers)
        print(f"Cost critic MLP: {self.cost_critic}")

    def evaluate_cost(self, critic_observations, **kwargs):
        return self.cost_critic(critic_observations)


class SafeRolloutStorage(RolloutStorage):
    """Rollout storage with cost returns and cost advantages."""

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape,
            device,
        )
        if training_type == "rl":
            self.costs = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

    def add_transitions(self, transition):
        if self.training_type == "rl":
            cost = getattr(transition, "costs", None)
            cost_value = getattr(transition, "cost_values", None)
            if cost is None:
                cost = torch.zeros_like(transition.rewards)
            if cost_value is None:
                cost_value = torch.zeros_like(transition.values)
            self.costs[self.step].copy_(cost.view(-1, 1))
            self.cost_values[self.step].copy_(cost_value.view(-1, 1))
        super().add_transitions(transition)

    def compute_returns(
        self,
        last_values,
        last_cost_values,
        gamma,
        lam,
        normalize_advantage: bool = True,
        normalize_cost_advantage: bool = True,
    ):
        reward_advantage = 0
        cost_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_cost_values = last_cost_values
            else:
                next_values = self.values[step + 1]
                next_cost_values = self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()

            reward_delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            reward_advantage = reward_delta + next_is_not_terminal * gamma * lam * reward_advantage
            self.returns[step] = reward_advantage + self.values[step]

            cost_delta = self.costs[step] + next_is_not_terminal * gamma * next_cost_values - self.cost_values[step]
            cost_advantage = cost_delta + next_is_not_terminal * gamma * lam * cost_advantage
            self.cost_returns[step] = cost_advantage + self.cost_values[step]

        self.advantages = self.returns - self.values
        self.cost_advantages = self.cost_returns - self.cost_values
        if normalize_advantage:
            self.advantages = _normalize(self.advantages)
        if normalize_cost_advantage:
            self.cost_advantages = _normalize(self.cost_advantages)

    def safe_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("Safe mini-batches are only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        privileged_observations = (
            self.privileged_observations.flatten(0, 1)
            if self.privileged_observations is not None
            else observations
        )
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)
        rnd_state = self.rnd_state.flatten(0, 1) if self.rnd_state_shape is not None else None

        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                end = (index + 1) * mini_batch_size
                batch_idx = indices[start:end]
                rnd_state_batch = rnd_state[batch_idx] if rnd_state is not None else None
                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    cost_values[batch_idx],
                    cost_returns[batch_idx],
                    cost_advantages[batch_idx],
                    (None, None),
                    None,
                    rnd_state_batch,
                )


class SafePPO(RslPPO):
    """PPO-Lagrangian for RSL-RL state observations."""

    def __init__(
        self,
        policy,
        *args,
        safety=None,
        cost_value_loss_coef: float | None = None,
        normalize_cost_advantage: bool | None = None,
        **kwargs,
    ):
        self.safety_cfg = safety or {}
        self.safety_enabled = bool(_cfg_get(self.safety_cfg, "enabled", True))
        self.safety_cost_limit = float(_cfg_get(self.safety_cfg, "cost_limit", 0.02))
        self.safety_lambda_lr = float(_cfg_get(self.safety_cfg, "lambda_lr", 0.5))
        self.safety_lambda_max = float(_cfg_get(self.safety_cfg, "lambda_max", 30.0))
        lambda_init = float(_cfg_get(self.safety_cfg, "lambda_init", 0.0))
        self.cost_value_loss_coef = float(
            _cfg_get(self.safety_cfg, "cost_value_loss_coef", 1.0)
            if cost_value_loss_coef is None
            else cost_value_loss_coef
        )
        self.normalize_cost_advantage = bool(
            _cfg_get(self.safety_cfg, "normalize_cost_advantage", True)
            if normalize_cost_advantage is None
            else normalize_cost_advantage
        )
        super().__init__(policy, *args, **kwargs)
        self.cost_lambda = torch.tensor(lambda_init, dtype=torch.float32, device=self.device)
        self.safety_logs: dict[str, float] = {
            "Safety_Lagrangian/lambda": float(self.cost_lambda.item()),
            "Safety_Lagrangian/cost_limit": self.safety_cost_limit,
            "Safety_Lagrangian/observed_cost_mean": 0.0,
            "Safety_Lagrangian/lambda_error": -self.safety_cost_limit,
        }

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        rnd_state_shape = [self.rnd.num_states] if self.rnd else None
        self.storage = SafeRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        actions = super().act(obs, critic_obs)
        self.transition.cost_values = self.policy.evaluate_cost(critic_obs).detach()
        return actions

    def _extract_cost(self, rewards: torch.Tensor, infos: dict) -> torch.Tensor:
        cost = infos.get("safety_cost")
        if cost is None:
            return torch.zeros_like(rewards)
        return cost.to(device=self.device, dtype=torch.float32).view_as(rewards)

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.costs = self._extract_cost(rewards, infos).clone()

        if self.rnd:
            rnd_state = infos["observations"]["rnd_state"]
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            self.transition.rewards += self.intrinsic_rewards
            self.transition.rnd_state = rnd_state.clone()

        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.costs += self.gamma * torch.squeeze(self.transition.cost_values * time_outs, 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        observed_cost = float(self.storage.costs.mean().detach().item())
        self.update_safety_lambda(observed_cost)
        last_values = self.policy.evaluate(last_critic_obs).detach()
        last_cost_values = self.policy.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            last_cost_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            normalize_cost_advantage=self.normalize_cost_advantage,
        )

    def update_safety_lambda(self, observed_cost_mean: float) -> dict[str, float]:
        error = float(observed_cost_mean) - self.safety_cost_limit
        if self.safety_enabled:
            with torch.no_grad():
                self.cost_lambda.copy_(
                    torch.clamp(
                        self.cost_lambda + self.safety_lambda_lr * error,
                        min=0.0,
                        max=self.safety_lambda_max,
                    )
                )
        self.safety_logs = {
            "Safety_Lagrangian/lambda": float(self.cost_lambda.item()),
            "Safety_Lagrangian/cost_limit": self.safety_cost_limit,
            "Safety_Lagrangian/observed_cost_mean": float(observed_cost_mean),
            "Safety_Lagrangian/lambda_error": error,
        }
        return self.safety_logs

    def update(self):
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        if self.policy.is_recurrent:
            raise NotImplementedError("SafePPO currently supports feed-forward RSL-RL policies.")
        generator = self.storage.safe_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            cost_target_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
            hid_states_batch,
            masks_batch,
            _rnd_state_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = _normalize(advantages_batch)
                if self.normalize_cost_advantage:
                    cost_advantages_batch = _normalize(cost_advantages_batch)

            safe_advantages_batch = advantages_batch - self.cost_lambda.detach() * cost_advantages_batch

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            cost_value_batch = self.policy.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            safe_advantages = torch.squeeze(safe_advantages_batch)
            surrogate = -safe_advantages * ratio
            surrogate_clipped = -safe_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max((value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2))
                value_loss = value_loss.mean()

                cost_value_clipped = cost_target_values_batch + (cost_value_batch - cost_target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                cost_value_loss = torch.max(
                    (cost_value_batch - cost_returns_batch).pow(2),
                    (cost_value_clipped - cost_returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "cost_value_function": mean_cost_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }


class SafeOnPolicyRunner(RslOnPolicyRunner):
    """RSL-RL runner that injects the local SafePPO and SafeActorCritic classes."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        import rsl_rl.runners.on_policy_runner as runner_module

        patched_cfg = copy.deepcopy(train_cfg)
        patched_cfg["algorithm"]["class_name"] = "PPO"
        patched_cfg["policy"]["class_name"] = "ActorCritic"
        runner_module.PPO = SafePPO
        runner_module.ActorCritic = SafeActorCritic
        super().__init__(env, patched_cfg, log_dir=log_dir, device=device)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if self.writer is not None and hasattr(self.alg, "safety_logs"):
            for key, value in self.alg.safety_logs.items():
                self.writer.add_scalar(key, value, locs["it"])

