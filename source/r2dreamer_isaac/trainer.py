from __future__ import annotations

from collections import defaultdict, deque
import pathlib
from typing import Any

import torch

from .env_adapter import IsaacR2DreamerEnvAdapter, IsaacStepOutput, obs_to_cpu, obs_to_device
from .replay_buffer import IsaacReplayBuffer
from .vendor.r2dreamer import tools


class IsaacOnlineTrainer:
    """Isaac-specific online trainer with terminal-aligned transitions."""

    _RECENT_EPISODE_WINDOW = 100

    def __init__(
        self,
        config,
        replay_buffer: IsaacReplayBuffer,
        logger: tools.Logger,
        logdir: pathlib.Path,
        train_env: IsaacR2DreamerEnvAdapter,
        eval_env: IsaacR2DreamerEnvAdapter | None,
        run_config: dict[str, Any] | None = None,
    ):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = pathlib.Path(logdir)
        self.train_env = train_env
        self.eval_env = eval_env
        self.run_config = run_config

        self.total_steps = int(config.total_steps)
        self.pretrain_updates = int(config.pretrain_updates)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.train_ratio = float(config.train_ratio)
        self.log_every = int(config.update_log_every)
        self.save_every = int(config.save_every)
        self.policy_save_every = int(config.policy_save_every)
        self.start_after_steps = int(config.start_after_steps)
        self.action_repeat = int(config.action_repeat)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)

        batch_steps = int(self.replay_buffer.batch_size * self.replay_buffer.batch_length)
        interval = max(batch_steps / max(self.train_ratio, 1.0) * self.action_repeat, 1.0)
        self._updates_needed = tools.Every(interval)
        self._should_eval = tools.Every(self.eval_every) if self.eval_every > 0 else lambda *_args, **_kwargs: 0
        self._should_log = tools.Every(self.log_every) if self.log_every > 0 else lambda *_args, **_kwargs: 0
        self._should_save = tools.Every(self.save_every) if self.save_every > 0 else lambda *_args, **_kwargs: 0
        self._should_policy_save = (
            tools.Every(self.policy_save_every) if self.policy_save_every > 0 else lambda *_args, **_kwargs: 0
        )
        self._should_pretrain = tools.Once()

        self.latest_env_metrics: dict[str, Any] = {}
        self._recent_episode_scores: deque[float] = deque(maxlen=self._RECENT_EPISODE_WINDOW)
        self._recent_episode_lengths: deque[float] = deque(maxlen=self._RECENT_EPISODE_WINDOW)
        self._recent_safety_costs: deque[float] = deque(maxlen=self._RECENT_EPISODE_WINDOW)
        self._interval_episode_scores: list[float] = []
        self._interval_episode_lengths: list[float] = []
        self._recent_episode_metrics: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._RECENT_EPISODE_WINDOW)
        )

    @staticmethod
    def _step_output_to_device(step_out: IsaacStepOutput, device: torch.device) -> IsaacStepOutput:
        return IsaacStepOutput(
            obs=obs_to_device(step_out.obs, device),
            aligned_next_obs=obs_to_device(step_out.aligned_next_obs, device),
            reward=step_out.reward.to(device),
            cost=step_out.cost.to(device),
            terminated=step_out.terminated.to(device),
            truncated=step_out.truncated.to(device),
            done=step_out.done.to(device),
            extras=step_out.extras,
        )

    def _initial_runtime(self, agent) -> dict[str, Any]:
        current_obs, _ = self.train_env.reset()
        current_obs = obs_to_device(current_obs, agent.device)
        num_envs = self.train_env.num_envs
        return {
            "global_step": 0,
            "update_count": 0,
            "next_episode_id": num_envs,
            "current_obs": current_obs,
            "current_is_first": torch.ones(num_envs, dtype=torch.bool, device=agent.device),
            "current_episode_ids": torch.arange(num_envs, dtype=torch.int32, device=agent.device),
            "episode_returns": torch.zeros(num_envs, dtype=torch.float32, device=agent.device),
            "episode_lengths": torch.zeros(num_envs, dtype=torch.int32, device=agent.device),
            "agent_state": {key: value.clone() for key, value in agent.get_initial_state(num_envs).items()},
        }

    @staticmethod
    def _metric_to_scalar(value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() <= 0:
                return None
            if value.numel() == 1:
                return float(value.item())
            return float(value.detach().float().mean().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _remember_recent_episode_metric(self, name: str, value: float, repeat: int = 1) -> None:
        if repeat <= 0:
            return
        for _ in range(repeat):
            self._recent_episode_metrics[name].append(float(value))

    @staticmethod
    def _recent_metric_name(name: str) -> str:
        metric = name
        if metric.startswith("Episode_"):
            metric = metric[len("Episode_") :]
        metric = metric.replace("/", "_").lower()
        return f"rollout/recent_{metric}"

    def _update_recent_episode_logs(self, env_metrics: dict[str, Any], done_count: int) -> None:
        if done_count <= 0:
            return
        for name, value in env_metrics.items():
            if not name.startswith("Episode_"):
                continue
            scalar = self._metric_to_scalar(value)
            if scalar is None:
                continue
            # The environment reports averages over the finished env_ids for this step.
            # Repeating the batch mean preserves the correct weighting across varying
            # numbers of completed episodes between training log intervals.
            self._remember_recent_episode_metric(name, scalar, repeat=done_count)

    def _log_recent_episode_metrics(self) -> None:
        if self._recent_episode_scores:
            self.logger.scalar(
                "rollout/recent_episode_score_mean",
                sum(self._recent_episode_scores) / len(self._recent_episode_scores),
            )
        if self._recent_episode_lengths:
            self.logger.scalar(
                "rollout/recent_episode_length_mean",
                sum(self._recent_episode_lengths) / len(self._recent_episode_lengths),
            )
        for name, values in sorted(self._recent_episode_metrics.items()):
            if not values:
                continue
            self.logger.scalar(self._recent_metric_name(name), sum(values) / len(values))

    def _log_interval_episode_metrics(self) -> None:
        if not self._interval_episode_scores:
            return
        self.logger.scalar(
            "rollout/interval_episode_score_mean",
            sum(self._interval_episode_scores) / len(self._interval_episode_scores),
        )
        self.logger.scalar(
            "rollout/interval_episode_length_mean",
            sum(self._interval_episode_lengths) / len(self._interval_episode_lengths),
        )
        self.logger.scalar("rollout/interval_episode_count", len(self._interval_episode_scores))
        self._interval_episode_scores.clear()
        self._interval_episode_lengths.clear()

    def _log_env_metrics(self, env_metrics: dict[str, Any]) -> None:
        for name, value in env_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.detach().float().mean().item()
            self.logger.scalar(name, value)

    def _log_completed_episodes(self, runtime: dict[str, Any], done: torch.Tensor) -> None:
        done_indices = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for index in done_indices.tolist():
            if runtime["episode_lengths"][index] <= 0:
                continue
            score = runtime["episode_returns"][index].item()
            length = runtime["episode_lengths"][index].item()
            self._recent_episode_scores.append(float(score))
            self._recent_episode_lengths.append(float(length))
            self._interval_episode_scores.append(float(score))
            self._interval_episode_lengths.append(float(length))
            self.logger.scalar("episode/score", score)
            self.logger.scalar("episode/length", length)
            self.logger.write(runtime["global_step"] + index)
            runtime["episode_returns"][index] = 0.0
            runtime["episode_lengths"][index] = 0

    def _advance_episode_ids(self, runtime: dict[str, Any], done: torch.Tensor) -> None:
        done_indices = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_indices.numel() == 0:
            return
        next_ids = torch.arange(
            runtime["next_episode_id"],
            runtime["next_episode_id"] + done_indices.numel(),
            dtype=torch.int32,
            device=done.device,
        )
        runtime["current_episode_ids"][done_indices] = next_ids
        runtime["next_episode_id"] += int(done_indices.numel())

    def _build_transition(
        self,
        current_is_first: torch.Tensor,
        current_episode_ids: torch.Tensor,
        action: torch.Tensor,
        agent_state: dict[str, torch.Tensor],
        step_out,
    ) -> dict[str, torch.Tensor]:
        transition = {
            "action": action.detach(),
            "reward": step_out.reward.detach(),
            "cost": step_out.cost.detach(),
            "is_first": current_is_first.detach(),
            "is_last": step_out.done.detach(),
            "is_terminal": step_out.terminated.detach(),
            "episode": current_episode_ids.detach(),
            "stoch": agent_state["stoch"].detach(),
            "deter": agent_state["deter"].detach(),
        }
        for key, value in self.train_env.observation_items(step_out.aligned_next_obs).items():
            transition[key] = value.detach()
        return transition

    def _save_checkpoint(self, agent, runtime: dict[str, Any], full: bool = True) -> None:
        checkpoint_dir = self.logdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if full:
            payload = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "scheduler_state_dict": agent._scheduler.state_dict(),
                "scaler_state_dict": agent._scaler.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "config": self.run_config,
                "runtime": {
                    key: (
                        obs_to_cpu(value)
                        if isinstance(value, dict)
                        else value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in runtime.items()
                },
            }
            torch.save(payload, checkpoint_dir / "latest.pt")
            return

        policy_payload = {
            "agent_state_dict": agent.state_dict(),
            "step": int(runtime["global_step"]),
        }
        torch.save(policy_payload, checkpoint_dir / f"policy_step_{int(runtime['global_step'])}.pt")

    def _load_checkpoint(self, agent, checkpoint: dict[str, Any]) -> dict[str, Any]:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])

        runtime = checkpoint["runtime"]
        loaded_runtime = {
            key: (
                obs_to_device(value, agent.device)
                if isinstance(value, dict)
                else value.to(agent.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in runtime.items()
        }
        # Resume learner/replay state exactly, but restart simulator episodes safely.
        # Isaac simulator state is not serialized in this first version, so we seed a
        # fresh runtime and only carry over counters that are independent of the live env.
        resumed_runtime = self._initial_runtime(agent)
        resumed_runtime["global_step"] = int(loaded_runtime["global_step"])
        resumed_runtime["update_count"] = int(loaded_runtime["update_count"])
        resumed_runtime["next_episode_id"] = int(loaded_runtime["next_episode_id"]) + self.train_env.num_envs
        resumed_runtime["current_episode_ids"] = torch.arange(
            int(loaded_runtime["next_episode_id"]),
            int(loaded_runtime["next_episode_id"]) + self.train_env.num_envs,
            dtype=torch.int32,
            device=agent.device,
        )
        return resumed_runtime

    @torch.no_grad()
    def eval(self, agent, train_step: int) -> None:
        if self.eval_env is None or self.eval_episode_num <= 0:
            return

        env = self.eval_env
        agent.eval()
        current_obs, _ = env.reset()
        current_obs = obs_to_device(current_obs, agent.device)
        current_is_first = torch.ones(env.num_envs, dtype=torch.bool, device=agent.device)
        agent_state = {key: value.clone() for key, value in agent.get_initial_state(env.num_envs).items()}
        returns = torch.zeros(env.num_envs, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(env.num_envs, dtype=torch.int32, device=agent.device)
        finished_returns = []
        finished_lengths = []

        while len(finished_returns) < self.eval_episode_num:
            action, agent_state = agent.act(env.build_agent_obs(current_obs, current_is_first), agent_state, eval=True)
            step_out = env.step(action)
            step_out = self._step_output_to_device(step_out, agent.device)
            returns += step_out.reward[:, 0]
            lengths += 1

            done_indices = torch.nonzero(step_out.done, as_tuple=False).squeeze(-1)
            for index in done_indices.tolist():
                finished_returns.append(returns[index].item())
                finished_lengths.append(lengths[index].item())
                returns[index] = 0.0
                lengths[index] = 0
                if len(finished_returns) >= self.eval_episode_num:
                    break

            current_obs = obs_to_device(step_out.obs, agent.device)
            current_is_first = step_out.done.to(agent.device)

        self.logger.scalar("episode/eval_score", sum(finished_returns) / len(finished_returns))
        self.logger.scalar("episode/eval_length", sum(finished_lengths) / len(finished_lengths))
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent, checkpoint: dict[str, Any] | None = None) -> None:
        runtime = self._initial_runtime(agent) if checkpoint is None else self._load_checkpoint(agent, checkpoint)
        train_metrics: dict[str, Any] = {}

        while int(runtime["global_step"]) < self.total_steps:
            if self._should_eval(int(runtime["global_step"])):
                self.eval(agent, int(runtime["global_step"]))

            action, next_agent_state = agent.act(
                self.train_env.build_agent_obs(runtime["current_obs"], runtime["current_is_first"]),
                runtime["agent_state"],
                eval=False,
            )
            step_out = self.train_env.step(action)
            step_out = self._step_output_to_device(step_out, agent.device)
            self.latest_env_metrics = step_out.extras.get("log", {})
            self._update_recent_episode_logs(
                self.latest_env_metrics,
                done_count=int(step_out.done.sum().item()),
            )

            transition = self._build_transition(
                current_is_first=runtime["current_is_first"],
                current_episode_ids=runtime["current_episode_ids"],
                action=action,
                agent_state=next_agent_state,
                step_out=step_out,
            )
            self.replay_buffer.add_transition(transition)
            self._recent_safety_costs.append(float(step_out.cost.detach().float().mean().item()))

            runtime["global_step"] += self.train_env.num_envs * self.action_repeat
            runtime["episode_returns"] += step_out.reward[:, 0]
            runtime["episode_lengths"] += 1

            self._log_completed_episodes(runtime, step_out.done)

            if (
                runtime["global_step"] >= self.start_after_steps
                and self.replay_buffer.count() >= self.replay_buffer.batch_length
                and self.replay_buffer.can_sample()
            ):
                if self._should_pretrain():
                    update_num = self.pretrain_updates
                else:
                    update_num = self._updates_needed(int(runtime["global_step"]))
                for _ in range(update_num):
                    safe_metrics = {}
                    if hasattr(agent, "update_safety_lambda"):
                        observed_cost = (
                            sum(self._recent_safety_costs) / len(self._recent_safety_costs)
                            if self._recent_safety_costs
                            else 0.0
                        )
                        safe_metrics = agent.update_safety_lambda(observed_cost)
                    train_metrics = agent.update(self.replay_buffer)
                    train_metrics.update(safe_metrics)
                runtime["update_count"] += update_num

                if train_metrics and self._should_log(int(runtime["global_step"])):
                    for name, value in train_metrics.items():
                        if isinstance(value, torch.Tensor):
                            value = value.detach().item()
                        if str(name).startswith("safe_dreamer/"):
                            self.logger.scalar(name, value)
                        else:
                            self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", runtime["update_count"])
                    self._log_env_metrics(self.latest_env_metrics)
                    self._log_interval_episode_metrics()
                    self._log_recent_episode_metrics()
                    self.logger.write(int(runtime["global_step"]), fps=True)

            if self._should_save(int(runtime["global_step"])):
                self._save_checkpoint(agent, runtime, full=True)
            if self._should_policy_save(int(runtime["global_step"])):
                self._save_checkpoint(agent, runtime, full=False)

            runtime["current_obs"] = obs_to_device(step_out.obs, agent.device)
            runtime["current_is_first"] = step_out.done.to(agent.device)
            runtime["agent_state"] = {key: value.detach() for key, value in next_agent_state.items()}
            self._advance_episode_ids(runtime, step_out.done.to(runtime["current_episode_ids"].device))

        self._save_checkpoint(agent, runtime, full=True)
