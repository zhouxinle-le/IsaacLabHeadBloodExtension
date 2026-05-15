from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

TensorObs = torch.Tensor | dict[str, torch.Tensor]


@dataclass
class IsaacStepOutput:
    obs: TensorObs
    aligned_next_obs: TensorObs
    reward: torch.Tensor
    cost: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    extras: dict[str, Any]


def obs_to_device(obs: TensorObs, device: torch.device | str) -> TensorObs:
    if isinstance(obs, dict):
        return {key: value.to(device=device, non_blocking=True) for key, value in obs.items()}
    return obs.to(device=device, non_blocking=True)


def obs_to_cpu(obs: TensorObs) -> TensorObs:
    if isinstance(obs, dict):
        return {key: value.detach().cpu() for key, value in obs.items()}
    return obs.detach().cpu()


class IsaacR2DreamerEnvAdapter:
    """Adapter that turns Isaac tasks into Dreamer-friendly observation dicts."""

    def __init__(self, env: gym.Env, obs_key: str = "policy"):
        self.env = env
        self.unwrapped = env.unwrapped
        self.obs_key = obs_key
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device

        cfg_obs_space = self.unwrapped.cfg.observation_space
        act_dim = int(self.unwrapped.cfg.action_space)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        if isinstance(cfg_obs_space, dict):
            self.modality = "vision"
            camera_shape = tuple(int(v) for v in cfg_obs_space["camera"])
            if len(camera_shape) != 3:
                raise ValueError(f"Expected blood_vision camera shape [C, H, W], got: {camera_shape}")
            channels, height, width = camera_shape
            position_dim = int(cfg_obs_space["position"])
            self.observation_keys = ("image", "position")
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(height, width, channels),
                        dtype=np.uint8,
                    ),
                    "position": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(position_dim,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.modality = "state"
            obs_dim = int(cfg_obs_space)
            self.observation_keys = ("state",)
            self.observation_space = gym.spaces.Dict(
                {"state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
            )

    @staticmethod
    def _first_tensor(obs: TensorObs) -> torch.Tensor:
        if isinstance(obs, dict):
            return next(iter(obs.values()))
        return obs

    @staticmethod
    def _mask_select(mask: torch.Tensor, terminal: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
        view_shape = (mask.shape[0],) + (1,) * (live.ndim - 1)
        return torch.where(mask.reshape(view_shape), terminal.to(device=live.device, dtype=live.dtype), live)

    @staticmethod
    def _camera_to_image(camera: torch.Tensor) -> torch.Tensor:
        if camera.ndim != 4:
            raise ValueError(f"Expected camera observation with shape (B, C, H, W), got: {tuple(camera.shape)}")

        if camera.dtype == torch.uint8:
            camera_uint8 = camera
        else:
            camera_uint8 = torch.clamp(camera.to(dtype=torch.float32), 0.0, 1.0).mul(255.0).round().to(torch.uint8)
        return camera_uint8.permute(0, 2, 3, 1).contiguous()

    def _convert_state_obs(self, policy_obs: TensorObs) -> torch.Tensor:
        if isinstance(policy_obs, torch.Tensor):
            obs = policy_obs
        elif "state" in policy_obs:
            obs = policy_obs["state"]
        else:
            raise TypeError(f"Expected tensor state observation, got keys: {tuple(policy_obs)}")
        return obs.to(dtype=torch.float32)

    def _convert_vision_obs(self, policy_obs: TensorObs) -> dict[str, torch.Tensor]:
        if not isinstance(policy_obs, dict):
            raise TypeError(f"Expected dict vision observation, got: {type(policy_obs)!r}")

        if "image" in policy_obs:
            image = policy_obs["image"]
            if image.dtype != torch.uint8:
                image = torch.clamp(image.to(dtype=torch.float32), 0.0, 1.0).mul(255.0).round().to(torch.uint8)
        else:
            image = self._camera_to_image(policy_obs["camera"])

        return {
            "image": image,
            "position": policy_obs["position"].to(dtype=torch.float32),
        }

    def _convert_policy_obs(self, policy_obs: TensorObs) -> TensorObs:
        if self.modality == "vision":
            return self._convert_vision_obs(policy_obs)
        return self._convert_state_obs(policy_obs)

    def _extract_policy_obs(self, obs_dict: dict[str, TensorObs] | TensorObs) -> TensorObs:
        if isinstance(obs_dict, torch.Tensor):
            policy_obs = obs_dict
        else:
            policy_obs = obs_dict[self.obs_key]
        return self._convert_policy_obs(policy_obs)

    def _align_next_obs(self, obs: TensorObs, terminal_policy: TensorObs | None, done: torch.Tensor, extras) -> TensorObs:
        first = self._first_tensor(obs)
        terminal_mask = extras.get("terminal_mask", done).to(dtype=torch.bool, device=first.device)
        if terminal_policy is None:
            return obs

        terminal_obs = self._convert_policy_obs(terminal_policy)
        if isinstance(obs, dict):
            if not isinstance(terminal_obs, dict):
                raise TypeError("Terminal observation type does not match live dict observation.")
            return {
                key: self._mask_select(terminal_mask, terminal_obs[key], value)
                for key, value in obs.items()
            }
        if isinstance(terminal_obs, dict):
            raise TypeError("Terminal observation type does not match live tensor observation.")
        return self._mask_select(terminal_mask, terminal_obs, obs)

    def reset(self) -> tuple[TensorObs, dict[str, Any]]:
        obs_dict, extras = self.env.reset()
        return self._extract_policy_obs(obs_dict), extras

    def step(self, actions: torch.Tensor) -> IsaacStepOutput:
        obs_dict, reward, terminated, truncated, extras = self.env.step(actions.to(device=self.device))
        obs = self._extract_policy_obs(obs_dict)
        cost = extras.get("safety_cost") if isinstance(extras, dict) else None
        if cost is None:
            cost = torch.zeros_like(reward)
        else:
            cost = cost.to(device=self.device)
        reward = reward.to(dtype=torch.float32).unsqueeze(-1)
        cost = cost.to(dtype=torch.float32).unsqueeze(-1)
        terminated = terminated.to(dtype=torch.bool)
        truncated = truncated.to(dtype=torch.bool)
        done = terminated | truncated

        terminal_obs = extras.get("terminal_observation", {})
        terminal_policy = terminal_obs.get(self.obs_key) if isinstance(terminal_obs, dict) else None
        aligned_next_obs = self._align_next_obs(obs, terminal_policy, done, extras)

        return IsaacStepOutput(
            obs=obs,
            aligned_next_obs=aligned_next_obs,
            reward=reward,
            cost=cost,
            terminated=terminated,
            truncated=truncated,
            done=done,
            extras=extras,
        )

    def observation_items(self, obs: TensorObs) -> dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            return {key: obs[key] for key in self.observation_keys}
        return {"state": obs}

    def build_agent_obs(self, obs: TensorObs, is_first: torch.Tensor) -> dict[str, torch.Tensor]:
        items = self.observation_items(obs)
        first = self._first_tensor(obs)
        agent_obs = {key: value for key, value in items.items()}
        agent_obs["is_first"] = is_first.to(device=first.device, dtype=torch.bool)
        return agent_obs

    def close(self) -> None:
        self.env.close()


class IsaacBloodStateEnvAdapter(IsaacR2DreamerEnvAdapter):
    """Backward-compatible adapter name for existing blood_state scripts/tests."""
