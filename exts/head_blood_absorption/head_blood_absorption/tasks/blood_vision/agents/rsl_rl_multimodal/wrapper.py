from __future__ import annotations

import torch

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


class BloodVisionRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Flatten blood_vision camera + position observations for the default rsl-rl pipeline."""

    def _flatten_policy_obs(self, policy_obs) -> torch.Tensor:
        if isinstance(policy_obs, torch.Tensor):
            return policy_obs

        camera = policy_obs["camera"].reshape(policy_obs["camera"].shape[0], -1)
        position = policy_obs["position"].reshape(policy_obs["position"].shape[0], -1)
        return torch.cat((camera, position), dim=-1)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._flatten_policy_obs(obs_dict["policy"]), {"observations": obs_dict}

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        obs_dict, _ = self.env.reset()
        return self._flatten_policy_obs(obs_dict["policy"]), {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = self._flatten_policy_obs(obs_dict["policy"])
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs, rew, dones, extras
