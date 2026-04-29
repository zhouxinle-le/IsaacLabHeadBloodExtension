from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ParticleTaskState:
    absorbed_delta: torch.Tensor
    absorbed_count: torch.Tensor
    nearest_particle: torch.Tensor
    nearest_particle_distance: torch.Tensor
    prev_nearest_particle_distance: torch.Tensor
    valid_in_cone_ratio: torch.Tensor
    valid_in_inlet_ratio: torch.Tensor


@dataclass(frozen=True)
class ParticleRewardInputs:
    raw_actions: torch.Tensor
    contact_force: torch.Tensor


class ParticleTaskTracker:
    def __init__(self, cfg, num_envs: int, device: torch.device | str):
        self._num_envs = int(num_envs)
        self.device = torch.device(device)

        self.state = ParticleTaskState(
            absorbed_delta=self._zeros(),
            absorbed_count=self._zeros(),
            nearest_particle=self._zeros((self._num_envs, 3)),
            nearest_particle_distance=self._zeros(),
            prev_nearest_particle_distance=self._zeros(),
            valid_in_cone_ratio=self._zeros(),
            valid_in_inlet_ratio=self._zeros(),
        )

    def _zeros(self, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        if shape is None:
            shape = (self._num_envs,)
        return torch.zeros(shape, dtype=torch.float32, device=self.device)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def refresh(
        self,
        tip_pos_w: torch.Tensor,
        step_count: torch.Tensor,
        particle_stats: dict[str, np.ndarray],
    ) -> None:
        # Update absorbed metrics
        absorbed_delta = torch.from_numpy(particle_stats["absorbed_delta"]).to(
            device=self.device, dtype=torch.float32
        )
        self.state.absorbed_delta[:] = absorbed_delta
        self.state.absorbed_count += absorbed_delta

        # Update nearest-particle target and ratios.
        prev_distance = self.state.nearest_particle_distance.clone()
        self.state.prev_nearest_particle_distance.copy_(prev_distance)

        step_count_np = self._to_numpy(step_count).astype(np.int64, copy=False)

        nearest_particle_w = torch.from_numpy(particle_stats["nearest_particle_w"]).to(
            device=self.device, dtype=torch.float32
        )
        nearest_distance = torch.from_numpy(particle_stats["nearest_particle_distance"]).to(
            device=self.device, dtype=torch.float32
        )
        in_cone_ratio = torch.from_numpy(particle_stats["valid_in_cone_ratio"]).to(
            device=self.device, dtype=torch.float32
        )
        in_inlet_ratio = torch.from_numpy(particle_stats["valid_in_inlet_ratio"]).to(
            device=self.device, dtype=torch.float32
        )

        self.state.nearest_particle[:] = nearest_particle_w
        self.state.nearest_particle_distance[:] = nearest_distance
        self.state.valid_in_cone_ratio[:] = in_cone_ratio
        self.state.valid_in_inlet_ratio[:] = in_inlet_ratio

        # Handle env_idx with step <= 0 (reset condition)
        reset_mask = torch.tensor(
            step_count_np <= 0, device=self.device, dtype=torch.bool
        )
        self.state.prev_nearest_particle_distance[reset_mask] = nearest_distance[
            reset_mask
        ]

    def reset(self, env_ids: torch.Tensor, tip_pos_w: torch.Tensor) -> None:
        self.state.absorbed_delta[env_ids] = 0.0
        self.state.absorbed_count[env_ids] = 0.0
        self.state.nearest_particle_distance[env_ids] = 0.0
        self.state.prev_nearest_particle_distance[env_ids] = 0.0
        self.state.valid_in_cone_ratio[env_ids] = 0.0
        self.state.valid_in_inlet_ratio[env_ids] = 0.0
        self.state.nearest_particle[env_ids] = tip_pos_w[env_ids]
