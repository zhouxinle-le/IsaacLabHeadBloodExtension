from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ParticleTaskState:
    absorbed_delta: torch.Tensor
    absorbed_count: torch.Tensor
    absorbed_delta_ema: torch.Tensor
    blood_centroid: torch.Tensor
    prev_blood_centroid: torch.Tensor
    blood_centroid_distance: torch.Tensor
    prev_blood_centroid_distance: torch.Tensor
    valid_in_cone_ratio: torch.Tensor
    valid_in_inlet_ratio: torch.Tensor


@dataclass(frozen=True)
class ParticleRewardInputs:
    raw_actions: torch.Tensor
    contact_force: torch.Tensor


class ParticleTaskTracker:
    def __init__(self, cfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self._num_envs = int(num_envs)
        self.device = torch.device(device)

        self._ema_alpha = float(self.cfg.absorbed_delta_ema_alpha)

        self.state = ParticleTaskState(
            absorbed_delta=self._zeros(),
            absorbed_count=self._zeros(),
            absorbed_delta_ema=self._zeros(),
            blood_centroid=self._zeros((self._num_envs, 3)),
            prev_blood_centroid=self._zeros((self._num_envs, 3)),
            blood_centroid_distance=self._zeros(),
            prev_blood_centroid_distance=self._zeros(),
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
        self.state.absorbed_delta_ema.mul_(1.0 - self._ema_alpha).add_(
            self._ema_alpha * absorbed_delta
        )

        # Update centroid and ratios
        prev_centroid = self.state.blood_centroid.clone()
        prev_distance = self.state.blood_centroid_distance.clone()
        self.state.prev_blood_centroid.copy_(prev_centroid)
        self.state.prev_blood_centroid_distance.copy_(prev_distance)

        step_count_np = self._to_numpy(step_count).astype(np.int64, copy=False)

        centroid_w = torch.from_numpy(particle_stats["blood_centroid_w"]).to(
            device=self.device, dtype=torch.float32
        )
        in_cone_ratio = torch.from_numpy(particle_stats["valid_in_cone_ratio"]).to(
            device=self.device, dtype=torch.float32
        )
        in_inlet_ratio = torch.from_numpy(particle_stats["valid_in_inlet_ratio"]).to(
            device=self.device, dtype=torch.float32
        )

        # Compute current distances (using tensor operations purely)
        current_distance = torch.linalg.vector_norm(centroid_w - tip_pos_w, dim=1)

        self.state.blood_centroid[:] = centroid_w
        self.state.blood_centroid_distance[:] = current_distance
        self.state.valid_in_cone_ratio[:] = in_cone_ratio
        self.state.valid_in_inlet_ratio[:] = in_inlet_ratio

        # Handle env_idx with step <= 0 (reset condition)
        reset_mask = torch.tensor(
            step_count_np <= 0, device=self.device, dtype=torch.bool
        )
        self.state.prev_blood_centroid[reset_mask] = centroid_w[reset_mask]
        self.state.prev_blood_centroid_distance[reset_mask] = current_distance[
            reset_mask
        ]

    def reset(self, env_ids: torch.Tensor, tip_pos_w: torch.Tensor) -> None:
        self.state.absorbed_delta[env_ids] = 0.0
        self.state.absorbed_count[env_ids] = 0.0
        self.state.absorbed_delta_ema[env_ids] = 0.0
        self.state.blood_centroid_distance[env_ids] = 0.0
        self.state.prev_blood_centroid_distance[env_ids] = 0.0
        self.state.valid_in_cone_ratio[env_ids] = 0.0
        self.state.valid_in_inlet_ratio[env_ids] = 0.0
        self.state.blood_centroid[env_ids] = tip_pos_w[env_ids]
        self.state.prev_blood_centroid[env_ids] = tip_pos_w[env_ids]
