from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None


class BloodPipeVisionObservationManager:
    """Build wrist-camera and non-privileged proprioceptive observations."""

    _POSITION_OBSERVATION_DIM = 1

    def __init__(self, cfg, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._camera: Any | None = None
        self._show_policy_input_image = bool(getattr(self.cfg, "show_policy_input_image", False))
        self._policy_input_window_name = str(getattr(self.cfg, "policy_input_window_name", "Policy Input - Env 0"))
        self._policy_input_display_available = cv2 is not None and (
            os.environ.get("DISPLAY") is not None or os.environ.get("WAYLAND_DISPLAY") is not None
        )
        self._policy_input_warning_emitted = False

        self._obs_camera = torch.zeros(
            (
                self.num_envs,
                int(self.cfg.num_channels),
                int(self.cfg.obs_camera_height),
                int(self.cfg.obs_camera_width),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._obs_position = torch.zeros(
            (self.num_envs, int(self.cfg.position_observation_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        if int(self.cfg.position_observation_dim) != self._POSITION_OBSERVATION_DIM:
            raise ValueError(
                "blood_pipe_vision_ur3 position_observation_dim must match the assembled proprioception "
                f"features ({self._POSITION_OBSERVATION_DIM}), got {self.cfg.position_observation_dim}"
            )

    def bind_runtime(self, camera: Any) -> None:
        self._camera = camera

    def reset(self, env_ids: torch.Tensor) -> None:
        self._obs_camera[env_ids] = 0.0
        self._obs_position[env_ids] = 0.0

    def _build_camera_observation(self) -> None:
        if self._camera is None:
            self._obs_camera.zero_()
            return

        camera_data = self._camera.data.output.get("rgb")
        if camera_data is None or camera_data.numel() == 0:
            self._obs_camera.zero_()
            return

        rgb = camera_data[..., :3]
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        else:
            if rgb.dtype != torch.float32:
                rgb = rgb.float()
            if rgb.numel() > 0 and torch.max(rgb) > 1.0:
                rgb = rgb / 255.0

        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()
        target_size = (int(self.cfg.obs_camera_height), int(self.cfg.obs_camera_width))
        if tuple(rgb_nchw.shape[-2:]) == target_size:
            self._obs_camera[:] = rgb_nchw
        else:
            self._obs_camera[:] = torch.nn.functional.interpolate(
                rgb_nchw,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        self._obs_camera.clamp_(0.0, 1.0)

    def _display_policy_input_image(self) -> None:
        if not self._show_policy_input_image:
            return
        if not self._policy_input_display_available:
            if not self._policy_input_warning_emitted:
                print("[WARN] Policy input image display disabled because OpenCV GUI/display is unavailable.")
                self._policy_input_warning_emitted = True
            return
        if self._obs_camera.numel() == 0 or self._obs_camera.shape[0] == 0:
            return

        img = self._obs_camera[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        try:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(self._policy_input_window_name, img_bgr)
            cv2.waitKey(1)
        except cv2.error as exc:
            if not self._policy_input_warning_emitted:
                print(f"[WARN] Failed to display policy input image: {exc}")
                self._policy_input_warning_emitted = True
            self._policy_input_display_available = False

    def update(self, position_obs: torch.Tensor) -> None:
        if position_obs.shape != self._obs_position.shape:
            raise ValueError(
                f"Expected proprio observation shape {tuple(self._obs_position.shape)}, "
                f"got {tuple(position_obs.shape)}"
            )
        self._build_camera_observation()
        self._display_policy_input_image()
        self._obs_position[:] = position_obs

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {
            "camera": self._obs_camera.clone(),
            "position": self._obs_position.clone(),
        }

    def close(self) -> None:
        if not self._show_policy_input_image or cv2 is None:
            return
        try:
            cv2.destroyWindow(self._policy_input_window_name)
        except cv2.error:
            return
