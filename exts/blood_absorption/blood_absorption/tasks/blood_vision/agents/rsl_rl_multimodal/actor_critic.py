from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation.__class__())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class BloodVisionActorCritic(nn.Module):
    """Actor-critic that reconstructs camera and proprio inputs from a flat observation tensor."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        cnn_activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        camera_shape: Sequence[int] = (3, 128, 128),
        position_dim: int = 11,
        cnn_channels: Sequence[int] = (32, 64, 64),
        **kwargs,
    ):
        if kwargs:
            print(
                "BloodVisionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.camera_shape = tuple(int(v) for v in camera_shape)
        self.position_dim = int(position_dim)
        self.camera_obs_dim = math.prod(self.camera_shape)
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)

        expected_obs_dim = self.camera_obs_dim + self.position_dim
        if self.num_actor_obs != expected_obs_dim:
            raise ValueError(
                f"Actor observation dimension mismatch. Expected {expected_obs_dim}, got {self.num_actor_obs}."
            )
        if self.num_critic_obs != expected_obs_dim:
            raise ValueError(
                f"Critic observation dimension mismatch. Expected {expected_obs_dim}, got {self.num_critic_obs}."
            )

        activation_module = resolve_nn_activation(activation)
        cnn_activation_module = resolve_nn_activation(cnn_activation)
        cnn_channels = [int(v) for v in cnn_channels]
        if len(cnn_channels) != 3:
            raise ValueError(f"cnn_channels must contain exactly 3 entries, got {cnn_channels}.")

        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.camera_shape[0], cnn_channels[0], kernel_size=10, stride=5, padding=0),
            cnn_activation_module.__class__(),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=3, padding=0),
            cnn_activation_module.__class__(),
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=4, stride=2, padding=0),
            cnn_activation_module.__class__(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_camera = torch.zeros((1, *self.camera_shape), dtype=torch.float32)
            image_feature_dim = int(self.image_encoder(sample_camera).shape[-1])

        fused_feature_dim = image_feature_dim + self.position_dim
        actor_output_dim = int(actor_hidden_dims[-1]) if len(actor_hidden_dims) > 0 else self.num_actions
        critic_output_dim = int(critic_hidden_dims[-1]) if len(critic_hidden_dims) > 0 else 1

        self.actor_backbone = _build_mlp(
            input_dim=fused_feature_dim,
            hidden_dims=list(actor_hidden_dims[:-1]),
            output_dim=actor_output_dim,
            activation=activation_module,
        )
        self.actor_head = nn.Linear(actor_output_dim, self.num_actions)

        self.critic_backbone = _build_mlp(
            input_dim=fused_feature_dim,
            hidden_dims=list(critic_hidden_dims[:-1]),
            output_dim=critic_output_dim,
            activation=activation_module,
        )
        self.critic_head = nn.Linear(critic_output_dim, 1)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _split_obs(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        camera_flat = observations[:, : self.camera_obs_dim]
        position = observations[:, self.camera_obs_dim : self.camera_obs_dim + self.position_dim]
        camera = camera_flat.reshape(-1, *self.camera_shape)
        return camera, position

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        camera, position = self._split_obs(observations)
        image_features = self.image_encoder(camera)
        return torch.cat((image_features, position), dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        features = self._encode(observations)
        mean = self.actor_head(self.actor_backbone(features))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        features = self._encode(observations)
        return self.actor_head(self.actor_backbone(features))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        features = self._encode(critic_observations)
        return self.critic_head(self.critic_backbone(features))
