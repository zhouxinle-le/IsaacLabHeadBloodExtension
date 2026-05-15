from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EpisodeStorage:
    episode_id: int
    finalized: bool = False
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def append(self, transition: dict[str, torch.Tensor]) -> None:
        if not self.data:
            self.data = {key: [] for key in transition}
        for key, value in transition.items():
            self.data[key].append(value.detach().clone())

    def __len__(self) -> int:
        if not self.data:
            return 0
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    def get(self, key: str, index: int) -> torch.Tensor:
        return self.data[key][index]

    def stack(self, key: str, start: int, length: int) -> torch.Tensor:
        return torch.stack(self.data[key][start : start + length], dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "finalized": self.finalized,
            "data": {key: torch.stack(values, dim=0) for key, values in self.data.items()},
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "EpisodeStorage":
        episode = cls(int(payload["episode_id"]), bool(payload["finalized"]))
        episode.data = {
            key: [tensor[index].clone() for index in range(tensor.shape[0])]
            for key, tensor in payload["data"].items()
        }
        return episode


class IsaacReplayBuffer:
    """Episode-aware replay buffer for Isaac auto-reset environments."""

    _NON_OBSERVATION_KEYS = {
        "obs",
        "next_obs",
        "action",
        "reward",
        "cost",
        "is_first",
        "is_last",
        "is_terminal",
        "stoch",
        "deter",
    }

    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.max_size = int(float(config.max_size))

        self._episodes: dict[int, EpisodeStorage] = {}
        self._finalized_order: deque[int] = deque()
        self._num_transitions = 0

    def add_transition(self, batch: dict[str, torch.Tensor]) -> None:
        episode_ids = batch["episode"].detach().cpu().tolist()
        for env_index, episode_id in enumerate(episode_ids):
            episode_id = int(episode_id)
            episode = self._episodes.get(episode_id)
            if episode is None:
                episode = EpisodeStorage(episode_id=episode_id)
                self._episodes[episode_id] = episode

            row: dict[str, torch.Tensor] = {}
            for key, value in batch.items():
                if key == "episode":
                    continue
                row[key] = value[env_index].detach().to(self.storage_device)
            episode.append(row)
            self._num_transitions += 1

            if bool(row["is_last"].item()) and not episode.finalized:
                episode.finalized = True
                self._finalized_order.append(episode_id)

        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self._num_transitions > self.max_size and self._finalized_order:
            oldest_episode_id = self._finalized_order.popleft()
            episode = self._episodes.pop(oldest_episode_id, None)
            if episode is not None:
                self._num_transitions -= len(episode)

    def _eligible_episodes(self) -> list[EpisodeStorage]:
        return [episode for episode in self._episodes.values() if len(episode) >= self.batch_length]

    def can_sample(self) -> bool:
        return bool(self._eligible_episodes())

    def _observation_storage_keys(self, episode: EpisodeStorage) -> list[str]:
        keys = [key for key in episode.data if key not in self._NON_OBSERVATION_KEYS]
        if keys:
            return keys
        # Backward compatibility with early state-only checkpoints/tests.
        if "next_obs" in episode.data:
            return ["next_obs"]
        raise RuntimeError("Episode does not contain any Dreamer observation tensors.")

    @staticmethod
    def _sample_output_key(storage_key: str) -> str:
        return "state" if storage_key == "next_obs" else storage_key

    @staticmethod
    def _normalize_sample_dtype(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(dtype=torch.float32)
        return tensor

    def sample(self):
        eligible = self._eligible_episodes()
        if not eligible:
            raise RuntimeError("Replay buffer does not contain enough data to sample a batch.")

        weights = torch.tensor(
            [len(episode) - self.batch_length + 1 for episode in eligible],
            dtype=torch.float32,
        )
        sampled_episode_indices = torch.multinomial(weights, self.batch_size, replacement=True).tolist()

        observation_keys = self._observation_storage_keys(eligible[0])
        observation_batches = {key: [] for key in observation_keys}
        action_batch = []
        reward_batch = []
        cost_batch = []
        is_first_batch = []
        is_last_batch = []
        is_terminal_batch = []
        initial_stoch = []
        initial_deter = []
        sampled_episode_ids: list[int] = []
        sampled_starts: list[int] = []

        for sampled_index in sampled_episode_indices:
            episode = eligible[sampled_index]
            max_start = len(episode) - self.batch_length
            start = random.randint(0, max_start)

            for key in observation_keys:
                observation_batches[key].append(episode.stack(key, start, self.batch_length))
            action_batch.append(episode.stack("action", start, self.batch_length))
            reward_batch.append(episode.stack("reward", start, self.batch_length))
            if "cost" in episode.data:
                cost_batch.append(episode.stack("cost", start, self.batch_length))
            else:
                cost_batch.append(torch.zeros_like(reward_batch[-1]))
            is_last_batch.append(episode.stack("is_last", start, self.batch_length))
            is_terminal_batch.append(episode.stack("is_terminal", start, self.batch_length))
            initial_stoch.append(episode.get("stoch", start))
            initial_deter.append(episode.get("deter", start))

            next_is_first = []
            for offset in range(self.batch_length):
                next_index = start + offset + 1
                if next_index < len(episode):
                    next_is_first.append(episode.get("is_first", next_index).to(dtype=torch.bool))
                else:
                    next_is_first.append(torch.zeros((), dtype=torch.bool, device=self.storage_device))
            is_first_batch.append(torch.stack(next_is_first, dim=0))

            sampled_episode_ids.append(episode.episode_id)
            sampled_starts.append(start)

        def _to_device(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=self.device, non_blocking=True)

        reward = _to_device(torch.stack(reward_batch, dim=0)).to(dtype=torch.float32)
        cost = _to_device(torch.stack(cost_batch, dim=0)).to(dtype=torch.float32)
        is_first = _to_device(torch.stack(is_first_batch, dim=0)).to(dtype=torch.bool)
        is_last = _to_device(torch.stack(is_last_batch, dim=0)).to(dtype=torch.bool)
        is_terminal = _to_device(torch.stack(is_terminal_batch, dim=0)).to(dtype=torch.bool)

        if reward.ndim == 2:
            reward = reward.unsqueeze(-1)
        if cost.ndim == 2:
            cost = cost.unsqueeze(-1)
        if is_last.ndim == 2:
            is_last = is_last.unsqueeze(-1)
        if is_terminal.ndim == 2:
            is_terminal = is_terminal.unsqueeze(-1)

        data = {
            "action": _to_device(torch.stack(action_batch, dim=0)).to(dtype=torch.float32),
            "reward": reward,
            "cost": cost,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
        for storage_key, values in observation_batches.items():
            output_key = self._sample_output_key(storage_key)
            data[output_key] = self._normalize_sample_dtype(_to_device(torch.stack(values, dim=0)))

        initial = (
            _to_device(torch.stack(initial_stoch, dim=0)).to(dtype=torch.float32),
            _to_device(torch.stack(initial_deter, dim=0)).to(dtype=torch.float32),
        )
        index = {"episode_ids": sampled_episode_ids, "starts": sampled_starts}
        return data, index, initial

    def update(self, index, stoch: torch.Tensor, deter: torch.Tensor) -> None:
        stoch = stoch.detach().to(self.storage_device)
        deter = deter.detach().to(self.storage_device)
        for batch_index, (episode_id, start) in enumerate(zip(index["episode_ids"], index["starts"])):
            episode = self._episodes.get(int(episode_id))
            if episode is None:
                continue
            for offset in range(stoch.shape[1]):
                next_index = int(start) + offset + 1
                if next_index >= len(episode):
                    continue
                episode.data["stoch"][next_index] = stoch[batch_index, offset].clone()
                episode.data["deter"][next_index] = deter[batch_index, offset].clone()

    def count(self) -> int:
        return self._num_transitions

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "batch_length": self.batch_length,
            "max_size": self.max_size,
            "num_transitions": self._num_transitions,
            "episodes": [episode.state_dict() for episode in self._episodes.values()],
            "finalized_order": list(self._finalized_order),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.batch_size = int(payload["batch_size"])
        self.batch_length = int(payload["batch_length"])
        self.max_size = int(payload["max_size"])
        self._num_transitions = int(payload["num_transitions"])
        self._episodes = {}
        for episode_payload in payload["episodes"]:
            episode = EpisodeStorage.from_state_dict(episode_payload)
            self._episodes[episode.episode_id] = episode
        self._finalized_order = deque(int(ep_id) for ep_id in payload["finalized_order"])
