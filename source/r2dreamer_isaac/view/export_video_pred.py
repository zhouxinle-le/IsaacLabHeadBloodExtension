from __future__ import annotations

import argparse
import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from r2dreamer_isaac.config import load_yaml, to_config
from r2dreamer_isaac.replay_buffer import IsaacReplayBuffer
from r2dreamer_isaac.vendor.r2dreamer import Dreamer


def _resolve_checkpoint(path_arg: str) -> pathlib.Path:
    path = pathlib.Path(path_arg).expanduser().resolve()
    if path.is_dir():
        candidate = path / "checkpoints" / "latest.pt"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Could not find checkpoint: {candidate}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _run_dir(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _load_config(checkpoint: dict, checkpoint_path: pathlib.Path, config_arg: str | None):
    if config_arg:
        return to_config(load_yaml(pathlib.Path(config_arg).expanduser().resolve()))
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return to_config(checkpoint["config"])
    config_path = _run_dir(checkpoint_path) / "params" / "r2dreamer.yaml"
    if config_path.is_file():
        return to_config(load_yaml(config_path))
    raise FileNotFoundError("Could not find saved r2dreamer.yaml and checkpoint has no config payload.")


def _make_spaces(data: dict[str, torch.Tensor]) -> tuple[gym.spaces.Dict, gym.spaces.Box]:
    obs_spaces = {}
    if "image" in data:
        image_shape = tuple(int(v) for v in data["image"].shape[2:])
        obs_spaces["image"] = gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
    if "position" in data:
        position_shape = tuple(int(v) for v in data["position"].shape[2:])
        obs_spaces["position"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=position_shape, dtype=np.float32)
    if "state" in data:
        state_shape = tuple(int(v) for v in data["state"].shape[2:])
        obs_spaces["state"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32)
    if "image" not in obs_spaces:
        raise ValueError("Replay sample has no 'image' observation. Use a vision Dreamer checkpoint.")

    action_dim = int(data["action"].shape[-1])
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return gym.spaces.Dict(obs_spaces), action_space


def _to_uint8(video: torch.Tensor) -> np.ndarray:
    array = video.detach().cpu().float().numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def _select_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or frames.shape[0] <= max_frames:
        return frames
    indices = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(np.int64)
    return frames[indices]


def _save_outputs(video: torch.Tensor, output_dir: pathlib.Path, batch_index: int, max_frames: int, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_uint8 = _to_uint8(video)
    if not 0 <= batch_index < video_uint8.shape[0]:
        raise IndexError(f"--batch_index must be in [0, {video_uint8.shape[0] - 1}], got {batch_index}")

    frames = _select_frames(video_uint8[batch_index], max_frames)
    montage = np.concatenate(list(frames), axis=1)
    imageio.imwrite(output_dir / "video_pred_montage.png", montage)
    imageio.mimsave(output_dir / "video_pred.gif", list(frames), fps=fps)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        imageio.imwrite(frames_dir / f"frame_{index:03d}.png", frame)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Dreamer image prediction from a saved checkpoint replay buffer."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_1_800k",
        help="Run directory or checkpoint path.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional saved r2dreamer.yaml path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for PNG/GIF outputs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used for model inference.")
    parser.add_argument("--batch_size", type=int, default=6, help="Number of replay sequences to sample.")
    parser.add_argument("--batch_length", type=int, default=32, help="Length of sampled replay sequences.")
    parser.add_argument("--batch_index", type=int, default=0, help="Which sampled sequence to visualize.")
    parser.add_argument("--max_frames", type=int, default=16, help="Maximum frames to place in the montage/GIF.")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate.")
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "replay_buffer" not in checkpoint:
        raise KeyError("This exporter needs a full latest.pt checkpoint containing replay_buffer.")

    config = _load_config(checkpoint, checkpoint_path, args.config)
    config.agent_device = args.device
    config.device = args.device
    config.model.device = args.device
    config.buffer.device = args.device
    config.buffer.batch_size = int(args.batch_size)
    config.buffer.batch_length = int(args.batch_length)
    if int(config.buffer.batch_length) <= 5:
        raise ValueError("--batch_length must be greater than 5 for open-loop image prediction.")

    replay_buffer = IsaacReplayBuffer(config.buffer)
    replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    data, _index, initial = replay_buffer.sample()

    obs_space, action_space = _make_spaces(data)
    agent = Dreamer(config.model, obs_space, action_space).to(args.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()

    with torch.no_grad():
        video = agent.video_pred(data, initial)

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _run_dir(checkpoint_path) / "video_pred"
    )
    _save_outputs(video, output_dir, int(args.batch_index), int(args.max_frames), int(args.fps))
    print(f"[INFO] Wrote image prediction montage: {output_dir / 'video_pred_montage.png'}")
    print(f"[INFO] Wrote image prediction GIF: {output_dir / 'video_pred.gif'}")
    print(f"[INFO] Wrote per-frame PNGs under: {output_dir / 'frames'}")
    print("[INFO] Layout: top=true image, middle=model prediction, bottom=prediction error.")


if __name__ == "__main__":
    main()

