from __future__ import annotations

import argparse
import csv
import json
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

from r2dreamer_isaac.config import Config, load_yaml, to_config
from r2dreamer_isaac.vendor.r2dreamer import Dreamer


def _infer_run_dir(trajectory_dir: pathlib.Path) -> pathlib.Path:
    if trajectory_dir.name == "real_trajectory":
        return trajectory_dir.parent.parent
    if (trajectory_dir / "checkpoints").is_dir() and (trajectory_dir / "params").is_dir():
        return trajectory_dir
    return trajectory_dir.parent


def _resolve_checkpoint(checkpoint_arg: str | None, trajectory_dir: pathlib.Path) -> pathlib.Path:
    if checkpoint_arg:
        path = pathlib.Path(checkpoint_arg).expanduser().resolve()
        if path.is_dir():
            path = path / "checkpoints" / "latest.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    run_dir = _infer_run_dir(trajectory_dir)
    path = run_dir / "checkpoints" / "latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Could not infer checkpoint. Pass --checkpoint explicitly. Tried: {path}")
    return path


def _run_dir_from_checkpoint(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _resolve_config(config_arg: str | None, checkpoint_path: pathlib.Path) -> pathlib.Path:
    if config_arg:
        path = pathlib.Path(config_arg).expanduser().resolve()
    else:
        path = _run_dir_from_checkpoint(checkpoint_path) / "params" / "r2dreamer.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def _set_device_fields(node, device: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "device":
                node[key] = device
            else:
                _set_device_fields(value, device)
    elif isinstance(node, list):
        for value in node:
            _set_device_fields(value, device)


def _load_config(config_path: pathlib.Path, device: str) -> Config:
    config = to_config(load_yaml(config_path))
    config.device = device
    config.agent_device = device
    if "model" not in config:
        raise KeyError(f"Saved config has no model section: {config_path}")
    _set_device_fields(config.model, device)
    return config


def _load_agent_state(checkpoint_path: pathlib.Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        return checkpoint["agent_state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def _make_spaces(trajectory: np.lib.npyio.NpzFile) -> tuple[gym.spaces.Dict, gym.spaces.Box]:
    image_shape = tuple(int(v) for v in trajectory["image"].shape[2:])
    position_shape = tuple(int(v) for v in trajectory["position"].shape[2:])
    action_dim = int(trajectory["action"].shape[-1])
    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            "position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=position_shape, dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return obs_space, action_space


def _to_tensor(array: np.ndarray, device: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(array, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _preprocess_obs(agent: Dreamer, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return agent.preprocess({key: value for key, value in obs.items()})


@torch.no_grad()
def _posterior_states(
    agent: Dreamer,
    image: torch.Tensor,
    position: torch.Tensor,
    action: torch.Tensor,
    initial_image: torch.Tensor,
    initial_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(action.shape[0])
    stoch, deter = agent.rssm.initial(batch_size)
    zero_action = torch.zeros(batch_size, action.shape[-1], dtype=torch.float32, device=action.device)
    is_first = torch.ones(batch_size, dtype=torch.bool, device=action.device)
    is_not_first = torch.zeros(batch_size, dtype=torch.bool, device=action.device)

    obs0 = {
        "image": initial_image[:, None],
        "position": initial_position[:, None],
    }
    embed0 = agent.encoder(_preprocess_obs(agent, obs0))[:, 0]
    stoch, deter, _ = agent.rssm.obs_step(stoch, deter, zero_action, embed0, is_first)

    stochs = [stoch.detach().clone()]
    deters = [deter.detach().clone()]
    for index in range(action.shape[1]):
        obs = {
            "image": image[:, index : index + 1],
            "position": position[:, index : index + 1],
        }
        embed = agent.encoder(_preprocess_obs(agent, obs))[:, 0]
        stoch, deter, _ = agent.rssm.obs_step(stoch, deter, action[:, index], embed, is_not_first)
        stochs.append(stoch.detach().clone())
        deters.append(deter.detach().clone())

    return torch.stack(stochs, dim=1), torch.stack(deters, dim=1)


def _uint8_image(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _select_montage_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or frames.shape[0] <= max_frames:
        return frames
    indices = np.linspace(0, frames.shape[0] - 1, max_frames).round().astype(np.int64)
    return frames[indices]


def _save_window_frames(
    output_dir: pathlib.Path,
    true_frames: np.ndarray,
    model_frames: np.ndarray,
    error_frames: np.ndarray,
    fps: int,
    max_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames = np.concatenate([true_frames, model_frames, error_frames], axis=1)

    for name, frames in {
        "true_frames": true_frames,
        "model_frames": model_frames,
        "error_frames": error_frames,
        "combined_frames": combined_frames,
    }.items():
        frames_dir = output_dir / name
        frames_dir.mkdir(exist_ok=True)
        for index, frame in enumerate(frames):
            imageio.imwrite(frames_dir / f"frame_{index:03d}.png", frame)

    selected = _select_montage_frames(combined_frames, max_frames)
    imageio.imwrite(output_dir / "montage.png", np.concatenate(list(selected), axis=1))
    imageio.mimsave(output_dir / "combined.gif", list(combined_frames), fps=fps)
    imageio.mimsave(output_dir / "model_prediction.gif", list(model_frames), fps=fps)
    imageio.mimsave(output_dir / "prediction_error.gif", list(error_frames), fps=fps)


@torch.no_grad()
def _predict_window(
    agent: Dreamer,
    stoch: torch.Tensor,
    deter: torch.Tensor,
    actions: torch.Tensor,
    truth_image: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prior_stoch, prior_deter = agent.rssm.imagine_with_action(stoch, deter, actions)
    model = agent.decoder(prior_stoch, prior_deter)["image"].mode()
    truth = truth_image.to(dtype=torch.float32) / 255.0
    error = (model - truth + 1.0) / 2.0

    diff = model - truth
    mse = diff.square().mean(dim=(0, 2, 3, 4))
    mae = diff.abs().mean(dim=(0, 2, 3, 4))
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-8))
    metrics = {
        "mse": mse.detach().cpu(),
        "mae": mae.detach().cpu(),
        "psnr": psnr.detach().cpu(),
    }
    video = torch.cat([truth, model, error], dim=2)
    return video, metrics


def _write_metrics_csv(path: pathlib.Path, metrics: dict[str, torch.Tensor]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon_step", "mse", "mae", "psnr"])
        writer.writeheader()
        horizon = int(metrics["mse"].shape[0])
        for index in range(horizon):
            writer.writerow(
                {
                    "horizon_step": index + 1,
                    "mse": float(metrics["mse"][index].item()),
                    "mae": float(metrics["mae"][index].item()),
                    "psnr": float(metrics["psnr"][index].item()),
                }
            )


def _write_summary_json(path: pathlib.Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pure 15-step Dreamer imagination from random positions in a saved real trajectory."
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=(
            "logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/"
            "2026-05-12_13-17-29_seed_0_600k/real_trajectory_video_pred/real_trajectory"
        ),
        help="Directory containing trajectory_data.npz from play_video_pred.py.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Run directory or checkpoint path.")
    parser.add_argument("--config", type=str, default=None, help="Optional saved r2dreamer.yaml path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for window prediction outputs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for offline model inference.")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon. Defaults to model.imag_horizon.")
    parser.add_argument("--num_windows", type=int, default=8, help="Number of random starts to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for start positions.")
    parser.add_argument(
        "--start_indices",
        type=str,
        default=None,
        help="Optional comma-separated start indices. Overrides --num_windows.",
    )
    parser.add_argument("--max_frames", type=int, default=15, help="Maximum frames in each montage.")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate.")
    args = parser.parse_args()

    trajectory_dir = pathlib.Path(args.trajectory_dir).expanduser().resolve()
    trajectory_path = trajectory_dir / "trajectory_data.npz"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory data not found: {trajectory_path}")

    checkpoint_path = _resolve_checkpoint(args.checkpoint, trajectory_dir)
    config_path = _resolve_config(args.config, checkpoint_path)
    config = _load_config(config_path, args.device)
    horizon = int(args.horizon) if args.horizon is not None else int(config.model.imag_horizon)
    if horizon <= 0:
        raise ValueError("--horizon must be positive.")

    trajectory = np.load(trajectory_path)
    obs_space, action_space = _make_spaces(trajectory)
    agent = Dreamer(config.model, obs_space, action_space).to(args.device)
    agent.load_state_dict(_load_agent_state(checkpoint_path))
    agent.eval()
    if not hasattr(agent, "decoder"):
        raise RuntimeError("This checkpoint/config has no image decoder. It must use rep_loss=dreamer.")

    image = _to_tensor(trajectory["image"], args.device)
    position = _to_tensor(trajectory["position"], args.device, dtype=torch.float32)
    action = _to_tensor(trajectory["action"], args.device, dtype=torch.float32)
    initial_image = _to_tensor(trajectory["initial_image"][None], args.device)
    initial_position = _to_tensor(trajectory["initial_position"][None], args.device, dtype=torch.float32)
    total_steps = int(action.shape[1])
    if total_steps < horizon:
        raise ValueError(f"Trajectory has only {total_steps} steps, shorter than horizon={horizon}.")

    output_dir = (
        pathlib.Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trajectory_dir.parent / f"imag_horizon_{horizon}_random_windows"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        post_stoch, post_deter = _posterior_states(
            agent=agent,
            image=image,
            position=position,
            action=action,
            initial_image=initial_image,
            initial_position=initial_position,
        )

    valid_starts = np.arange(0, total_steps - horizon + 1, dtype=np.int64)
    if args.start_indices:
        starts = np.array([int(item.strip()) for item in args.start_indices.split(",") if item.strip()], dtype=np.int64)
        invalid = starts[(starts < 0) | (starts > total_steps - horizon)]
        if invalid.size:
            raise ValueError(f"Invalid start indices for horizon={horizon}: {invalid.tolist()}")
    else:
        rng = np.random.default_rng(int(args.seed))
        replace = int(args.num_windows) > len(valid_starts)
        starts = rng.choice(valid_starts, size=int(args.num_windows), replace=replace)
        starts = np.sort(starts)

    window_rows = []
    per_step_mse = []
    per_step_mae = []
    per_step_psnr = []

    for window_index, start in enumerate(starts.tolist()):
        window_dir = output_dir / f"window_{window_index:03d}_start_{start:04d}"
        actions = action[:, start : start + horizon]
        truth = image[:, start : start + horizon]
        stoch = post_stoch[:, start]
        deter = post_deter[:, start]

        video, metrics = _predict_window(agent, stoch, deter, actions, truth)
        video_np = _uint8_image(video[0])
        frame_height = video_np.shape[1] // 3
        true_frames = video_np[:, :frame_height]
        model_frames = video_np[:, frame_height : 2 * frame_height]
        error_frames = video_np[:, 2 * frame_height : 3 * frame_height]
        _save_window_frames(window_dir, true_frames, model_frames, error_frames, int(args.fps), int(args.max_frames))
        _write_metrics_csv(window_dir / "metrics.csv", metrics)

        mse = metrics["mse"].numpy()
        mae = metrics["mae"].numpy()
        psnr = metrics["psnr"].numpy()
        per_step_mse.append(mse)
        per_step_mae.append(mae)
        per_step_psnr.append(psnr)
        summary = {
            "window_index": int(window_index),
            "start_index": int(start),
            "horizon": int(horizon),
            "mse_mean": float(mse.mean()),
            "mse_final": float(mse[-1]),
            "mae_mean": float(mae.mean()),
            "psnr_mean": float(psnr.mean()),
            "psnr_final": float(psnr[-1]),
        }
        _write_summary_json(window_dir / "summary.json", summary)
        window_rows.append(summary)

    with (output_dir / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["window_index", "start_index", "horizon", "mse_mean", "mse_final", "mae_mean", "psnr_mean", "psnr_final"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_rows)

    mse_arr = np.stack(per_step_mse, axis=0)
    mae_arr = np.stack(per_step_mae, axis=0)
    psnr_arr = np.stack(per_step_psnr, axis=0)
    with (output_dir / "aggregate_by_horizon_step.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "horizon_step",
            "mse_mean",
            "mse_std",
            "mae_mean",
            "mae_std",
            "psnr_mean",
            "psnr_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(horizon):
            writer.writerow(
                {
                    "horizon_step": step + 1,
                    "mse_mean": float(mse_arr[:, step].mean()),
                    "mse_std": float(mse_arr[:, step].std()),
                    "mae_mean": float(mae_arr[:, step].mean()),
                    "mae_std": float(mae_arr[:, step].std()),
                    "psnr_mean": float(psnr_arr[:, step].mean()),
                    "psnr_std": float(psnr_arr[:, step].std()),
                }
            )

    _write_summary_json(
        output_dir / "summary.json",
        {
            "trajectory_path": str(trajectory_path),
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "total_steps": int(total_steps),
            "horizon": int(horizon),
            "num_windows": int(len(starts)),
            "starts": [int(value) for value in starts.tolist()],
            "mse_mean": float(mse_arr.mean()),
            "mse_final_mean": float(mse_arr[:, -1].mean()),
            "mae_mean": float(mae_arr.mean()),
            "psnr_mean": float(psnr_arr.mean()),
            "psnr_final_mean": float(psnr_arr[:, -1].mean()),
        },
    )
    print(f"[INFO] Wrote random-start horizon predictions to: {output_dir}")
    print(f"[INFO] Starts: {[int(value) for value in starts.tolist()]}")
    print(f"[INFO] Horizon: {horizon}")
    print("[INFO] Per-window layout: top=true image, middle=model prediction, bottom=prediction error.")


if __name__ == "__main__":
    main()
