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

from omni.isaac.lab.app import AppLauncher


def _candidate_saved_config_path(checkpoint_arg: str, config_arg: str | None) -> pathlib.Path | None:
    if config_arg:
        return pathlib.Path(config_arg).expanduser()

    checkpoint_path = pathlib.Path(checkpoint_arg).expanduser()
    if checkpoint_path.is_dir():
        return checkpoint_path / "params" / "r2dreamer.yaml"
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "params" / "r2dreamer.yaml"
    return checkpoint_path.parent / "params" / "r2dreamer.yaml"


def _saved_config_mentions_vision(checkpoint_arg: str, config_arg: str | None) -> bool:
    config_path = _candidate_saved_config_path(checkpoint_arg, config_arg)
    if config_path is None or not config_path.is_file():
        return False
    return "vision" in config_path.read_text(encoding="utf-8", errors="ignore").lower()


parser = argparse.ArgumentParser(
    description="Replay an Isaac R2-Dreamer policy and export world-model image prediction."
)
parser.add_argument("--task", type=str, default=None, help="Gym task name. Defaults to the saved run config.")
parser.add_argument(
    "--cfg_entry_point",
    type=str,
    default="dreamer_cfg_entry_point",
    help="Registry key for the Dreamer-family config to load when no saved run config is provided.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-12_13-17-29_seed_0_600k",
    help="Checkpoint file or run directory to load.",
)
parser.add_argument("--config", type=str, default=None, help="Optional path to a saved r2dreamer.yaml config.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This exporter uses env 0.")
parser.add_argument("--seed", type=int, default=None, help="Optional seed used for the replay environment.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for PNG/GIF outputs.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=300,
    help="Maximum environment steps to collect for the real trajectory.",
)
parser.add_argument(
    "--prediction_length",
    type=int,
    default=0,
    help="Number of trajectory steps used for world-model prediction. Use 0 for the full collected trajectory.",
)
parser.add_argument(
    "--batch_length",
    type=int,
    default=None,
    help="Deprecated alias for --prediction_length, kept for old commands.",
)
parser.add_argument("--max_frames", type=int, default=16, help="Maximum frames to place in the montage/GIF.")
parser.add_argument("--fps", type=int, default=8, help="GIF frame rate.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=8,
    help="Deprecated and ignored. Kept so old commands still parse.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
parser.add_argument(
    "--agent_device",
    type=str,
    default=None,
    help="Device for the Dreamer policy. Defaults to --device when omitted.",
)
parser.add_argument(
    "--env_device",
    type=str,
    default=None,
    help="Device for the Isaac environment. Defaults to --device when omitted.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, overrides = parser.parse_known_args()
uses_vision_task = "vision" in str(args_cli.task).lower()
uses_vision_config = _saved_config_mentions_vision(args_cli.checkpoint, args_cli.config)
if uses_vision_task or uses_vision_config:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import head_blood_absorption.tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from r2dreamer_isaac.config import build_runtime_config, load_yaml
from r2dreamer_isaac.env_adapter import IsaacR2DreamerEnvAdapter, obs_to_device
from r2dreamer_isaac.export_video_pred import _save_outputs, _to_uint8
from r2dreamer_isaac.vendor.r2dreamer import Dreamer


def _resolve_checkpoint_path(checkpoint_arg: str) -> pathlib.Path:
    checkpoint_path = pathlib.Path(checkpoint_arg).expanduser().resolve()
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "checkpoints" / "latest.pt"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Could not find latest checkpoint under: {candidate}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _run_dir_from_checkpoint(checkpoint_path: pathlib.Path) -> pathlib.Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _find_saved_config_path(checkpoint_path: pathlib.Path) -> pathlib.Path | None:
    run_dir = _run_dir_from_checkpoint(checkpoint_path)
    candidate = run_dir / "params" / "r2dreamer.yaml"
    if candidate.is_file():
        return candidate
    return None


def _load_task_cfg(task_name: str | None, config_path: pathlib.Path | None) -> tuple[str, dict]:
    if config_path is not None:
        task_cfg = load_yaml(config_path)
    else:
        if not task_name:
            raise ValueError("Either --task or --config/--checkpoint with a saved run config is required.")
        task_cfg = load_cfg_from_registry(task_name, args_cli.cfg_entry_point)

    resolved_task = task_name or task_cfg.get("env", {}).get("task")
    if not resolved_task:
        raise KeyError("Unable to determine task name from the provided config. Pass --task explicitly.")
    return resolved_task, task_cfg


def _apply_env_cfg_overrides(env_cfg, cfg_overrides) -> None:
    if not cfg_overrides:
        return
    for name, value in cfg_overrides.items():
        if not hasattr(env_cfg, name):
            raise AttributeError(f"Environment config has no field '{name}' for env.cfg_overrides.")
        setattr(env_cfg, name, value)
        print(f"[INFO] Applying env cfg override: {name}={value}")


def _load_agent_state(checkpoint_path: pathlib.Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint:
        return checkpoint["agent_state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def _detach_state(agent_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in agent_state.items()}


def _image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _tensor_env0(value: torch.Tensor, env_index: int = 0) -> torch.Tensor:
    if value.ndim > 0 and value.shape[0] > env_index:
        return value[env_index]
    return value


def _tensor_scalar(value: torch.Tensor | float | int | bool, env_index: int = 0) -> float | bool:
    if isinstance(value, torch.Tensor):
        tensor = _tensor_env0(value.detach().cpu(), env_index)
        if tensor.numel() == 1:
            item = tensor.item()
            return bool(item) if tensor.dtype == torch.bool else float(item)
        return float(tensor.float().mean().item())
    if isinstance(value, bool):
        return value
    return float(value)


def _collect_real_trajectory(
    adapter: IsaacR2DreamerEnvAdapter,
    agent: Dreamer,
    device: str,
    max_steps: int,
    env_index: int = 0,
) -> dict[str, object]:
    if max_steps <= 5:
        raise ValueError("--max_steps must be greater than 5 for open-loop image prediction.")
    if adapter.modality != "vision":
        raise ValueError("This exporter needs a vision task with an 'image' observation.")
    if not 0 <= env_index < adapter.num_envs:
        raise IndexError(f"env_index must be in [0, {adapter.num_envs - 1}], got {env_index}")

    current_obs, reset_extras = adapter.reset()
    current_obs = obs_to_device(current_obs, device)
    current_is_first = torch.ones(adapter.num_envs, dtype=torch.bool, device=device)
    agent_state = _detach_state(agent.get_initial_state(adapter.num_envs))
    initial_observation = {
        key: _tensor_env0(value.detach(), env_index).cpu()
        for key, value in adapter.observation_items(current_obs).items()
    }

    initial: tuple[torch.Tensor, torch.Tensor] | None = None
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    costs: list[torch.Tensor] = []
    is_firsts: list[torch.Tensor] = []
    is_lasts: list[torch.Tensor] = []
    is_terminals: list[torch.Tensor] = []
    observations: dict[str, list[torch.Tensor]] = {key: [] for key in adapter.observation_keys}
    step_records: list[dict[str, object]] = []
    completed = False
    final_log = {}

    for step in range(max_steps):
        with torch.inference_mode():
            action, next_agent_state = agent.act(
                adapter.build_agent_obs(current_obs, current_is_first),
                agent_state,
                eval=True,
            )
            if initial is None:
                initial = (
                    next_agent_state["stoch"][env_index : env_index + 1].detach().clone(),
                    next_agent_state["deter"][env_index : env_index + 1].detach().clone(),
                )
            step_out = adapter.step(action)

        aligned_next_obs = obs_to_device(step_out.aligned_next_obs, device)
        aligned_items = adapter.observation_items(aligned_next_obs)
        for key, value in aligned_items.items():
            observations[key].append(value[env_index : env_index + 1].detach().cpu())
        actions.append(action[env_index : env_index + 1].detach().cpu())
        rewards.append(step_out.reward[env_index : env_index + 1].detach().cpu())
        costs.append(step_out.cost[env_index : env_index + 1].detach().cpu())
        # The stored image is obs_{t+1}. Within a single episode, it is not an episode-first observation.
        is_firsts.append(torch.zeros(1, dtype=torch.bool))
        is_lasts.append(step_out.done[env_index : env_index + 1].detach().cpu())
        is_terminals.append(step_out.terminated[env_index : env_index + 1].detach().cpu())

        record: dict[str, object] = {
            "step": int(step),
            "reward": _tensor_scalar(step_out.reward, env_index),
            "cost": _tensor_scalar(step_out.cost, env_index),
            "done": _tensor_scalar(step_out.done, env_index),
            "terminated": _tensor_scalar(step_out.terminated, env_index),
            "truncated": _tensor_scalar(step_out.truncated, env_index),
            "action_norm": float(action[env_index].detach().float().norm().cpu().item()),
        }
        action_values = action[env_index].detach().float().cpu().tolist()
        for index, value in enumerate(action_values):
            record[f"action/{index}"] = float(value)
        if "position" in aligned_items:
            position_values = aligned_items["position"][env_index].detach().float().cpu().tolist()
            for index, value in enumerate(position_values):
                record[f"position/{index}"] = float(value)
        safety_terms = step_out.extras.get("safety_terms", {}) if isinstance(step_out.extras, dict) else {}
        for name, value in safety_terms.items():
            if isinstance(value, torch.Tensor):
                record[f"safety/{name}"] = _tensor_scalar(value, env_index)
        step_records.append(record)

        final_log = step_out.extras.get("log", {}) if isinstance(step_out.extras, dict) else {}
        completed = bool(step_out.done[env_index].detach().cpu().item())
        if completed:
            break

        current_obs = obs_to_device(step_out.obs, device)
        current_is_first = step_out.done.to(device=device)
        agent_state = _detach_state(next_agent_state)

    if initial is None:
        raise RuntimeError("No trajectory steps were collected.")

    data = {
        "action": torch.stack(actions, dim=1).to(device=device, dtype=torch.float32),
        "reward": torch.stack(rewards, dim=1).to(device=device, dtype=torch.float32),
        "cost": torch.stack(costs, dim=1).to(device=device, dtype=torch.float32),
        "is_first": torch.stack(is_firsts, dim=1).to(device=device, dtype=torch.bool),
        "is_last": torch.stack(is_lasts, dim=1).to(device=device, dtype=torch.bool).unsqueeze(-1),
        "is_terminal": torch.stack(is_terminals, dim=1).to(device=device, dtype=torch.bool).unsqueeze(-1),
    }
    for key, values in observations.items():
        data[key] = torch.stack(values, dim=1).to(device=device, non_blocking=True)

    return {
        "data": data,
        "initial": (initial[0].to(device=device), initial[1].to(device=device)),
        "initial_observation": initial_observation,
        "step_records": step_records,
        "completed": completed,
        "length": int(data["action"].shape[1]),
        "reset_extras": reset_extras,
        "final_log": final_log,
    }


def _time_slice(data: dict[str, torch.Tensor], length: int) -> dict[str, torch.Tensor]:
    if length <= 0:
        return data
    sliced = {}
    for key, value in data.items():
        if value.ndim >= 2 and value.shape[1] >= length:
            sliced[key] = value[:, :length]
        else:
            sliced[key] = value
    return sliced


def _save_real_trajectory_outputs(trajectory: dict[str, object], output_dir: pathlib.Path, fps: int) -> None:
    real_dir = output_dir / "real_trajectory"
    frames_dir = real_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    data = trajectory["data"]
    assert isinstance(data, dict)
    initial_observation = trajectory["initial_observation"]
    assert isinstance(initial_observation, dict)

    if "image" in initial_observation:
        imageio.imwrite(real_dir / "initial_image.png", _image_to_uint8(initial_observation["image"]))
    if "position" in initial_observation:
        np.savetxt(real_dir / "initial_position.txt", initial_observation["position"].numpy()[None], fmt="%.8f")

    real_frames = []
    for index, image in enumerate(data["image"][0]):
        frame = _image_to_uint8(image)
        real_frames.append(frame)
        imageio.imwrite(frames_dir / f"frame_{index:04d}.png", frame)
    if real_frames:
        imageio.mimsave(real_dir / "real_trajectory.gif", real_frames, fps=fps)

    arrays = {}
    for key, value in data.items():
        arrays[key] = value.detach().cpu().numpy()
    if "image" in initial_observation:
        arrays["initial_image"] = _image_to_uint8(initial_observation["image"])
    if "position" in initial_observation:
        arrays["initial_position"] = initial_observation["position"].numpy()
    np.savez_compressed(real_dir / "trajectory_data.npz", **arrays)

    step_records = trajectory["step_records"]
    assert isinstance(step_records, list)
    if step_records:
        fieldnames = sorted({key for record in step_records for key in record})
        if "step" in fieldnames:
            fieldnames.remove("step")
            fieldnames.insert(0, "step")
        with (real_dir / "steps.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(step_records)

    final_log = trajectory.get("final_log", {})
    summary = {
        "length": int(trajectory["length"]),
        "completed_episode": bool(trajectory["completed"]),
        "frame_count": len(real_frames),
        "final_log_keys": sorted(final_log.keys()) if isinstance(final_log, dict) else [],
    }
    with (real_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _save_prediction_outputs(video: torch.Tensor, output_dir: pathlib.Path, max_frames: int, fps: int) -> None:
    pred_dir = output_dir / "world_model_prediction"
    _save_outputs(video, pred_dir, batch_index=0, max_frames=max_frames, fps=fps)

    video_uint8 = _to_uint8(video)[0]
    height = video_uint8.shape[1] // 3
    true_frames = video_uint8[:, :height]
    model_frames = video_uint8[:, height : 2 * height]
    error_frames = video_uint8[:, 2 * height : 3 * height]

    for name, frames in {
        "true_frames": true_frames,
        "model_frames": model_frames,
        "error_frames": error_frames,
        "combined_frames": video_uint8,
    }.items():
        frames_dir = pred_dir / name
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            imageio.imwrite(frames_dir / f"frame_{index:04d}.png", frame)

    imageio.mimsave(pred_dir / "true.gif", list(true_frames), fps=fps)
    imageio.mimsave(pred_dir / "model_prediction.gif", list(model_frames), fps=fps)
    imageio.mimsave(pred_dir / "prediction_error.gif", list(error_frames), fps=fps)


def main() -> None:
    checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
    config_path = pathlib.Path(args_cli.config).expanduser().resolve() if args_cli.config else None
    if config_path is None:
        config_path = _find_saved_config_path(checkpoint_path)

    task_name, task_cfg = _load_task_cfg(args_cli.task, config_path)
    env_device = args_cli.env_device or args_cli.device
    agent_device = args_cli.agent_device or args_cli.device

    cli_updates = {
        "device": agent_device,
        "agent_device": agent_device,
        "env_device": env_device,
        "env": {"num_envs": int(args_cli.num_envs)},
    }
    if args_cli.seed is not None:
        cli_updates["seed"] = args_cli.seed

    config = build_runtime_config(task_cfg=task_cfg, cli_updates=cli_updates, dotlist_overrides=overrides)
    run_dir = _run_dir_from_checkpoint(checkpoint_path)
    output_dir = (
        pathlib.Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else run_dir / "real_trajectory_video_pred"
    )

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    print(f"[INFO] Using task: {task_name}")
    print(f"[INFO] Devices: env={config.env_device}, agent={config.agent_device}")
    if config_path is not None:
        print(f"[INFO] Loaded saved config from: {config_path}")

    env_cfg = parse_env_cfg(
        task_name,
        device=str(config.env_device),
        num_envs=int(config.env.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    _apply_env_cfg_overrides(env_cfg, getattr(config.env, "cfg_overrides", {}))

    env = gym.make(task_name, cfg=env_cfg)
    adapter = IsaacR2DreamerEnvAdapter(env)
    agent = Dreamer(config.model, adapter.observation_space, adapter.action_space).to(config.agent_device)
    agent.load_state_dict(_load_agent_state(checkpoint_path))
    agent.eval()

    try:
        trajectory = _collect_real_trajectory(
            adapter=adapter,
            agent=agent,
            device=str(config.agent_device),
            max_steps=int(args_cli.max_steps),
        )
        _save_real_trajectory_outputs(trajectory, output_dir, fps=int(args_cli.fps))

        data = trajectory["data"]
        initial = trajectory["initial"]
        assert isinstance(data, dict)
        assert isinstance(initial, tuple)
        prediction_length = int(args_cli.prediction_length)
        if args_cli.batch_length is not None and prediction_length <= 0:
            prediction_length = int(args_cli.batch_length)
        if prediction_length > 0:
            data = _time_slice(data, prediction_length)
        if data["action"].shape[1] <= 5:
            raise RuntimeError(
                f"Need at least 6 trajectory steps for Dreamer.video_pred, got {data['action'].shape[1]}."
            )

        with torch.inference_mode():
            video = agent.video_pred(data, initial)
        _save_prediction_outputs(video, output_dir, max_frames=int(args_cli.max_frames), fps=int(args_cli.fps))
        print(f"[INFO] Wrote real trajectory under: {output_dir / 'real_trajectory'}")
        print(f"[INFO] Wrote world-model prediction under: {output_dir / 'world_model_prediction'}")
        print(f"[INFO] Trajectory length: {trajectory['length']} steps")
        print(f"[INFO] Completed episode: {trajectory['completed']}")
        print("[INFO] Layout: top=true image, middle=model prediction, bottom=prediction error.")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
