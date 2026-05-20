from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from datetime import datetime

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


parser = argparse.ArgumentParser(description="Play or evaluate a checkpoint of the Isaac R2-Dreamer agent.")
parser.add_argument("--task", type=str, default=None, help="Gym task name. Defaults to the saved run config.")
parser.add_argument(
    "--cfg_entry_point",
    type=str,
    default="dreamer_cfg_entry_point",
    help="Registry key for the Dreamer-family config to load when no saved run config is provided.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or run directory to load.")
parser.add_argument("--config", type=str, default=None, help="Optional path to a saved r2dreamer.yaml config.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate. Use 0 for unlimited.")
parser.add_argument("--seed", type=int, default=None, help="Optional seed used for the evaluation environment.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write episode_summary.csv and summary.json. Defaults to <run>/eval/<timestamp>.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=300, help="Maximum video length in environment steps.")
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
if args_cli.video or uses_vision_task or uses_vision_config:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import head_blood_absorption.tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from r2dreamer_isaac.config import build_runtime_config, load_yaml
from r2dreamer_isaac.env_adapter import IsaacR2DreamerEnvAdapter, obs_to_device
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


def _metric_to_float(value) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 0:
            return None
        if value.numel() == 1:
            return float(value.item())
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


EVAL_FIELDNAMES = (
    "algorithm",
    "task",
    "seed",
    "checkpoint",
    "episode_id",
    "return",
    "episode_length",
    "success",
    "severe_collision",
    "time_out",
    "absorbed_ratio_final",
    "ur3_contact_force_max",
    "tip_goal_error_mean",
    "tip_pipe_clearance_mean",
)


def _make_output_dir(run_dir: pathlib.Path, output_dir: str | None) -> pathlib.Path | None:
    if output_dir is None:
        return None
    path = pathlib.Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_output_dir(run_dir: pathlib.Path) -> pathlib.Path:
    root = run_dir / "eval"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _extract_metric(log_data: dict, key: str, default: float = float("nan")) -> float:
    value = _metric_to_float(log_data.get(key))
    return default if value is None else value


def _apply_env_cfg_overrides(env_cfg, overrides) -> None:
    if not overrides:
        return
    for name, value in overrides.items():
        if not hasattr(env_cfg, name):
            raise AttributeError(f"Environment config has no field '{name}' for env.cfg_overrides.")
        setattr(env_cfg, name, value)


def _termination_flags(log_data: dict) -> dict[str, bool]:
    success = _extract_metric(log_data, "Episode_Termination/success", 0.0) > 0.5
    severe_collision = _extract_metric(log_data, "Episode_Termination/severe_collision", 0.0) > 0.5
    time_out = _extract_metric(log_data, "Episode_Termination/time_out", 0.0) > 0.5
    return {
        "success": success,
        "severe_collision": severe_collision,
        "time_out": time_out,
    }


def _episode_row(
    *,
    task_name: str,
    checkpoint_path: pathlib.Path,
    episode_id: int,
    episode_return: float,
    episode_length: int,
    log_data: dict,
) -> dict[str, object]:
    flags = _termination_flags(log_data)
    return {
        "algorithm": "dreamer_v3",
        "task": task_name,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint_path),
        "episode_id": int(episode_id),
        "return": float(episode_return),
        "episode_length": int(episode_length),
        "success": flags["success"],
        "severe_collision": flags["severe_collision"],
        "time_out": flags["time_out"],
        "absorbed_ratio_final": _extract_metric(log_data, "Metrics/absorbed_ratio_mean"),
        "ur3_contact_force_max": _extract_metric(log_data, "Metrics/ur3_contact_force_max"),
        "tip_goal_error_mean": _extract_metric(log_data, "Metrics/tip_goal_error_mean"),
        "tip_pipe_clearance_mean": _extract_metric(log_data, "Metrics/tip_pipe_clearance_mean"),
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    finite = [float(value) for value in values if value == value]
    if not finite:
        return None
    tensor = torch.as_tensor(finite, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _write_eval_outputs(
    output_dir: pathlib.Path,
    rows: list[dict[str, object]],
    task_name: str,
    checkpoint_path: pathlib.Path,
    step_count: int,
) -> None:
    csv_path = output_dir / "episode_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    completed = len(rows)
    summary = {
        "algorithm": "dreamer_v3",
        "task": task_name,
        "seed": args_cli.seed,
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "requested_episodes": int(args_cli.episodes),
        "completed_episodes": completed,
        "executed_steps": int(step_count),
        "complete": args_cli.episodes <= 0 or completed >= args_cli.episodes,
        "success_rate": float(sum(bool(row["success"]) for row in rows) / completed) if completed else None,
        "severe_collision_rate": (
            float(sum(bool(row["severe_collision"]) for row in rows) / completed) if completed else None
        ),
        "time_out_rate": float(sum(bool(row["time_out"]) for row in rows) / completed) if completed else None,
        "aggregates": {
            key: _stats([float(row[key]) for row in rows])
            for key in (
                "return",
                "episode_length",
                "absorbed_ratio_final",
                "ur3_contact_force_max",
                "tip_goal_error_mean",
                "tip_pipe_clearance_mean",
            )
        },
    }
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(f"[INFO] Saved: {csv_path}")
    print(f"[INFO] Saved: {json_path}")


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
    }
    if args_cli.seed is not None:
        cli_updates["seed"] = args_cli.seed
    if args_cli.num_envs is not None:
        cli_updates.setdefault("env", {})["num_envs"] = args_cli.num_envs

    config = build_runtime_config(task_cfg=task_cfg, cli_updates=cli_updates, dotlist_overrides=overrides)
    run_dir = _run_dir_from_checkpoint(checkpoint_path)
    output_dir = _make_output_dir(run_dir, args_cli.output_dir)
    if output_dir is None and args_cli.episodes > 0:
        output_dir = _default_output_dir(run_dir)

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

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(run_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    adapter = IsaacR2DreamerEnvAdapter(env)
    agent = Dreamer(config.model, adapter.observation_space, adapter.action_space).to(config.agent_device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["agent_state_dict"]
        if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint
        else checkpoint
    )
    agent.load_state_dict(state_dict)
    agent.eval()

    current_obs, _ = adapter.reset()
    current_obs = obs_to_device(current_obs, config.agent_device)
    current_is_first = torch.ones(adapter.num_envs, dtype=torch.bool, device=config.agent_device)
    agent_state = {key: value.clone() for key, value in agent.get_initial_state(adapter.num_envs).items()}
    returns = torch.zeros(adapter.num_envs, dtype=torch.float32, device=config.agent_device)
    lengths = torch.zeros(adapter.num_envs, dtype=torch.int32, device=config.agent_device)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    termination_counts = {
        "success": 0.0,
        "joint_limit": 0.0,
        "severe_collision": 0.0,
        "time_out": 0.0,
    }
    episode_rows: list[dict[str, object]] = []
    step_count = 0

    while simulation_app.is_running():
        if args_cli.episodes > 0 and len(episode_returns) >= args_cli.episodes:
            break
        if args_cli.video and step_count >= args_cli.video_length:
            break

        with torch.inference_mode():
            action, agent_state = agent.act(
                adapter.build_agent_obs(current_obs, current_is_first),
                agent_state,
                eval=True,
            )
            step_out = adapter.step(action)

        returns += step_out.reward[:, 0].to(config.agent_device)
        lengths += 1
        current_obs = obs_to_device(step_out.obs, config.agent_device)
        current_is_first = step_out.done.to(config.agent_device)
        step_count += 1

        done_indices = torch.nonzero(step_out.done, as_tuple=False).squeeze(-1)
        done_count = int(done_indices.numel())
        if done_count <= 0:
            continue

        env_metrics = step_out.extras.get("log", {})
        for name in tuple(termination_counts):
            scalar = _metric_to_float(env_metrics.get(f"Episode_Termination/{name}"))
            if scalar is not None:
                termination_counts[name] += scalar * done_count

        for index in done_indices.tolist():
            if args_cli.episodes > 0 and len(episode_returns) >= args_cli.episodes:
                break
            episode_return = float(returns[index].item())
            episode_length = int(lengths[index].item())
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_rows.append(
                _episode_row(
                    task_name=task_name,
                    checkpoint_path=checkpoint_path,
                    episode_id=len(episode_returns),
                    episode_return=episode_return,
                    episode_length=episode_length,
                    log_data=env_metrics,
                )
            )
            print(
                f"[PLAY] episode={len(episode_returns):03d} "
                f"return={episode_return:.3f} length={episode_length}"
            )
            returns[index] = 0.0
            lengths[index] = 0

    completed_episodes = len(episode_returns)
    if completed_episodes > 0:
        mean_return = sum(episode_returns) / completed_episodes
        mean_length = sum(episode_lengths) / completed_episodes
        print(
            f"[SUMMARY] episodes={completed_episodes} "
            f"mean_return={mean_return:.3f} mean_length={mean_length:.2f}"
        )
        for name, count in termination_counts.items():
            print(f"[SUMMARY] termination/{name}={count / completed_episodes:.3f}")
        if output_dir is not None:
            _write_eval_outputs(output_dir, episode_rows, task_name, checkpoint_path, step_count)
    else:
        print("[SUMMARY] No episodes finished before playback stopped.")

    adapter.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
