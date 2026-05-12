from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a skrl checkpoint for a fixed number of episodes.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for episode statistics.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the skrl checkpoint.")
parser.add_argument("--episodes", type=int, default=20, help="Number of completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=0, help="Safety cap. 0 derives from episode length.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for episode_summary.csv and summary.json.")
parser.add_argument("--seed", type=int, default=None, help="Optional evaluation environment seed.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The skrl algorithm used for training.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video or "vision" in args_cli.task.lower():
    args_cli.enable_cameras = True
if args_cli.num_envs != 1:
    parser.error("This evaluator expects --num_envs 1 for precise per-episode statistics.")
if args_cli.episodes <= 0:
    parser.error("--episodes must be a positive integer.")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be greater than or equal to 0.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import skrl
import torch
import yaml
from packaging import version

if version.parse(skrl.__version__) < version.parse("1.3.0"):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.3.0")
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

import head_blood_absorption  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401


FIELDNAMES = (
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


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    array = _to_numpy(value)
    if array.size <= 0:
        return default
    return float(array.reshape(-1)[0])


def _metric(log_data: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    return _to_float(log_data.get(key), default=default)


def _bool_metric(log_data: Mapping[str, Any], key: str) -> bool:
    return _metric(log_data, key, default=0.0) > 0.5


def _load_agent_cfg() -> dict:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    saved_cfg = checkpoint_path.parent.parent / "params" / "agent.yaml"
    if saved_cfg.is_file():
        with saved_cfg.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if isinstance(cfg, dict):
            return cfg

    algorithm = args_cli.algorithm.lower()
    try:
        return load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        return load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")


def _make_output_dir() -> Path:
    if args_cli.output_dir is not None:
        output_dir = Path(args_cli.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    root = run_dir / "eval"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _row(episode_id: int, episode_return: float, episode_length: int, log_data: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    return {
        "algorithm": "skrl_ppo",
        "task": args_cli.task,
        "seed": args_cli.seed,
        "checkpoint": str(checkpoint_path),
        "episode_id": int(episode_id),
        "return": float(episode_return),
        "episode_length": int(episode_length),
        "success": _bool_metric(log_data, "Episode_Termination/success"),
        "severe_collision": _bool_metric(log_data, "Episode_Termination/severe_collision"),
        "time_out": _bool_metric(log_data, "Episode_Termination/time_out"),
        "absorbed_ratio_final": _metric(log_data, "Metrics/absorbed_ratio_mean"),
        "ur3_contact_force_max": _metric(log_data, "Metrics/ur3_contact_force_max"),
        "tip_goal_error_mean": _metric(log_data, "Metrics/tip_goal_error_mean"),
        "tip_pipe_clearance_mean": _metric(log_data, "Metrics/tip_pipe_clearance_mean"),
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size <= 0:
        return None
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], executed_steps: int) -> None:
    csv_path = output_dir / "episode_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    completed = len(rows)
    summary = {
        "algorithm": "skrl_ppo",
        "task": args_cli.task,
        "seed": args_cli.seed,
        "checkpoint_path": str(Path(args_cli.checkpoint).expanduser().resolve()),
        "output_dir": str(output_dir),
        "requested_episodes": int(args_cli.episodes),
        "completed_episodes": completed,
        "executed_steps": int(executed_steps),
        "complete": completed >= args_cli.episodes,
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
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    agent_cfg = _load_agent_cfg()
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)

    output_dir = _make_output_dir()
    print(f"[INFO] Evaluation results will be written to: {output_dir}")
    print(f"[INFO] Loading model checkpoint from: {args_cli.checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording video during evaluation.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm.lower() == "ppo":
            env = multi_agent_to_single_agent(env)

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        raw_env = env.unwrapped
        runner = Runner(env, agent_cfg)
        runner.agent.load(os.path.abspath(args_cli.checkpoint))
        runner.agent.set_running_mode("eval")

        obs, _ = env.reset()
        episode_limit = int(getattr(raw_env, "max_episode_length", 0))
        max_steps = int(args_cli.max_steps)
        if max_steps == 0:
            if episode_limit <= 0:
                raise RuntimeError("Unable to infer --max_steps from the environment.")
            max_steps = episode_limit * (args_cli.episodes + 2)

        rows: list[dict[str, Any]] = []
        current_return = 0.0
        current_length = 0
        total_steps = 0

        with torch.inference_mode():
            while simulation_app.is_running() and total_steps < max_steps and len(rows) < args_cli.episodes:
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
                obs, reward, terminated, truncated, infos = env.step(actions)
                log_data = infos.get("log", {}) if isinstance(infos, Mapping) else {}
                if not log_data and hasattr(raw_env, "extras"):
                    raw_log = getattr(raw_env, "extras", {}).get("log", {})
                    if isinstance(raw_log, Mapping):
                        log_data = raw_log

                reward_value = _to_float(reward, default=0.0)
                done = bool(_to_float(terminated, default=0.0) or _to_float(truncated, default=0.0))
                current_return += reward_value
                current_length += 1
                total_steps += 1

                if done:
                    episode_id = len(rows) + 1
                    row = _row(episode_id, current_return, current_length, log_data)
                    rows.append(row)
                    print(
                        "[INFO] Episode completed: "
                        f"{episode_id}/{args_cli.episodes}, return={current_return:.4f}, "
                        f"length={current_length}, success={row['success']}, "
                        f"severe_collision={row['severe_collision']}"
                    )
                    current_return = 0.0
                    current_length = 0

        _write_outputs(output_dir, rows, total_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
