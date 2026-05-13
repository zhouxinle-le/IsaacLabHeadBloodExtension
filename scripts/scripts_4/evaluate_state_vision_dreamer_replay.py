"""Replay-evaluate State Dreamer and Vision Dreamer checkpoints.

This script mirrors the PPO replay evaluator in scripts_3, but loads policies
through the R2-Dreamer integration. The scheduler process launches one worker
process per run so each Isaac environment is created in a fresh process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omni.isaac.lab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

STATE_TASK = "Isaac-Ur3-Blood-Pipe-State-Direct-v0"
VISION_TASK = "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0"
RESULTS_ROOT = Path("scripts/scripts_4/results")

DEFAULT_STATE_RUN0_CHECKPOINT = Path(
    "logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_0_800k/checkpoints/policy_step_550004.pt"
)
DEFAULT_STATE_RUN1_CHECKPOINT = Path(
    "logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_1_800k/checkpoints/policy_step_550004.pt"
)
DEFAULT_VISION_RUN0_CHECKPOINT = Path(
    "logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_0_800k/checkpoints/policy_step_450001.pt"
)
DEFAULT_VISION_RUN1_CHECKPOINT = Path(
    "logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/"
    "2026-05-12_21-14-29_seed_0_600k/checkpoints/policy_step_550001.pt"
)

EPISODE_FIELDNAMES = (
    "method",
    "run_index",
    "run_name",
    "algorithm",
    "task",
    "seed",
    "checkpoint",
    "config",
    "episode_id",
    "return",
    "episode_steps",
    "completion_time_s",
    "success",
    "severe_collision",
    "time_out",
    "termination_reason",
    "absorbed_ratio_final",
    "absorbed_count_final",
    "tip_contact_force_mean_n",
    "tip_contact_force_max_n",
    "tip_contact_force_final_n",
    "ur3_contact_force_max_log",
    "tip_goal_error_mean_final",
    "tip_pipe_clearance_mean_final",
)

SUMMARY_FIELDNAMES = (
    "method",
    "run_index",
    "run_name",
    "algorithm",
    "task",
    "seed",
    "checkpoint",
    "config",
    "requested_episodes",
    "completed_episodes",
    "success_count",
    "success_rate",
    "severe_collision_count",
    "severe_collision_rate",
    "time_out_count",
    "time_out_rate",
    "return_mean",
    "return_std",
    "episode_steps_mean",
    "episode_steps_std",
    "completion_time_s_mean",
    "completion_time_s_std",
    "success_completion_time_s_mean",
    "success_completion_time_s_std",
    "tip_contact_force_mean_n_mean",
    "tip_contact_force_mean_n_std",
    "tip_contact_force_max_n_mean",
    "tip_contact_force_max_n_std",
    "success_tip_contact_force_mean_n_mean",
    "success_tip_contact_force_mean_n_std",
    "success_tip_contact_force_max_n_mean",
    "success_tip_contact_force_max_n_std",
    "saved_success_trajectories",
    "saved_failure_trajectories",
    "camera_frames_saved",
    "contact_force_source",
    "env_step_dt",
    "executed_steps",
    "complete",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    run_index: int
    run_name: str
    algorithm: str
    task: str
    seed: int
    checkpoint: Path
    config: Path


@dataclass
class TraceStep:
    step: int
    time_s: float
    tip_pos_w: tuple[float, float, float]
    tip_contact_force_n: float
    reward: float
    done: bool
    action: list[float]


@dataclass
class EpisodeTrace:
    steps: list[TraceStep]
    camera_frames: list[tuple[int, Any]]


@dataclass
class RunResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    trajectory_paths: list[str]
    camera_frame_dirs: list[str]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay-evaluate State Dreamer and Vision Dreamer checkpoints.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("state_dreamer", "vision_dreamer"),
        default=["state_dreamer", "vision_dreamer"],
        help="Dreamer methods to evaluate.",
    )
    parser.add_argument(
        "--run-indices",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=[0, 1],
        help="Run indices to evaluate for each selected method.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100, help="Completed episodes to evaluate per run.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments. Must be 1.")
    parser.add_argument("--max-steps", type=int, default=0, help="Safety cap per run. 0 derives from episode length.")
    parser.add_argument("--disable-fabric", action="store_true", default=True, help="Disable Fabric.")
    parser.add_argument(
        "--use-fabric",
        action="store_false",
        dest="disable_fabric",
        help="Use Fabric instead of the paper-experiment default.",
    )
    parser.add_argument("--env-device", type=str, default="cpu", help="Device for the Isaac environment.")
    parser.add_argument("--agent-device", type=str, default="cuda:0", help="Device for the Dreamer policy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent output directory. A timestamped result directory is created inside it.",
    )
    parser.add_argument(
        "--success-trajectories-per-run",
        type=int,
        default=2,
        help="Number of successful tip trajectories to save for each run.",
    )
    parser.add_argument(
        "--failure-trajectories-per-run",
        type=int,
        default=2,
        help="Number of failed tip trajectories to save for each run.",
    )
    parser.add_argument(
        "--no-record-success-trajectories",
        action="store_true",
        default=False,
        help="Disable successful tip trajectory CSV output.",
    )
    parser.add_argument(
        "--no-record-failure-trajectories",
        action="store_true",
        default=False,
        help="Disable failed tip trajectory CSV output.",
    )
    parser.add_argument(
        "--no-record-camera-frames",
        action="store_true",
        default=False,
        help="Disable camera frame PNG output for saved Vision Dreamer trajectories.",
    )
    parser.add_argument(
        "--camera-frame-stride",
        type=int,
        default=1,
        help="Save one Vision camera frame every N environment steps for saved trajectories.",
    )
    parser.add_argument("--state-run0-checkpoint", type=Path, default=DEFAULT_STATE_RUN0_CHECKPOINT)
    parser.add_argument("--state-run1-checkpoint", type=Path, default=DEFAULT_STATE_RUN1_CHECKPOINT)
    parser.add_argument("--vision-run0-checkpoint", type=Path, default=DEFAULT_VISION_RUN0_CHECKPOINT)
    parser.add_argument("--vision-run1-checkpoint", type=Path, default=DEFAULT_VISION_RUN1_CHECKPOINT)
    parser.add_argument("--worker-run-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-artifact-dir", type=Path, default=None, help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", enable_cameras=False)
    return parser


parser = _build_arg_parser()
args_cli, overrides = parser.parse_known_args()

if args_cli.num_envs != 1:
    parser.error("This evaluator only supports --num-envs 1 for precise per-episode statistics.")
if args_cli.eval_episodes <= 0:
    parser.error("--eval-episodes must be positive.")
if args_cli.max_steps < 0:
    parser.error("--max-steps must be greater than or equal to 0.")
if args_cli.success_trajectories_per_run < 0:
    parser.error("--success-trajectories-per-run must be greater than or equal to 0.")
if args_cli.failure_trajectories_per_run < 0:
    parser.error("--failure-trajectories-per-run must be greater than or equal to 0.")
if args_cli.camera_frame_stride <= 0:
    parser.error("--camera-frame-stride must be positive.")

args_cli.methods = list(dict.fromkeys(args_cli.methods))
args_cli.run_indices = list(dict.fromkeys(args_cli.run_indices))

simulation_app = None
_runtime_loaded = False


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _run_dir_from_checkpoint(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _config_path_from_checkpoint(checkpoint: Path) -> Path:
    return _run_dir_from_checkpoint(checkpoint) / "params" / "r2dreamer.yaml"


def _default_specs() -> dict[tuple[str, int], RunSpec]:
    state0 = _resolve_path(args_cli.state_run0_checkpoint)
    state1 = _resolve_path(args_cli.state_run1_checkpoint)
    vision0 = _resolve_path(args_cli.vision_run0_checkpoint)
    vision1 = _resolve_path(args_cli.vision_run1_checkpoint)
    return {
        ("state_dreamer", 0): RunSpec(
            method="state_dreamer",
            run_index=0,
            run_name="state_dreamer_run0",
            algorithm="dreamer_v3",
            task=STATE_TASK,
            seed=0,
            checkpoint=state0,
            config=_config_path_from_checkpoint(state0),
        ),
        ("state_dreamer", 1): RunSpec(
            method="state_dreamer",
            run_index=1,
            run_name="state_dreamer_run1",
            algorithm="dreamer_v3",
            task=STATE_TASK,
            seed=1,
            checkpoint=state1,
            config=_config_path_from_checkpoint(state1),
        ),
        ("vision_dreamer", 0): RunSpec(
            method="vision_dreamer",
            run_index=0,
            run_name="vision_dreamer_run0",
            algorithm="dreamer_v3",
            task=VISION_TASK,
            seed=0,
            checkpoint=vision0,
            config=_config_path_from_checkpoint(vision0),
        ),
        ("vision_dreamer", 1): RunSpec(
            method="vision_dreamer",
            run_index=1,
            run_name="vision_dreamer_run1",
            algorithm="dreamer_v3",
            task=VISION_TASK,
            seed=0,
            checkpoint=vision1,
            config=_config_path_from_checkpoint(vision1),
        ),
    }


def _build_run_specs() -> list[RunSpec]:
    defaults = _default_specs()
    specs: list[RunSpec] = []
    for method in args_cli.methods:
        for run_index in args_cli.run_indices:
            spec = defaults.get((method, run_index))
            if spec is not None:
                specs.append(spec)
    return specs


def _selected_worker_spec() -> RunSpec | None:
    if args_cli.worker_run_index is None:
        return None
    specs = _build_run_specs()
    if 0 <= args_cli.worker_run_index < len(specs):
        return specs[args_cli.worker_run_index]
    return None


selected_worker_spec = _selected_worker_spec()
if selected_worker_spec is not None and selected_worker_spec.method == "vision_dreamer":
    args_cli.enable_cameras = True


def _load_isaac_runtime() -> None:
    global Dreamer
    global IsaacR2DreamerEnvAdapter
    global build_runtime_config
    global gym
    global load_yaml
    global np
    global obs_to_device
    global parse_env_cfg
    global simulation_app
    global torch
    global _runtime_loaded

    if _runtime_loaded:
        return

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym_module
    import numpy as np_module
    import torch as torch_module

    import head_blood_absorption.tasks  # noqa: F401
    import omni.isaac.lab_tasks  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg as parse_env_cfg_func
    from r2dreamer_isaac.config import build_runtime_config as build_runtime_config_func
    from r2dreamer_isaac.config import load_yaml as load_yaml_func
    from r2dreamer_isaac.env_adapter import IsaacR2DreamerEnvAdapter as DreamerEnvAdapter
    from r2dreamer_isaac.env_adapter import obs_to_device as obs_to_device_func
    from r2dreamer_isaac.vendor.r2dreamer import Dreamer as DreamerClass

    gym = gym_module
    np = np_module
    torch = torch_module
    parse_env_cfg = parse_env_cfg_func
    build_runtime_config = build_runtime_config_func
    load_yaml = load_yaml_func
    IsaacR2DreamerEnvAdapter = DreamerEnvAdapter
    obs_to_device = obs_to_device_func
    Dreamer = DreamerClass
    _runtime_loaded = True


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


def _bool_metric(log_data: Mapping[str, Any], key: str) -> bool:
    return _to_float(log_data.get(key), default=0.0) > 0.5


def _metric(log_data: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    return _to_float(log_data.get(key), default=default)


def _action_vector(actions: Any) -> list[float]:
    array = np.asarray(_to_numpy(actions), dtype=np.float32).reshape(args_cli.num_envs, -1)
    return [float(value) for value in array[0]]


def _env_step_dt(raw_env: Any) -> float:
    cfg = getattr(raw_env, "cfg", None)
    sim_cfg = getattr(cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(cfg, "decimation", None)
    if sim_dt is not None and decimation is not None:
        return float(sim_dt) * float(decimation)
    step_dt = getattr(raw_env, "step_dt", None)
    if step_dt is not None:
        return float(step_dt)
    return float("nan")


def _tip_position(raw_env: Any) -> tuple[float, float, float]:
    tip_pos_w, _ = raw_env._compute_tip_pose_and_direction_w()
    array = np.asarray(_to_numpy(tip_pos_w), dtype=np.float32).reshape(raw_env.num_envs, -1)[0]
    return float(array[0]), float(array[1]), float(array[2])


def _contact_force_source(raw_env: Any) -> str:
    cfg = getattr(raw_env, "cfg", None)
    tip_body_name = str(getattr(cfg, "ur3_tip_body_name", "tip_link"))
    sensors = getattr(raw_env, "_ur3_contact_sensors", {})
    if isinstance(sensors, Mapping) and tip_body_name in sensors:
        return f"tip_contact_sensor:{tip_body_name}"
    if hasattr(raw_env, "_get_ur3_contact_force"):
        return "fallback:_get_ur3_contact_force"
    return "unavailable"


def _tip_contact_force(raw_env: Any) -> float:
    cfg = getattr(raw_env, "cfg", None)
    tip_body_name = str(getattr(cfg, "ur3_tip_body_name", "tip_link"))
    sensors = getattr(raw_env, "_ur3_contact_sensors", {})
    if isinstance(sensors, Mapping) and tip_body_name in sensors:
        net_forces_w = sensors[tip_body_name].data.net_forces_w
        contact_force = torch.linalg.vector_norm(net_forces_w, dim=-1)
        if contact_force.ndim > 1:
            contact_force = torch.amax(contact_force, dim=1)
        return _to_float(contact_force)
    if hasattr(raw_env, "_get_ur3_contact_force"):
        return _to_float(raw_env._get_ur3_contact_force())
    return float("nan")


def _camera_frame(raw_env: Any) -> np.ndarray | None:
    camera = getattr(raw_env, "_camera", None)
    if camera is None:
        return None
    output = getattr(getattr(camera, "data", None), "output", None)
    if not isinstance(output, Mapping):
        return None
    rgb = output.get("rgb")
    if rgb is None:
        return None
    array = np.asarray(_to_numpy(rgb))
    if array.size == 0:
        return None
    array = array[0, ..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _save_png(path: Path, frame: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(frame).save(path)
        return
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, frame)
        return
    except Exception:
        pass

    import matplotlib.image as mpimg

    mpimg.imsave(path, frame)


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size <= 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _termination_reason(log_data: Mapping[str, Any]) -> str:
    if _bool_metric(log_data, "Episode_Termination/success"):
        return "success"
    if _bool_metric(log_data, "Episode_Termination/severe_collision"):
        return "severe_collision"
    if _bool_metric(log_data, "Episode_Termination/time_out"):
        return "time_out"
    if _bool_metric(log_data, "Episode_Termination/joint_limit"):
        return "joint_limit"
    return "unknown"


def _new_trace() -> EpisodeTrace:
    return EpisodeTrace(steps=[], camera_frames=[])


def _append_trace_step(
    trace: EpisodeTrace,
    raw_env: Any,
    env_step_dt: float,
    local_step: int,
    reward: float,
    done: bool,
    actions: Any,
    record_camera: bool,
) -> None:
    trace.steps.append(
        TraceStep(
            step=int(local_step),
            time_s=float(local_step * env_step_dt),
            tip_pos_w=_tip_position(raw_env),
            tip_contact_force_n=_tip_contact_force(raw_env),
            reward=float(reward),
            done=bool(done),
            action=_action_vector(actions),
        )
    )
    if record_camera and local_step % int(args_cli.camera_frame_stride) == 0:
        frame = _camera_frame(raw_env)
        if frame is not None:
            trace.camera_frames.append((int(local_step), frame))


def _episode_row(
    spec: RunSpec,
    episode_id: int,
    episode_return: float,
    trace: EpisodeTrace,
    log_data: Mapping[str, Any],
    env_step_dt: float,
) -> dict[str, Any]:
    forces = [step.tip_contact_force_n for step in trace.steps]
    reason = _termination_reason(log_data)
    episode_steps = len(trace.steps)
    return {
        "method": spec.method,
        "run_index": int(spec.run_index),
        "run_name": spec.run_name,
        "algorithm": spec.algorithm,
        "task": spec.task,
        "seed": int(spec.seed),
        "checkpoint": str(spec.checkpoint),
        "config": str(spec.config),
        "episode_id": int(episode_id),
        "return": float(episode_return),
        "episode_steps": int(episode_steps),
        "completion_time_s": float(episode_steps * env_step_dt),
        "success": reason == "success",
        "severe_collision": reason == "severe_collision",
        "time_out": reason == "time_out",
        "termination_reason": reason,
        "absorbed_ratio_final": _metric(log_data, "Metrics/absorbed_ratio_mean"),
        "absorbed_count_final": _metric(log_data, "Metrics/absorbed_count"),
        "tip_contact_force_mean_n": float(np.nanmean(forces)) if forces else float("nan"),
        "tip_contact_force_max_n": float(np.nanmax(forces)) if forces else float("nan"),
        "tip_contact_force_final_n": float(forces[-1]) if forces else float("nan"),
        "ur3_contact_force_max_log": _metric(log_data, "Metrics/ur3_contact_force_max"),
        "tip_goal_error_mean_final": _metric(log_data, "Metrics/tip_goal_error_mean"),
        "tip_pipe_clearance_mean_final": _metric(log_data, "Metrics/tip_pipe_clearance_mean"),
    }


def _write_trajectory_csv(path: Path, trace: EpisodeTrace) -> None:
    max_action_dim = max((len(step.action) for step in trace.steps), default=0)
    fieldnames = [
        "step",
        "time_s",
        "tip_pos_w_x",
        "tip_pos_w_y",
        "tip_pos_w_z",
        "tip_contact_force_n",
        "reward",
        "done",
    ]
    fieldnames.extend(f"action_{index}" for index in range(max_action_dim))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in trace.steps:
            row = {
                "step": step.step,
                "time_s": step.time_s,
                "tip_pos_w_x": step.tip_pos_w[0],
                "tip_pos_w_y": step.tip_pos_w[1],
                "tip_pos_w_z": step.tip_pos_w[2],
                "tip_contact_force_n": step.tip_contact_force_n,
                "reward": step.reward,
                "done": step.done,
            }
            for index, value in enumerate(step.action):
                row[f"action_{index}"] = value
            writer.writerow(row)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _artifact_metadata(
    spec: RunSpec,
    episode_id: int,
    outcome: str,
    outcome_index: int,
    episode_row: Mapping[str, Any],
    trace: EpisodeTrace,
    camera_frames_saved: int,
) -> dict[str, Any]:
    failure_reason = str(episode_row.get("termination_reason", "")) if outcome == "failure" else None
    return {
        "method": spec.method,
        "run_index": int(spec.run_index),
        "run_name": spec.run_name,
        "algorithm": spec.algorithm,
        "task": spec.task,
        "seed": int(spec.seed),
        "checkpoint": str(spec.checkpoint),
        "config": str(spec.config),
        "episode_id": int(episode_id),
        "outcome": outcome,
        "outcome_index": int(outcome_index),
        "failure_reason": failure_reason,
        "return": _json_ready(episode_row.get("return")),
        "episode_steps": _json_ready(episode_row.get("episode_steps")),
        "completion_time_s": _json_ready(episode_row.get("completion_time_s")),
        "termination_reason": _json_ready(episode_row.get("termination_reason")),
        "success": _json_ready(episode_row.get("success")),
        "severe_collision": _json_ready(episode_row.get("severe_collision")),
        "time_out": _json_ready(episode_row.get("time_out")),
        "absorbed_ratio_final": _json_ready(episode_row.get("absorbed_ratio_final")),
        "absorbed_count_final": _json_ready(episode_row.get("absorbed_count_final")),
        "tip_contact_force_mean_n": _json_ready(episode_row.get("tip_contact_force_mean_n")),
        "tip_contact_force_max_n": _json_ready(episode_row.get("tip_contact_force_max_n")),
        "tip_contact_force_final_n": _json_ready(episode_row.get("tip_contact_force_final_n")),
        "tip_goal_error_mean_final": _json_ready(episode_row.get("tip_goal_error_mean_final")),
        "tip_pipe_clearance_mean_final": _json_ready(episode_row.get("tip_pipe_clearance_mean_final")),
        "trajectory_points": len(trace.steps),
        "camera_frame_stride": int(args_cli.camera_frame_stride),
        "camera_frames_saved": int(camera_frames_saved),
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _save_episode_artifacts(
    output_dir: Path,
    spec: RunSpec,
    episode_id: int,
    outcome: str,
    outcome_index: int,
    episode_row: Mapping[str, Any],
    trace: EpisodeTrace,
    save_trajectory: bool,
    save_camera: bool,
) -> tuple[str | None, str | None, int]:
    stem = (
        f"{spec.method}_run{spec.run_index}_seed{spec.seed}_"
        f"{outcome}{outcome_index:02d}_episode{episode_id:03d}"
    )
    trajectory_path = None
    trajectory_metadata_path = None
    camera_dir = None
    camera_frames_saved = 0

    if save_trajectory:
        trajectory_dir = output_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = trajectory_dir / f"{stem}_tip_trajectory.csv"
        _write_trajectory_csv(path, trace)
        trajectory_metadata_path = trajectory_dir / f"{stem}_metadata.json"
        trajectory_path = str(path)

    if save_camera and trace.camera_frames:
        frames_dir = output_dir / "camera_frames" / stem
        frames_dir.mkdir(parents=True, exist_ok=True)
        for step_index, frame in trace.camera_frames:
            frame_path = frames_dir / f"frame_{step_index:06d}.png"
            _save_png(frame_path, frame)
            camera_frames_saved += 1
        metadata = _artifact_metadata(
            spec=spec,
            episode_id=episode_id,
            outcome=outcome,
            outcome_index=outcome_index,
            episode_row=episode_row,
            trace=trace,
            camera_frames_saved=camera_frames_saved,
        )
        metadata["frames_saved"] = camera_frames_saved
        _write_json(frames_dir / "metadata.json", metadata)
        camera_dir = str(frames_dir)

    if trajectory_metadata_path is not None:
        _write_json(
            trajectory_metadata_path,
            _artifact_metadata(
                spec=spec,
                episode_id=episode_id,
                outcome=outcome,
                outcome_index=outcome_index,
                episode_row=episode_row,
                trace=trace,
                camera_frames_saved=camera_frames_saved,
            ),
        )

    return trajectory_path, camera_dir, camera_frames_saved


def _make_output_dir(methods: Sequence[str]) -> Path:
    parent = args_cli.output_dir.expanduser() if args_cli.output_dir is not None else RESULTS_ROOT
    parent = _resolve_path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if set(methods) == {"state_dreamer", "vision_dreamer"}:
        method_label = "state_vision_dreamer"
    else:
        method_label = "_".join(methods)
    candidate = parent / f"{timestamp}_{method_label}_replay"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{timestamp}_{method_label}_replay_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _evaluate_dreamer_run(spec: RunSpec, output_dir: Path) -> RunResult:
    _load_isaac_runtime()

    task_cfg = load_yaml(spec.config)
    cli_updates: dict[str, Any] = {
        "device": args_cli.agent_device,
        "agent_device": args_cli.agent_device,
        "env_device": args_cli.env_device,
        "seed": int(spec.seed),
        "env": {"num_envs": int(args_cli.num_envs)},
    }
    config = build_runtime_config(task_cfg=task_cfg, cli_updates=cli_updates, dotlist_overrides=overrides)

    env_cfg = parse_env_cfg(
        spec.task,
        device=str(config.env_device),
        num_envs=int(args_cli.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = int(spec.seed)

    print(f"[INFO] Evaluating {spec.run_name}: {spec.checkpoint}")
    env = gym.make(spec.task, cfg=env_cfg, render_mode=None)
    adapter = IsaacR2DreamerEnvAdapter(env)
    try:
        raw_env = adapter.unwrapped
        contact_source = _contact_force_source(raw_env)
        env_step_dt = _env_step_dt(raw_env)

        agent = Dreamer(config.model, adapter.observation_space, adapter.action_space).to(config.agent_device)
        checkpoint = torch.load(spec.checkpoint, map_location="cpu")
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

        return _rollout_dreamer(
            spec=spec,
            output_dir=output_dir,
            adapter=adapter,
            raw_env=raw_env,
            env_step_dt=env_step_dt,
            contact_source=contact_source,
            agent=agent,
            agent_state=agent_state,
            current_obs=current_obs,
            current_is_first=current_is_first,
            agent_device=config.agent_device,
            record_camera=(spec.method == "vision_dreamer" and not args_cli.no_record_camera_frames),
        )
    finally:
        adapter.close()


def _rollout_dreamer(
    spec: RunSpec,
    output_dir: Path,
    adapter: Any,
    raw_env: Any,
    env_step_dt: float,
    contact_source: str,
    agent: Any,
    agent_state: Mapping[str, Any],
    current_obs: Any,
    current_is_first: Any,
    agent_device: str,
    record_camera: bool,
) -> RunResult:
    episode_limit = int(getattr(raw_env, "max_episode_length", 0))
    max_steps = int(args_cli.max_steps)
    if max_steps == 0:
        if episode_limit <= 0:
            raise RuntimeError("Unable to infer max steps because the environment has no max_episode_length.")
        max_steps = episode_limit * (args_cli.eval_episodes + 2)

    rows: list[dict[str, Any]] = []
    trajectory_paths: list[str] = []
    camera_frame_dirs: list[str] = []
    saved_success_count = 0
    saved_failure_count = 0
    camera_frames_saved = 0
    current_trace = _new_trace()
    current_return = 0.0
    current_local_step = 0
    total_steps = 0

    with torch.inference_mode():
        while simulation_app.is_running() and total_steps < max_steps and len(rows) < args_cli.eval_episodes:
            action, agent_state = agent.act(
                adapter.build_agent_obs(current_obs, current_is_first),
                agent_state,
                eval=True,
            )
            step_out = adapter.step(action)
            reward = _to_float(step_out.reward, default=0.0)
            done = bool(_to_float(step_out.done, default=0.0))
            log_data = step_out.extras.get("log", {}) if isinstance(step_out.extras, Mapping) else {}

            should_buffer_success_artifacts = (
                not args_cli.no_record_success_trajectories
                and saved_success_count < args_cli.success_trajectories_per_run
            )
            should_buffer_failure_artifacts = (
                not args_cli.no_record_failure_trajectories
                and saved_failure_count < args_cli.failure_trajectories_per_run
            )
            should_buffer_artifacts = should_buffer_success_artifacts or should_buffer_failure_artifacts
            _append_trace_step(
                trace=current_trace,
                raw_env=raw_env,
                env_step_dt=env_step_dt,
                local_step=current_local_step,
                reward=reward,
                done=done,
                actions=action,
                record_camera=record_camera and should_buffer_artifacts,
            )

            current_obs = obs_to_device(step_out.obs, agent_device)
            current_is_first = step_out.done.to(agent_device)
            current_return += reward
            current_local_step += 1
            total_steps += 1

            if done:
                episode_id = len(rows) + 1
                row = _episode_row(
                    spec=spec,
                    episode_id=episode_id,
                    episode_return=current_return,
                    trace=current_trace,
                    log_data=log_data,
                    env_step_dt=env_step_dt,
                )
                rows.append(row)

                if bool(row["success"]) and should_buffer_success_artifacts:
                    saved_success_count += 1
                    trajectory_path, camera_dir, frame_count = _save_episode_artifacts(
                        output_dir=output_dir,
                        spec=spec,
                        episode_id=episode_id,
                        outcome="success",
                        outcome_index=saved_success_count,
                        episode_row=row,
                        trace=current_trace,
                        save_trajectory=True,
                        save_camera=record_camera,
                    )
                    if trajectory_path is not None:
                        trajectory_paths.append(trajectory_path)
                    if camera_dir is not None:
                        camera_frame_dirs.append(camera_dir)
                    camera_frames_saved += frame_count
                elif (not bool(row["success"])) and should_buffer_failure_artifacts:
                    saved_failure_count += 1
                    trajectory_path, camera_dir, frame_count = _save_episode_artifacts(
                        output_dir=output_dir,
                        spec=spec,
                        episode_id=episode_id,
                        outcome="failure",
                        outcome_index=saved_failure_count,
                        episode_row=row,
                        trace=current_trace,
                        save_trajectory=True,
                        save_camera=record_camera,
                    )
                    if trajectory_path is not None:
                        trajectory_paths.append(trajectory_path)
                    if camera_dir is not None:
                        camera_frame_dirs.append(camera_dir)
                    camera_frames_saved += frame_count

                print(
                    "[INFO] Episode completed: "
                    f"{spec.run_name} {episode_id}/{args_cli.eval_episodes}, "
                    f"steps={row['episode_steps']}, return={row['return']:.4f}, "
                    f"success={row['success']}, reason={row['termination_reason']}"
                )

                current_trace = _new_trace()
                current_return = 0.0
                current_local_step = 0

    summary = _run_summary(
        spec=spec,
        rows=rows,
        requested_episodes=args_cli.eval_episodes,
        total_steps=total_steps,
        max_steps=max_steps,
        env_step_dt=env_step_dt,
        contact_source=contact_source,
        saved_success_trajectories=saved_success_count,
        saved_failure_trajectories=saved_failure_count,
        camera_frames_saved=camera_frames_saved,
        trajectory_paths=trajectory_paths,
        camera_frame_dirs=camera_frame_dirs,
    )
    return RunResult(
        rows=rows,
        summary=summary,
        trajectory_paths=trajectory_paths,
        camera_frame_dirs=camera_frame_dirs,
    )


def _run_summary(
    spec: RunSpec,
    rows: list[dict[str, Any]],
    requested_episodes: int,
    total_steps: int,
    max_steps: int,
    env_step_dt: float,
    contact_source: str,
    saved_success_trajectories: int,
    saved_failure_trajectories: int,
    camera_frames_saved: int,
    trajectory_paths: list[str],
    camera_frame_dirs: list[str],
) -> dict[str, Any]:
    completed = len(rows)
    success_rows = [row for row in rows if bool(row["success"])]
    severe_rows = [row for row in rows if bool(row["severe_collision"])]
    timeout_rows = [row for row in rows if bool(row["time_out"])]

    def column(name: str, source_rows: list[dict[str, Any]] = rows) -> list[float]:
        return [float(row[name]) for row in source_rows]

    return {
        "method": spec.method,
        "run_index": int(spec.run_index),
        "run_name": spec.run_name,
        "algorithm": spec.algorithm,
        "task": spec.task,
        "seed": int(spec.seed),
        "checkpoint": str(spec.checkpoint),
        "config": str(spec.config),
        "requested_episodes": int(requested_episodes),
        "completed_episodes": int(completed),
        "success_count": int(len(success_rows)),
        "success_rate": float(len(success_rows) / completed) if completed else None,
        "severe_collision_count": int(len(severe_rows)),
        "severe_collision_rate": float(len(severe_rows) / completed) if completed else None,
        "time_out_count": int(len(timeout_rows)),
        "time_out_rate": float(len(timeout_rows) / completed) if completed else None,
        "return": _stats(column("return")),
        "episode_steps": _stats(column("episode_steps")),
        "completion_time_s": _stats(column("completion_time_s")),
        "success_completion_time_s": _stats(column("completion_time_s", success_rows)),
        "tip_contact_force_mean_n": _stats(column("tip_contact_force_mean_n")),
        "tip_contact_force_max_n": _stats(column("tip_contact_force_max_n")),
        "success_tip_contact_force_mean_n": _stats(column("tip_contact_force_mean_n", success_rows)),
        "success_tip_contact_force_max_n": _stats(column("tip_contact_force_max_n", success_rows)),
        "saved_success_trajectories": int(saved_success_trajectories),
        "saved_failure_trajectories": int(saved_failure_trajectories),
        "camera_frames_saved": int(camera_frames_saved),
        "trajectory_paths": trajectory_paths,
        "camera_frame_dirs": camera_frame_dirs,
        "contact_force_source": contact_source,
        "env_step_dt": float(env_step_dt),
        "executed_steps": int(total_steps),
        "max_steps": int(max_steps),
        "complete": completed >= requested_episodes,
    }


def _flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    flat = {
        "method": summary["method"],
        "run_index": summary["run_index"],
        "run_name": summary["run_name"],
        "algorithm": summary["algorithm"],
        "task": summary["task"],
        "seed": summary["seed"],
        "checkpoint": summary["checkpoint"],
        "config": summary["config"],
        "requested_episodes": summary["requested_episodes"],
        "completed_episodes": summary["completed_episodes"],
        "success_count": summary["success_count"],
        "success_rate": summary["success_rate"],
        "severe_collision_count": summary["severe_collision_count"],
        "severe_collision_rate": summary["severe_collision_rate"],
        "time_out_count": summary["time_out_count"],
        "time_out_rate": summary["time_out_rate"],
        "saved_success_trajectories": summary["saved_success_trajectories"],
        "saved_failure_trajectories": summary["saved_failure_trajectories"],
        "camera_frames_saved": summary["camera_frames_saved"],
        "contact_force_source": summary["contact_force_source"],
        "env_step_dt": summary["env_step_dt"],
        "executed_steps": summary["executed_steps"],
        "complete": summary["complete"],
    }
    for key in (
        "return",
        "episode_steps",
        "completion_time_s",
        "success_completion_time_s",
        "tip_contact_force_mean_n",
        "tip_contact_force_max_n",
        "success_tip_contact_force_mean_n",
        "success_tip_contact_force_max_n",
    ):
        stats = summary[key]
        flat[f"{key}_mean"] = stats["mean"]
        flat[f"{key}_std"] = stats["std"]
    return flat


def _write_episode_summary(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_path = output_dir / "episode_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_summary_by_run(output_dir: Path, summaries: list[dict[str, Any]]) -> Path:
    output_path = output_dir / "summary_by_run.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_flatten_summary(summary))
    return output_path


def _write_summary_json(
    output_dir: Path,
    specs: list[RunSpec],
    results: list[RunResult],
    episode_csv: Path,
    summary_csv: Path,
) -> Path:
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "episode_summary_csv": str(episode_csv),
        "summary_by_run_csv": str(summary_csv),
        "requested_methods": args_cli.methods,
        "requested_run_indices": args_cli.run_indices,
        "eval_episodes": int(args_cli.eval_episodes),
        "num_envs": int(args_cli.num_envs),
        "success_trajectories_per_run": int(args_cli.success_trajectories_per_run),
        "failure_trajectories_per_run": int(args_cli.failure_trajectories_per_run),
        "record_success_trajectories": not bool(args_cli.no_record_success_trajectories),
        "record_failure_trajectories": not bool(args_cli.no_record_failure_trajectories),
        "record_camera_frames": not bool(args_cli.no_record_camera_frames),
        "camera_frame_stride": int(args_cli.camera_frame_stride),
        "runs": [
            {
                "method": spec.method,
                "run_index": int(spec.run_index),
                "run_name": spec.run_name,
                "algorithm": spec.algorithm,
                "task": spec.task,
                "seed": int(spec.seed),
                "checkpoint": str(spec.checkpoint),
                "config": str(spec.config),
            }
            for spec in specs
        ],
        "summaries": [result.summary for result in results],
    }
    output_path = output_dir / "summary.json"
    _write_json(output_path, summary)
    return output_path


def _run_worker() -> None:
    specs = _build_run_specs()
    if args_cli.worker_run_index is None:
        raise RuntimeError("--worker-run-index is required in worker mode.")
    if args_cli.worker_output_dir is None:
        raise RuntimeError("--worker-output-dir is required in worker mode.")
    if args_cli.worker_run_index < 0 or args_cli.worker_run_index >= len(specs):
        raise IndexError(f"Worker run index {args_cli.worker_run_index} is out of range for {len(specs)} runs.")

    spec = specs[args_cli.worker_run_index]
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")
    if not spec.config.is_file():
        raise FileNotFoundError(f"Saved Dreamer config not found: {spec.config}")

    worker_output_dir = _resolve_path(args_cli.worker_output_dir)
    artifact_output_dir = _resolve_path(args_cli.worker_artifact_dir or args_cli.worker_output_dir)
    worker_output_dir.mkdir(parents=True, exist_ok=True)
    artifact_output_dir.mkdir(parents=True, exist_ok=True)

    result = _evaluate_dreamer_run(spec, artifact_output_dir)
    episode_csv = _write_episode_summary(worker_output_dir, result.rows)
    summary_csv = _write_summary_by_run(worker_output_dir, [result.summary])
    summary_json = _write_summary_json(worker_output_dir, [spec], [result], episode_csv, summary_csv)
    print(f"[INFO] Worker saved: {episode_csv}")
    print(f"[INFO] Worker saved: {summary_csv}")
    print(f"[INFO] Worker saved: {summary_json}")


def _append_option(argv: list[str], flag: str, value: Any) -> None:
    argv.extend((flag, str(value)))


def _worker_command(run_index: int, spec: RunSpec, worker_output_dir: Path, artifact_output_dir: Path) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--methods",
        *args_cli.methods,
        "--run-indices",
        *(str(index) for index in args_cli.run_indices),
        "--eval-episodes",
        str(args_cli.eval_episodes),
        "--num-envs",
        str(args_cli.num_envs),
        "--max-steps",
        str(args_cli.max_steps),
        "--success-trajectories-per-run",
        str(args_cli.success_trajectories_per_run),
        "--failure-trajectories-per-run",
        str(args_cli.failure_trajectories_per_run),
        "--camera-frame-stride",
        str(args_cli.camera_frame_stride),
        "--device",
        str(args_cli.device),
        "--env-device",
        str(args_cli.env_device),
        "--agent-device",
        str(args_cli.agent_device),
        "--worker-run-index",
        str(run_index),
        "--worker-output-dir",
        str(worker_output_dir),
        "--worker-artifact-dir",
        str(artifact_output_dir),
    ]
    if spec.method == "vision_dreamer" or args_cli.enable_cameras:
        argv.append("--enable_cameras")
    argv.append("--disable-fabric" if args_cli.disable_fabric else "--use-fabric")
    if args_cli.no_record_success_trajectories:
        argv.append("--no-record-success-trajectories")
    if args_cli.no_record_failure_trajectories:
        argv.append("--no-record-failure-trajectories")
    if args_cli.no_record_camera_frames:
        argv.append("--no-record-camera-frames")
    if getattr(args_cli, "headless", False):
        argv.append("--headless")
    if getattr(args_cli, "verbose", False):
        argv.append("--verbose")
    if getattr(args_cli, "info", False):
        argv.append("--info")

    livestream = getattr(args_cli, "livestream", None)
    if livestream in (0, 1, 2):
        _append_option(argv, "--livestream", livestream)
    experience = getattr(args_cli, "experience", None)
    if experience:
        _append_option(argv, "--experience", experience)
    kit_args = getattr(args_cli, "kit_args", None)
    if kit_args:
        _append_option(argv, "--kit_args", kit_args)

    _append_option(argv, "--state-run0-checkpoint", args_cli.state_run0_checkpoint)
    _append_option(argv, "--state-run1-checkpoint", args_cli.state_run1_checkpoint)
    _append_option(argv, "--vision-run0-checkpoint", args_cli.vision_run0_checkpoint)
    _append_option(argv, "--vision-run1-checkpoint", args_cli.vision_run1_checkpoint)
    argv.extend(overrides)
    return argv


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_worker_result(worker_output_dir: Path) -> RunResult:
    rows = _read_csv_rows(worker_output_dir / "episode_summary.csv")
    with (worker_output_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary_data = json.load(handle)
    summary = summary_data["summaries"][0]
    return RunResult(
        rows=rows,
        summary=summary,
        trajectory_paths=list(summary.get("trajectory_paths", [])),
        camera_frame_dirs=list(summary.get("camera_frame_dirs", [])),
    )


def _run_scheduler() -> None:
    specs = _build_run_specs()
    if not specs:
        raise RuntimeError("No runs selected. Check --methods and --run-indices.")
    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")
        if not spec.config.is_file():
            raise FileNotFoundError(f"Saved Dreamer config not found: {spec.config}")

    output_dir = _make_output_dir(args_cli.methods)
    worker_root = output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Dreamer replay results will be written to: {output_dir}")

    results: list[RunResult] = []
    for index, spec in enumerate(specs):
        worker_output_dir = worker_root / f"{spec.method}_run{spec.run_index}_seed{spec.seed}"
        command = _worker_command(index, spec, worker_output_dir, output_dir)
        print(f"[INFO] Running worker: {spec.run_name}")
        print("[INFO] " + " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        results.append(_load_worker_result(worker_output_dir))

    all_rows = [row for result in results for row in result.rows]
    summaries = [result.summary for result in results]
    episode_csv = _write_episode_summary(output_dir, all_rows)
    summary_csv = _write_summary_by_run(output_dir, summaries)
    summary_json = _write_summary_json(output_dir, specs, results, episode_csv, summary_csv)

    print(f"[INFO] Saved: {episode_csv}")
    print(f"[INFO] Saved: {summary_csv}")
    print(f"[INFO] Saved: {summary_json}")
    for summary in summaries:
        print(
            "[INFO] Summary: "
            f"{summary['run_name']} seed={summary['seed']} "
            f"episodes={summary['completed_episodes']}/{summary['requested_episodes']} "
            f"success_rate={summary['success_rate']}"
        )


def main() -> None:
    if args_cli.worker_run_index is not None:
        _run_worker()
    else:
        _run_scheduler()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
