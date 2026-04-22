"""Script to evaluate a checkpoint of an RL agent from RSL-RL.

Launch Isaac Sim Simulator first.
"""

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

# local imports
import cli_args  # isort: skip


STEP_METRIC_SPECS = (
    ("Metrics/absorbed_count", "absorbed_count"),
    ("Metrics/absorbed_ratio_mean", "absorbed_ratio"),
    ("Metrics/blood_centroid_distance", "blood_centroid_distance"),
    ("Metrics/valid_in_cone_ratio", "valid_in_cone_ratio"),
    ("Metrics/valid_in_inlet_ratio", "valid_in_inlet_ratio"),
    ("Metrics/raw_contact_force_mean", "raw_contact_force_mean"),
    ("Metrics/raw_contact_force_max", "raw_contact_force_max"),
    ("Metrics/psm_contact_force_mean", "psm_contact_force_mean"),
    ("Metrics/psm_contact_force_max", "psm_contact_force_max"),
)

VECTOR_TRACE_SPECS = (
    ("tip_pos_w", 3),
    ("blood_centroid_w", 3),
)

FINAL_METRIC_SPECS = STEP_METRIC_SPECS + (
    ("Metrics/initial_particle_count", "initial_particle_count"),
    ("Metrics/success_threshold", "success_threshold"),
)

EPISODE_REWARD_SPECS = (
    ("Episode_Reward/absorb_reward", "episode_reward_absorb_reward"),
    ("Episode_Reward/centroid_progress_reward", "episode_reward_centroid_progress_reward"),
    ("Episode_Reward/action_penalty", "episode_reward_action_penalty"),
    ("Episode_Reward/collision_force_penalty", "episode_reward_collision_force_penalty"),
    ("Episode_Reward/time_penalty", "episode_reward_time_penalty"),
    ("Episode_Reward/task_complete", "episode_reward_task_complete"),
)

TERMINATION_SPECS = (
    ("success", "Episode_Termination/success"),
    ("joint_limit", "Episode_Termination/joint_limit"),
    ("severe_collision", "Episode_Termination/severe_collision"),
    ("time_out", "Episode_Termination/time_out"),
)

CSV_FIELDNAMES = (
    "episode_id",
    "steps",
    "return",
    "success",
    "termination_reason",
    "absorbed_count_final",
    "absorbed_ratio_final",
    "blood_centroid_distance_final",
    "valid_in_cone_ratio_final",
    "valid_in_inlet_ratio_final",
    "raw_contact_force_mean_final",
    "raw_contact_force_max_final",
    "psm_contact_force_mean_final",
    "psm_contact_force_max_final",
    "initial_particle_count",
    "success_threshold",
    "episode_reward_absorb_reward",
    "episode_reward_centroid_progress_reward",
    "episode_reward_action_penalty",
    "episode_reward_collision_force_penalty",
    "episode_reward_time_penalty",
    "episode_reward_task_complete",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to simulate. This script only supports 1 for precise evaluation.",
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Number of episodes to evaluate.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Safety cap on rollout steps. Set to 0 to derive it from the episode length and eval_episodes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write episode_summary.csv, summary.json, and trajectories.npz.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed used for the environment.")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_arg_parser()
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.num_envs != 1:
    parser.error("This evaluation script only supports --num_envs 1 for precise episode statistics.")
if args_cli.eval_episodes <= 0:
    parser.error("--eval_episodes must be a positive integer.")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be greater than or equal to 0.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import head_blood_absorption.tasks  # noqa: F401


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


def _unwrap_actions(actions: Any) -> Any:
    if isinstance(actions, (tuple, list)):
        return actions[0]
    if isinstance(actions, Mapping):
        return actions.get("action", next(iter(actions.values())))
    return actions


def _extract_log_value(log_data: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    if key not in log_data:
        return default
    return _to_float(log_data[key], default=default)


def _empty_trace() -> dict[str, list[Any]]:
    trace = {"action": [], "reward": [], "done": []}
    for _, field_name in STEP_METRIC_SPECS:
        trace[field_name] = []
    for field_name, _ in VECTOR_TRACE_SPECS:
        trace[field_name] = []
    return trace


def _pack_trace(trace: Mapping[str, list[Any]], episode_id: int, terminated: bool) -> dict[str, Any]:
    packed: dict[str, Any] = {
        "episode_id": int(episode_id),
        "terminated": bool(terminated),
        "length": int(len(trace["reward"])),
        "action": np.asarray(trace["action"], dtype=np.float32),
        "reward": np.asarray(trace["reward"], dtype=np.float32),
        "done": np.asarray(trace["done"], dtype=np.bool_),
    }
    for _, field_name in STEP_METRIC_SPECS:
        packed[field_name] = np.asarray(trace[field_name], dtype=np.float32)
    for field_name, _ in VECTOR_TRACE_SPECS:
        packed[field_name] = np.asarray(trace[field_name], dtype=np.float32)
    return packed


def _snapshot_trace_vectors(raw_env: Any) -> dict[str, np.ndarray]:
    tip_pos_w, _ = raw_env._compute_tip_pose_and_direction_w()
    blood_centroid_w = raw_env._particle_state.blood_centroid
    return {
        "tip_pos_w": np.asarray(_to_numpy(tip_pos_w), dtype=np.float32).reshape(raw_env.num_envs, -1)[0].copy(),
        "blood_centroid_w": np.asarray(_to_numpy(blood_centroid_w), dtype=np.float32)
        .reshape(raw_env.num_envs, -1)[0]
        .copy(),
    }


def _pad_float_series(
    trajectories: list[dict[str, Any]], key: str, sample_shape: tuple[int, ...] = ()
) -> np.ndarray:
    if not trajectories:
        return np.empty((0, 0, *sample_shape), dtype=np.float32)

    lengths = np.asarray([traj["length"] for traj in trajectories], dtype=np.int32)
    max_length = int(lengths.max()) if lengths.size > 0 else 0

    if sample_shape:
        padded = np.full((len(trajectories), max_length, *sample_shape), np.nan, dtype=np.float32)
    else:
        padded = np.full((len(trajectories), max_length), np.nan, dtype=np.float32)

    for index, traj in enumerate(trajectories):
        length = int(traj["length"])
        if length > 0:
            padded[index, :length] = traj[key]
    return padded


def _pad_bool_series(trajectories: list[dict[str, Any]], key: str) -> np.ndarray:
    if not trajectories:
        return np.empty((0, 0), dtype=np.bool_)

    lengths = np.asarray([traj["length"] for traj in trajectories], dtype=np.int32)
    max_length = int(lengths.max()) if lengths.size > 0 else 0
    padded = np.zeros((len(trajectories), max_length), dtype=np.bool_)

    for index, traj in enumerate(trajectories):
        length = int(traj["length"])
        if length > 0:
            padded[index, :length] = traj[key]
    return padded


def _save_trajectories_npz(trajectories: list[dict[str, Any]], output_dir: Path) -> Path:
    output_path = output_dir / "trajectories.npz"

    if trajectories:
        action_dim = int(trajectories[0]["action"].shape[1]) if trajectories[0]["action"].ndim == 2 else 0
    else:
        action_dim = 0

    arrays: dict[str, np.ndarray] = {
        "episode_ids": np.asarray([traj["episode_id"] for traj in trajectories], dtype=np.int32),
        "lengths": np.asarray([traj["length"] for traj in trajectories], dtype=np.int32),
        "terminated": np.asarray([traj["terminated"] for traj in trajectories], dtype=np.bool_),
        "action": _pad_float_series(trajectories, "action", sample_shape=(action_dim,)),
        "reward": _pad_float_series(trajectories, "reward"),
        "done": _pad_bool_series(trajectories, "done"),
    }
    for _, field_name in STEP_METRIC_SPECS:
        arrays[field_name] = _pad_float_series(trajectories, field_name)
    for field_name, width in VECTOR_TRACE_SPECS:
        arrays[field_name] = _pad_float_series(trajectories, field_name, sample_shape=(width,))

    np.savez_compressed(output_path, **arrays)
    return output_path


def _write_episode_csv(episode_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_path = output_dir / "episode_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)
    return output_path


def _make_output_dir(run_dir: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    root = Path(run_dir).resolve() / "eval"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_resume_path(log_root_path: str, agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    if args_cli.checkpoint is not None:
        checkpoint_candidate = os.path.abspath(args_cli.checkpoint)
        if os.path.isfile(checkpoint_candidate):
            return checkpoint_candidate

    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _resolve_termination_reason(log_data: Mapping[str, Any]) -> str:
    active_reasons = [
        reason for reason, key in TERMINATION_SPECS if _extract_log_value(log_data, key, default=0.0) > 0.5
    ]
    if len(active_reasons) == 1:
        return active_reasons[0]
    if len(active_reasons) > 1:
        raise RuntimeError(f"Ambiguous termination flags in episode log: {active_reasons}")
    raise RuntimeError("Unable to resolve termination reason from episode log.")


def _build_episode_summary(
    episode_id: int,
    episode_return: float,
    episode_steps: int,
    log_data: Mapping[str, Any],
) -> dict[str, Any]:
    termination_reason = _resolve_termination_reason(log_data)
    row: dict[str, Any] = {
        "episode_id": int(episode_id),
        "steps": int(episode_steps),
        "return": float(episode_return),
        "success": termination_reason == "success",
        "termination_reason": termination_reason,
    }

    for log_key, field_name in FINAL_METRIC_SPECS:
        row[f"{field_name}_final" if field_name in {name for _, name in STEP_METRIC_SPECS} else field_name] = (
            _extract_log_value(log_data, log_key)
        )
    for log_key, field_name in EPISODE_REWARD_SPECS:
        row[field_name] = _extract_log_value(log_data, log_key)

    return row


def _stats_from_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite_array = array[np.isfinite(array)]
    if finite_array.size <= 0:
        return None
    return {
        "mean": float(finite_array.mean()),
        "std": float(finite_array.std()),
        "min": float(finite_array.min()),
        "max": float(finite_array.max()),
    }


def _build_run_summary(
    episode_rows: list[dict[str, Any]],
    requested_episodes: int,
    completed_episodes: int,
    total_steps: int,
    episode_limit: int,
    checkpoint_path: str,
    output_dir: Path,
    max_steps: int,
) -> dict[str, Any]:
    termination_counts = {reason: 0 for reason, _ in TERMINATION_SPECS}
    for row in episode_rows:
        termination_counts[row["termination_reason"]] += 1

    if completed_episodes > 0:
        termination_rates = {
            reason: float(count) / float(completed_episodes) for reason, count in termination_counts.items()
        }
    else:
        termination_rates = {reason: 0.0 for reason in termination_counts}

    aggregates: dict[str, dict[str, float] | None] = {}
    aggregate_fields = [field for field in CSV_FIELDNAMES if field not in {"episode_id", "success", "termination_reason"}]
    for field in aggregate_fields:
        aggregates[field] = _stats_from_values([float(row[field]) for row in episode_rows])

    success_rate = None
    if completed_episodes > 0:
        success_rate = float(np.mean([float(bool(row["success"])) for row in episode_rows]))

    return {
        "task": args_cli.task,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "output_dir": str(output_dir),
        "complete": completed_episodes >= requested_episodes,
        "requested_episodes": int(requested_episodes),
        "completed_episodes": int(completed_episodes),
        "max_steps": int(max_steps),
        "executed_steps": int(total_steps),
        "episode_limit": int(episode_limit),
        "num_envs": int(args_cli.num_envs),
        "video_enabled": bool(args_cli.video),
        "seed": args_cli.seed,
        "success_rate": success_rate,
        "termination_counts": termination_counts,
        "termination_rates": termination_rates,
        "aggregates": aggregates,
    }


def _write_summary_json(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=True)
        json_file.write("\n")
    return output_path


def _print_console_summary(summary: Mapping[str, Any]) -> None:
    print(
        "[INFO] Evaluation summary: "
        f"episodes={summary['completed_episodes']}/{summary['requested_episodes']}, "
        f"complete={summary['complete']}, "
        f"steps={summary['executed_steps']}"
    )
    if summary["success_rate"] is not None:
        print(f"[INFO] Success rate: {summary['success_rate']:.4f}")

    return_stats = summary["aggregates"].get("return")
    steps_stats = summary["aggregates"].get("steps")
    absorbed_ratio_stats = summary["aggregates"].get("absorbed_ratio_final")

    if return_stats is not None:
        print(
            "[INFO] Return stats: "
            f"mean={return_stats['mean']:.4f}, std={return_stats['std']:.4f}, "
            f"min={return_stats['min']:.4f}, max={return_stats['max']:.4f}"
        )
    if steps_stats is not None:
        print(
            "[INFO] Episode length stats: "
            f"mean={steps_stats['mean']:.2f}, min={steps_stats['min']:.0f}, max={steps_stats['max']:.0f}"
        )
    if absorbed_ratio_stats is not None:
        print(
            "[INFO] Final absorbed ratio stats: "
            f"mean={absorbed_ratio_stats['mean']:.4f}, "
            f"min={absorbed_ratio_stats['min']:.4f}, max={absorbed_ratio_stats['max']:.4f}"
        )
    print(f"[INFO] Termination counts: {summary['termination_counts']}")


def main():
    """Evaluate an RSL-RL checkpoint for a fixed number of episodes."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = _resolve_resume_path(log_root_path, agent_cfg)
    run_dir = os.path.dirname(resume_path)
    output_dir = _make_output_dir(run_dir, args_cli.output_dir)
    print(f"[INFO] Evaluation results will be written to: {output_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            video_kwargs = {
                "video_folder": str(output_dir / "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during evaluation.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env)
        raw_env = env.unwrapped

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=raw_env.device)
        if hasattr(policy, "eval"):
            policy.eval()

        obs, _ = env.get_observations()

        episode_limit = int(getattr(raw_env, "max_episode_length", 0))
        max_steps = int(args_cli.max_steps)
        if max_steps == 0:
            if episode_limit <= 0:
                raise RuntimeError(
                    "Unable to infer --max_steps because the environment does not expose max_episode_length."
                )
            max_steps = episode_limit * (args_cli.eval_episodes + 2)

        print(
            "[INFO] Starting evaluation rollout: "
            f"target_episodes={args_cli.eval_episodes}, max_steps={max_steps}, episode_limit={episode_limit}"
        )

        completed_rows: list[dict[str, Any]] = []
        saved_trajectories: list[dict[str, Any]] = []
        current_trace = _empty_trace()
        current_episode_return = 0.0
        current_episode_steps = 0
        total_steps = 0

        with torch.inference_mode():
            while (
                simulation_app.is_running()
                and total_steps < max_steps
                and len(completed_rows) < args_cli.eval_episodes
            ):
                # Save the state before applying the action so terminal steps are not polluted by auto-reset.
                trace_vectors = _snapshot_trace_vectors(raw_env)
                actions = _unwrap_actions(policy(obs))
                obs, reward, dones, extras = env.step(actions)

                log_data = extras.get("log", {}) if isinstance(extras, Mapping) else {}
                if not log_data and hasattr(raw_env, "extras"):
                    raw_log_data = getattr(raw_env, "extras", {}).get("log", {})
                    if isinstance(raw_log_data, Mapping):
                        log_data = raw_log_data

                action_step = np.asarray(_to_numpy(actions), dtype=np.float32).reshape(args_cli.num_envs, -1)[0].copy()
                reward_value = float(np.asarray(_to_numpy(reward), dtype=np.float32).reshape(-1)[0])
                done_value = bool(np.asarray(_to_numpy(dones)).reshape(-1)[0])

                current_trace["action"].append(action_step)
                current_trace["reward"].append(reward_value)
                current_trace["done"].append(done_value)
                for log_key, field_name in STEP_METRIC_SPECS:
                    current_trace[field_name].append(_extract_log_value(log_data, log_key))
                for field_name, _ in VECTOR_TRACE_SPECS:
                    current_trace[field_name].append(trace_vectors[field_name])

                current_episode_return += reward_value
                current_episode_steps += 1
                total_steps += 1

                if done_value:
                    episode_id = len(completed_rows) + 1
                    episode_row = _build_episode_summary(
                        episode_id=episode_id,
                        episode_return=current_episode_return,
                        episode_steps=current_episode_steps,
                        log_data=log_data,
                    )
                    completed_rows.append(episode_row)
                    saved_trajectories.append(_pack_trace(current_trace, episode_id=episode_id, terminated=True))

                    print(
                        "[INFO] Episode completed: "
                        f"{episode_id}/{args_cli.eval_episodes}, "
                        f"steps={episode_row['steps']}, return={episode_row['return']:.4f}, "
                        f"success={episode_row['success']}, reason={episode_row['termination_reason']}, "
                        f"absorbed_ratio={episode_row['absorbed_ratio_final']:.4f}"
                    )

                    current_trace = _empty_trace()
                    current_episode_return = 0.0
                    current_episode_steps = 0

        if current_episode_steps > 0:
            partial_episode_id = len(completed_rows) + 1
            saved_trajectories.append(_pack_trace(current_trace, episode_id=partial_episode_id, terminated=False))
            print(
                "[INFO] Saved a partial trajectory: "
                f"episode_id={partial_episode_id}, steps={current_episode_steps}, terminated=False"
            )

        csv_path = _write_episode_csv(completed_rows, output_dir)
        npz_path = _save_trajectories_npz(saved_trajectories, output_dir)
        summary = _build_run_summary(
            episode_rows=completed_rows,
            requested_episodes=args_cli.eval_episodes,
            completed_episodes=len(completed_rows),
            total_steps=total_steps,
            episode_limit=episode_limit,
            checkpoint_path=resume_path,
            output_dir=output_dir,
            max_steps=max_steps,
        )
        json_path = _write_summary_json(summary, output_dir)

        _print_console_summary(summary)
        print(f"[INFO] Saved: {csv_path}")
        print(f"[INFO] Saved: {json_path}")
        print(f"[INFO] Saved: {npz_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
