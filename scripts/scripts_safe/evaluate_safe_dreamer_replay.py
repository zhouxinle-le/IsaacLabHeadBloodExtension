#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path("scripts/scripts_safe/results")
PLAY_SCRIPT = Path("source/r2dreamer_isaac/play.py")

DEFAULT_CHECKPOINTS = {
    "state_dreamer": Path(
        "logs/r2dreamer/ur3_blood_pipe_state_dreamer/"
        "2026-05-15_21-32-23/checkpoints/policy_step_500004.pt"
    ),
    "state_safe_dreamer": Path(
        "logs/r2dreamer/ur3_blood_pipe_state_safe_dreamer/"
        "2026-05-15_09-18-25/checkpoints/policy_step_500004.pt"
    ),
    "vision_dreamer": Path(
        "logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/"
        "2026-05-12_21-14-29_seed_0_600k/checkpoints/policy_step_500001.pt"
    ),
    "vision_safe_dreamer": Path(
        "logs/r2dreamer/ur3_blood_pipe_vision_wrist_safe_dreamer/"
        "2026-05-16_06-36-51/checkpoints/policy_step_500002.pt"
    ),
}

METHOD_LABELS = {
    "state_dreamer": ("state", "Dreamer", "State Dreamer"),
    "state_safe_dreamer": ("state", "Safe-Dreamer", "State Safe-Dreamer"),
    "vision_dreamer": ("vision", "Dreamer", "Vision Wrist Dreamer"),
    "vision_safe_dreamer": ("vision", "Safe-Dreamer", "Vision Wrist Safe-Dreamer"),
}

EPISODE_NUMERIC_FIELDS = (
    "return",
    "episode_length",
    "absorbed_ratio_final",
    "ur3_contact_force_max",
    "tip_goal_error_mean",
    "tip_pipe_clearance_mean",
)

SUMMARY_RATE_FIELDS = (
    "success_rate",
    "safe_success_rate",
    "severe_collision_rate",
    "time_out_rate",
)

SUMMARY_NUMERIC_FIELDS = (
    "return_mean",
    "return_std",
    "episode_length_mean",
    "episode_length_std",
    "absorbed_ratio_final_mean",
    "absorbed_ratio_final_std",
    "ur3_contact_force_max_mean",
    "ur3_contact_force_max_std",
    "tip_goal_error_mean_mean",
    "tip_goal_error_mean_std",
    "tip_pipe_clearance_mean_mean",
    "tip_pipe_clearance_mean_std",
)

EPISODE_OUTPUT_FIELDS = (
    "method",
    "observation",
    "algorithm_label",
    "eval_seed",
    "run_output_dir",
    "algorithm",
    "task",
    "seed",
    "checkpoint",
    "episode_id",
    "return",
    "episode_length",
    "success",
    "safe_success",
    "severe_collision",
    "time_out",
    "absorbed_ratio_final",
    "ur3_contact_force_max",
    "tip_goal_error_mean",
    "tip_pipe_clearance_mean",
)

SUMMARY_BY_SEED_FIELDS = (
    "method",
    "observation",
    "algorithm_label",
    "eval_seed",
    "checkpoint",
    "run_output_dir",
    "requested_episodes",
    "completed_episodes",
    "complete",
    "success_count",
    "success_rate",
    "safe_success_count",
    "safe_success_rate",
    "severe_collision_count",
    "severe_collision_rate",
    "time_out_count",
    "time_out_rate",
    *SUMMARY_NUMERIC_FIELDS,
)

SUMMARY_BY_METHOD_FIELDS = (
    "method",
    "observation",
    "algorithm_label",
    "checkpoint",
    "num_eval_seeds",
    "requested_episodes_per_seed",
    "completed_episodes_total",
    "complete_seed_count",
    "success_rate_seed_mean",
    "success_rate_seed_std",
    "safe_success_rate_seed_mean",
    "safe_success_rate_seed_std",
    "severe_collision_rate_seed_mean",
    "severe_collision_rate_seed_std",
    "time_out_rate_seed_mean",
    "time_out_rate_seed_std",
    "return_mean_seed_mean",
    "return_mean_seed_std",
    "episode_length_mean_seed_mean",
    "episode_length_mean_seed_std",
    "absorbed_ratio_final_mean_seed_mean",
    "absorbed_ratio_final_mean_seed_std",
    "ur3_contact_force_max_mean_seed_mean",
    "ur3_contact_force_max_mean_seed_std",
    "tip_goal_error_mean_mean_seed_mean",
    "tip_goal_error_mean_mean_seed_std",
    "tip_pipe_clearance_mean_mean_seed_mean",
    "tip_pipe_clearance_mean_mean_seed_std",
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    observation: str
    algorithm_label: str
    display_name: str
    checkpoint: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay-evaluate Dreamer and Safe-Dreamer checkpoints with seeds 0/1/2 "
            "and aggregate paper-ready CSV/JSON results."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(DEFAULT_CHECKPOINTS),
        default=list(DEFAULT_CHECKPOINTS),
        help="Methods to evaluate.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Evaluation seeds.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per method and seed.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of vectorized envs for play.py. Use 1 for clean per-episode statistics.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Isaac AppLauncher device argument.")
    parser.add_argument("--env-device", type=str, default="cpu", help="Isaac environment device.")
    parser.add_argument("--agent-device", type=str, default="cuda:0", help="Dreamer policy device.")
    parser.add_argument("--state-dreamer-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["state_dreamer"])
    parser.add_argument("--state-safe-dreamer-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["state_safe_dreamer"])
    parser.add_argument("--vision-dreamer-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["vision_dreamer"])
    parser.add_argument(
        "--vision-safe-dreamer-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINTS["vision_safe_dreamer"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to scripts/scripts_safe/results/<timestamp>_safe_dreamer_replay.",
    )
    parser.add_argument("--python", type=Path, default=None, help="Python executable. Defaults to sys.executable.")
    parser.add_argument("--use-fabric", action="store_true", default=False, help="Do not pass --disable_fabric.")
    parser.add_argument("--no-headless", action="store_true", default=True, help="Do not pass --headless.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without running them.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip a method/seed when its output directory already has a complete summary.json.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=False,
        help="Continue evaluating remaining runs if one subprocess fails.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _checkpoint_overrides(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "state_dreamer": args.state_dreamer_checkpoint,
        "state_safe_dreamer": args.state_safe_dreamer_checkpoint,
        "vision_dreamer": args.vision_dreamer_checkpoint,
        "vision_safe_dreamer": args.vision_safe_dreamer_checkpoint,
    }


def _build_specs(args: argparse.Namespace) -> list[RunSpec]:
    checkpoint_overrides = _checkpoint_overrides(args)
    specs: list[RunSpec] = []
    for method in dict.fromkeys(args.methods):
        observation, algorithm_label, display_name = METHOD_LABELS[method]
        specs.append(
            RunSpec(
                method=method,
                observation=observation,
                algorithm_label=algorithm_label,
                display_name=display_name,
                checkpoint=_resolve(checkpoint_overrides[method]),
            )
        )
    return specs


def _make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = _resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    root = _resolve(RESULTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"{timestamp}_safe_dreamer_replay_{len(args.seeds)}seed_{args.episodes}ep"
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{timestamp}_safe_dreamer_replay_{len(args.seeds)}seed_{args.episodes}ep_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _run_output_dir(output_dir: Path, spec: RunSpec, seed: int) -> Path:
    return output_dir / "runs" / spec.method / f"seed_{seed}"


def _is_complete_run(output_dir: Path, episodes: int) -> bool:
    summary_path = output_dir / "summary.json"
    episode_path = output_dir / "episode_summary.csv"
    if not summary_path.is_file() or not episode_path.is_file():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:
        return False
    return bool(summary.get("complete")) and int(summary.get("completed_episodes", 0)) >= int(episodes)


def _play_command(args: argparse.Namespace, spec: RunSpec, seed: int, run_output_dir: Path) -> list[str]:
    python_exe = str(_resolve(args.python)) if args.python is not None else sys.executable
    command = [
        python_exe,
        str(_resolve(PLAY_SCRIPT)),
        "--checkpoint",
        str(spec.checkpoint),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--num_envs",
        str(args.num_envs),
        "--device",
        str(args.device),
        "--env_device",
        str(args.env_device),
        "--agent_device",
        str(args.agent_device),
        "--output_dir",
        str(run_output_dir),
    ]
    if not args.use_fabric:
        command.append("--disable_fabric")
    if not args.no_headless:
        command.append("--headless")
    if spec.observation == "vision":
        command.append("--enable_cameras")
    return command


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def _run_subprocess(command: list[str], command_log: Path) -> None:
    command_text = " ".join(shlex.quote(part) for part in command)
    print(f"[INFO] {command_text}")
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write(command_text)
        handle.write("\n")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return value


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(finite),
        "max": max(finite),
    }


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in writer.fieldnames})


def _load_episode_rows(output_dir: Path, spec: RunSpec, seed: int) -> list[dict[str, Any]]:
    rows = _read_csv(output_dir / "episode_summary.csv")
    enriched: list[dict[str, Any]] = []
    for row in rows:
        success = _parse_bool(row.get("success"))
        severe_collision = _parse_bool(row.get("severe_collision"))
        enriched.append(
            {
                **row,
                "method": spec.method,
                "observation": spec.observation,
                "algorithm_label": spec.algorithm_label,
                "eval_seed": int(seed),
                "run_output_dir": str(output_dir),
                "success": success,
                "safe_success": bool(success and not severe_collision),
                "severe_collision": severe_collision,
                "time_out": _parse_bool(row.get("time_out")),
            }
        )
    return enriched


def _summary_for_seed(spec: RunSpec, seed: int, run_output_dir: Path, rows: list[dict[str, Any]], episodes: int) -> dict[str, Any]:
    completed = len(rows)
    success_count = sum(1 for row in rows if bool(row["success"]))
    safe_success_count = sum(1 for row in rows if bool(row["safe_success"]))
    severe_count = sum(1 for row in rows if bool(row["severe_collision"]))
    timeout_count = sum(1 for row in rows if bool(row["time_out"]))

    summary: dict[str, Any] = {
        "method": spec.method,
        "observation": spec.observation,
        "algorithm_label": spec.algorithm_label,
        "eval_seed": int(seed),
        "checkpoint": str(spec.checkpoint),
        "run_output_dir": str(run_output_dir),
        "requested_episodes": int(episodes),
        "completed_episodes": int(completed),
        "complete": completed >= episodes,
        "success_count": int(success_count),
        "success_rate": success_count / completed if completed else None,
        "safe_success_count": int(safe_success_count),
        "safe_success_rate": safe_success_count / completed if completed else None,
        "severe_collision_count": int(severe_count),
        "severe_collision_rate": severe_count / completed if completed else None,
        "time_out_count": int(timeout_count),
        "time_out_rate": timeout_count / completed if completed else None,
    }
    for field in EPISODE_NUMERIC_FIELDS:
        stats = _stats(_parse_float(row.get(field)) for row in rows)
        summary[f"{field}_mean"] = stats["mean"]
        summary[f"{field}_std"] = stats["std"]
    return summary


def _summary_by_method(seed_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in seed_summaries:
        grouped[str(summary["method"])].append(summary)

    rows: list[dict[str, Any]] = []
    for method, summaries in grouped.items():
        first = summaries[0]
        row: dict[str, Any] = {
            "method": method,
            "observation": first["observation"],
            "algorithm_label": first["algorithm_label"],
            "checkpoint": first["checkpoint"],
            "num_eval_seeds": len(summaries),
            "requested_episodes_per_seed": first["requested_episodes"],
            "completed_episodes_total": sum(int(summary["completed_episodes"]) for summary in summaries),
            "complete_seed_count": sum(1 for summary in summaries if bool(summary["complete"])),
        }
        for field in SUMMARY_RATE_FIELDS:
            stats = _stats(
                float(summary[field])
                for summary in summaries
                if summary.get(field) is not None
            )
            row[f"{field}_seed_mean"] = stats["mean"]
            row[f"{field}_seed_std"] = stats["std"]
        for field in (
            "return_mean",
            "episode_length_mean",
            "absorbed_ratio_final_mean",
            "ur3_contact_force_max_mean",
            "tip_goal_error_mean_mean",
            "tip_pipe_clearance_mean_mean",
        ):
            stats = _stats(
                float(summary[field])
                for summary in summaries
                if summary.get(field) is not None
            )
            row[f"{field}_seed_mean"] = stats["mean"]
            row[f"{field}_seed_std"] = stats["std"]
        rows.append(row)
    return rows


def _write_summary_json(
    output_dir: Path,
    args: argparse.Namespace,
    specs: list[RunSpec],
    failures: list[dict[str, Any]],
) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "episodes_per_seed": int(args.episodes),
        "eval_seeds": [int(seed) for seed in args.seeds],
        "num_envs": int(args.num_envs),
        "device": args.device,
        "env_device": args.env_device,
        "agent_device": args.agent_device,
        "use_fabric": bool(args.use_fabric),
        "headless": not bool(args.no_headless),
        "methods": [
            {
                "method": spec.method,
                "observation": spec.observation,
                "algorithm_label": spec.algorithm_label,
                "display_name": spec.display_name,
                "checkpoint": str(spec.checkpoint),
            }
            for spec in specs
        ],
        "files": {
            "episode_summary_all": str(output_dir / "episode_summary_all.csv"),
            "summary_by_seed": str(output_dir / "summary_by_seed.csv"),
            "summary_by_method": str(output_dir / "summary_by_method.csv"),
            "commands": str(output_dir / "commands.txt"),
        },
        "failures": failures,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    args = _parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")

    specs = _build_specs(args)
    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {spec.method}: {spec.checkpoint}")

    output_dir = _make_output_dir(args)
    command_log = output_dir / "commands.txt"
    _write_text(command_log, "")
    print(f"[INFO] Results will be written to: {output_dir}")

    failures: list[dict[str, Any]] = []
    for spec in specs:
        for seed in args.seeds:
            run_output_dir = _run_output_dir(output_dir, spec, int(seed))
            run_output_dir.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and _is_complete_run(run_output_dir, args.episodes):
                print(f"[INFO] Skip complete run: {spec.method} seed={seed}")
                continue
            command = _play_command(args, spec, int(seed), run_output_dir)
            if args.dry_run:
                print("[DRY-RUN] " + " ".join(shlex.quote(part) for part in command))
                with command_log.open("a", encoding="utf-8") as handle:
                    handle.write(" ".join(shlex.quote(part) for part in command))
                    handle.write("\n")
                continue
            try:
                _run_subprocess(command, command_log)
            except subprocess.CalledProcessError as exc:
                failure = {"method": spec.method, "seed": int(seed), "returncode": exc.returncode}
                failures.append(failure)
                print(f"[ERROR] Failed: {failure}")
                if not args.keep_going:
                    raise

    if args.dry_run:
        print(f"[INFO] Dry run only. Commands saved to: {command_log}")
        return

    all_episode_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for spec in specs:
        for seed in args.seeds:
            run_output_dir = _run_output_dir(output_dir, spec, int(seed))
            episode_csv = run_output_dir / "episode_summary.csv"
            if not episode_csv.is_file():
                if args.keep_going:
                    continue
                raise FileNotFoundError(f"Missing episode summary: {episode_csv}")
            rows = _load_episode_rows(run_output_dir, spec, int(seed))
            all_episode_rows.extend(rows)
            seed_summaries.append(_summary_for_seed(spec, int(seed), run_output_dir, rows, args.episodes))

    method_summaries = _summary_by_method(seed_summaries)
    _write_csv(output_dir / "episode_summary_all.csv", EPISODE_OUTPUT_FIELDS, all_episode_rows)
    _write_csv(output_dir / "summary_by_seed.csv", SUMMARY_BY_SEED_FIELDS, seed_summaries)
    _write_csv(output_dir / "summary_by_method.csv", SUMMARY_BY_METHOD_FIELDS, method_summaries)
    _write_summary_json(output_dir, args, specs, failures)

    print(f"[INFO] Saved: {output_dir / 'episode_summary_all.csv'}")
    print(f"[INFO] Saved: {output_dir / 'summary_by_seed.csv'}")
    print(f"[INFO] Saved: {output_dir / 'summary_by_method.csv'}")
    for row in method_summaries:
        print(
            "[INFO] Summary: "
            f"{row['method']} success={row['success_rate_seed_mean']} "
            f"severe_collision={row['severe_collision_rate_seed_mean']} "
            f"contact_max={row['ur3_contact_force_max_mean_seed_mean']}"
        )


if __name__ == "__main__":
    main()
