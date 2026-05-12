#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC_PYTHON = Path("/home/le/miniconda3/envs/isaacsim-4.2/bin/python")


@dataclass(frozen=True)
class EvalRun:
    name: str
    seed: int
    run_dir: Path
    checkpoint: Path
    output_dir: Path
    argv: list[str]

    def shell(self) -> str:
        return " ".join(shlex.quote(str(part)) for part in self.argv)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or execute final 20-episode evaluation commands for the paper runs."
    )
    parser.add_argument("--execute", action="store_true", help="Run evaluation commands sequentially.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Training seeds.")
    parser.add_argument("--eval-seed", type=int, default=1000, help="Shared evaluation environment seed.")
    parser.add_argument("--episodes", type=int, default=20, help="Completed episodes per trained policy.")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON, help="Python inside isaacsim-4.2.")
    parser.add_argument("--dreamer-python", type=str, default="python", help="Python used for Dreamer eval.")
    parser.add_argument("--device", type=str, default="cpu", help="Isaac simulation device.")
    parser.add_argument("--env-device", type=str, default="cpu", help="Dreamer Isaac environment device.")
    parser.add_argument("--agent-device", type=str, default="cuda:0", help="Dreamer policy device.")
    parser.add_argument("--run-label", type=str, default="800k", help="Suffix used to discover training runs.")
    parser.add_argument("--target-policy-step", type=int, default=800_000, help="Dreamer checkpoint step to prefer.")
    parser.add_argument("--output-root", type=Path, default=Path("logs/paper_final_eval"), help="Eval output root.")
    parser.add_argument(
        "--only",
        choices=["all", "state-rsl", "state-dreamer", "vision-skrl", "vision-dreamer"],
        default="all",
        help="Limit the generated command group.",
    )
    return parser.parse_args()


def _latest_dir(pattern: str) -> Path | None:
    matches = [path for path in REPO_ROOT.glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _latest_numbered_checkpoint(run_dir: Path, prefix: str, suffix: str = ".pt") -> Path | None:
    best: tuple[int, Path] | None = None
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    for path in run_dir.glob(f"{prefix}*{suffix}"):
        match = pattern.match(path.name)
        if match is None:
            continue
        value = int(match.group(1))
        if best is None or value > best[0]:
            best = (value, path)
    return best[1] if best is not None else None


def _closest_policy_checkpoint(run_dir: Path, target_step: int) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    best: tuple[int, int, Path] | None = None
    pattern = re.compile(r"^policy_step_(\d+)\.pt$")
    for path in checkpoint_dir.glob("policy_step_*.pt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        rank = (abs(step - target_step), -step, path)
        if best is None or rank < best:
            best = rank
    if best is not None:
        return best[2]
    latest = checkpoint_dir / "latest.pt"
    return latest if latest.is_file() else None


def _state_rsl(seed: int, args: argparse.Namespace) -> EvalRun | None:
    run_dir = _latest_dir(f"logs/rsl_rl/ur3_blood_pipe_state_direct/*seed_{seed}_{args.run_label}")
    if run_dir is None:
        return None
    checkpoint = _latest_numbered_checkpoint(run_dir, "model_")
    if checkpoint is None:
        return None
    output_dir = (REPO_ROOT / args.output_root / "state_rsl_ppo" / f"seed_{seed}").resolve()
    argv = [
        str(args.isaac_python),
        str(REPO_ROOT / "scripts/rsl_rl/eval.py"),
        "--task",
        "Isaac-Ur3-Blood-Pipe-State-Direct-v0",
        "--experiment_name",
        "ur3_blood_pipe_state_direct",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--eval_episodes",
        str(args.episodes),
        "--num_envs",
        "1",
        "--seed",
        str(args.eval_seed),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--disable_fabric",
        "--enable_cameras",
    ]
    return EvalRun("state-rsl", seed, run_dir, checkpoint, output_dir, argv)


def _vision_skrl(seed: int, args: argparse.Namespace) -> EvalRun | None:
    run_dir = _latest_dir(f"logs/skrl/ur3_blood_pipe_vision_direct_wrist/*seed_{seed}_{args.run_label}")
    if run_dir is None:
        return None
    checkpoint = run_dir / "checkpoints" / "best_agent.pt"
    if not checkpoint.is_file():
        return None
    output_dir = (REPO_ROOT / args.output_root / "vision_skrl_ppo" / f"seed_{seed}").resolve()
    argv = [
        str(args.isaac_python),
        str(REPO_ROOT / "source/skrl/eval.py"),
        "--task",
        "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--episodes",
        str(args.episodes),
        "--num_envs",
        "1",
        "--seed",
        str(args.eval_seed),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--disable_fabric",
        "--enable_cameras",
    ]
    return EvalRun("vision-skrl", seed, run_dir, checkpoint, output_dir, argv)


def _dreamer(seed: int, args: argparse.Namespace, task: str, group: str, run_pattern: str) -> EvalRun | None:
    run_dir = _latest_dir(run_pattern)
    if run_dir is None:
        return None
    checkpoint = _closest_policy_checkpoint(run_dir, target_step=args.target_policy_step)
    if checkpoint is None:
        return None
    output_dir = (REPO_ROOT / args.output_root / group / f"seed_{seed}").resolve()
    argv = [
        args.dreamer_python,
        str(REPO_ROOT / "source/r2dreamer_isaac/play.py"),
        "--task",
        task,
        "--cfg_entry_point",
        "dreamer_cfg_entry_point",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--episodes",
        str(args.episodes),
        "--num_envs",
        "1",
        "--seed",
        str(args.eval_seed),
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--env_device",
        args.env_device,
        "--agent_device",
        args.agent_device,
        "--disable_fabric",
        "--enable_cameras",
    ]
    return EvalRun(group.replace("_", "-"), seed, run_dir, checkpoint, output_dir, argv)


def _state_dreamer(seed: int, args: argparse.Namespace) -> EvalRun | None:
    return _dreamer(
        seed,
        args,
        "Isaac-Ur3-Blood-Pipe-State-Direct-v0",
        "state_dreamer_v3",
        f"logs/r2dreamer/ur3_blood_pipe_state_dreamer/*seed_{seed}_{args.run_label}",
    )


def _vision_dreamer(seed: int, args: argparse.Namespace) -> EvalRun | None:
    return _dreamer(
        seed,
        args,
        "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
        "vision_dreamer_v3",
        f"logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/*seed_{seed}_{args.run_label}",
    )


def _runs(args: argparse.Namespace) -> list[EvalRun]:
    builders = {
        "state-rsl": _state_rsl,
        "state-dreamer": _state_dreamer,
        "vision-skrl": _vision_skrl,
        "vision-dreamer": _vision_dreamer,
    }
    selected = builders if args.only == "all" else {args.only: builders[args.only]}
    runs: list[EvalRun] = []
    for name, builder in selected.items():
        for seed in args.seeds:
            run = builder(seed, args)
            if run is None:
                print(f"[WARN] Missing run/checkpoint for {name} seed={seed}")
                continue
            runs.append(run)
    return runs


def main() -> None:
    args = _parse_args()
    runs = _runs(args)
    for run in runs:
        print(f"\n# {run.name} seed={run.seed}")
        print(f"# run_dir={run.run_dir}")
        print(f"# checkpoint={run.checkpoint}")
        print(run.shell())

    if not args.execute:
        print("\n[INFO] Dry run only. Re-run with --execute to start final evaluation sequentially.")
        return

    for run in runs:
        print(f"\n[INFO] Evaluating {run.name} seed={run.seed}")
        subprocess.run(run.argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
