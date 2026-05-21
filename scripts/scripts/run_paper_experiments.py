#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISAAC_PYTHON = Path("/home/le/miniconda3/envs/isaacsim-4.2/bin/python")


@dataclass(frozen=True)
class ExperimentCommand:
    name: str
    seed: int
    argv: list[str]

    def shell(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or execute the paper training matrix for PPO, PPO-Lagrangian, and Dreamer."
    )
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially. Default only prints them.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Training seeds.")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON, help="Python inside isaacsim-4.2.")
    parser.add_argument("--dreamer-python", type=str, default="python", help="Python used for r2dreamer_isaac.")
    parser.add_argument("--device", type=str, default="cpu", help="Isaac simulation device.")
    parser.add_argument("--env-device", type=str, default="cpu", help="Dreamer Isaac environment device.")
    parser.add_argument("--agent-device", type=str, default="cuda:0", help="Dreamer learner device.")
    parser.add_argument("--run-label", type=str, default="2_800k", help="Suffix used in run names and logdirs.")
    parser.add_argument(
        "--dreamer-run-stamp",
        type=str,
        default=None,
        help="Timestamp/run id prepended to Dreamer logdirs. Defaults to current time.",
    )
    parser.add_argument("--dreamer-total-steps", type=int, default=800_000, help="Dreamer true env steps.")
    parser.add_argument("--rsl-max-iterations", type=int, default=6250, help="RSL-RL PPO iterations.")
    parser.add_argument("--skrl-max-iterations", type=int, default=1563, help="skrl PPO iterations.")
    parser.add_argument(
        "--only",
        choices=[
            "all",
            "state-rsl",
            "state-safe-ppo",
            "state-dreamer",
            "vision-skrl",
            "vision-safe-ppo",
            "vision-dreamer",
        ],
        default="all",
        help="Limit the generated command group.",
    )
    args = parser.parse_args()
    if args.dreamer_run_stamp is None:
        args.dreamer_run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return args


def _repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def _dreamer_run_name(seed: int, args: argparse.Namespace) -> str:
    return f"{args.dreamer_run_stamp}_seed_{seed}_{args.run_label}"


def _rsl_state(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="state-rsl",
        seed=seed,
        argv=[
            str(args.isaac_python),
            _repo_path("scripts", "rsl_rl", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-State-Direct-v0",
            "--num_envs",
            "4",
            "--seed",
            str(seed),
            "--max_iterations",
            str(args.rsl_max_iterations),
            "--run_name",
            f"seed_{seed}_{args.run_label}",
            "--device",
            args.device,
            "--disable_fabric",
            "--enable_cameras",
        ],
    )


def _skrl_vision(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="vision-skrl",
        seed=seed,
        argv=[
            str(args.isaac_python),
            _repo_path("source", "skrl", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
            "--num_envs",
            "4",
            "--seed",
            str(seed),
            "--max_iterations",
            str(args.skrl_max_iterations),
            "--device",
            args.device,
            "--disable_fabric",
            "--enable_cameras",
            f"agent.agent.experiment.experiment_name=seed_{seed}_{args.run_label}",
        ],
    )


def _safe_rsl_state(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="state-safe-ppo",
        seed=seed,
        argv=[
            str(args.isaac_python),
            _repo_path("scripts", "rsl_rl", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-State-Direct-v0",
            "--cfg_entry_point",
            "safe_rsl_rl_cfg_entry_point",
            "--num_envs",
            "4",
            "--seed",
            str(seed),
            "--max_iterations",
            str(args.rsl_max_iterations),
            "--run_name",
            f"seed_{seed}_{args.run_label}",
            "--device",
            args.device,
            "--disable_fabric",
            "--enable_cameras",
        ],
    )


def _safe_skrl_vision(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="vision-safe-ppo",
        seed=seed,
        argv=[
            str(args.isaac_python),
            _repo_path("source", "skrl", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
            "--cfg_entry_point",
            "safe_skrl_cfg_entry_point",
            "--algorithm",
            "SafePPO",
            "--num_envs",
            "4",
            "--seed",
            str(seed),
            "--max_iterations",
            str(args.skrl_max_iterations),
            "--device",
            args.device,
            "--disable_fabric",
            "--enable_cameras",
            f"agent.agent.experiment.experiment_name=seed_{seed}_{args.run_label}",
        ],
    )


def _dreamer_state(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="state-dreamer",
        seed=seed,
        argv=[
            args.dreamer_python,
            _repo_path("source", "r2dreamer_isaac", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-State-Direct-v0",
            "--cfg_entry_point",
            "dreamer_cfg_entry_point",
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--env_device",
            args.env_device,
            "--agent_device",
            args.agent_device,
            "--num_envs",
            "4",
            "--disable_fabric",
            "--enable_cameras",
            "--logdir",
            _repo_path("logs", "r2dreamer", "ur3_blood_pipe_state_dreamer", _dreamer_run_name(seed, args)),
            f"trainer.total_steps={args.dreamer_total_steps}",
        ],
    )


def _dreamer_vision(seed: int, args: argparse.Namespace) -> ExperimentCommand:
    return ExperimentCommand(
        name="vision-dreamer",
        seed=seed,
        argv=[
            args.dreamer_python,
            _repo_path("source", "r2dreamer_isaac", "train.py"),
            "--task",
            "Isaac-Ur3-Blood-Pipe-Vision-Wrist-Direct-v0",
            "--cfg_entry_point",
            "dreamer_cfg_entry_point",
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--env_device",
            args.env_device,
            "--agent_device",
            args.agent_device,
            "--num_envs",
            "1",
            "--disable_fabric",
            "--enable_cameras",
            "--logdir",
            _repo_path(
                "logs",
                "r2dreamer",
                "ur3_blood_pipe_vision_wrist_dreamer",
                _dreamer_run_name(seed, args),
            ),
            f"trainer.total_steps={args.dreamer_total_steps}",
        ],
    )


def _commands(args: argparse.Namespace) -> list[ExperimentCommand]:
    builders = {
        "state-rsl": _rsl_state,
        "state-safe-ppo": _safe_rsl_state,
        "state-dreamer": _dreamer_state,
        "vision-skrl": _skrl_vision,
        "vision-safe-ppo": _safe_skrl_vision,
        "vision-dreamer": _dreamer_vision,
    }
    selected = builders if args.only == "all" else {args.only: builders[args.only]}
    return [builder(seed, args) for name, builder in selected.items() for seed in args.seeds]


def main() -> None:
    args = _parse_args()
    commands = _commands(args)
    for command in commands:
        print(f"\n# {command.name} seed={command.seed}")
        print(command.shell())

    if not args.execute:
        print("\n[INFO] Dry run only. Re-run with --execute to start training sequentially.")
        return

    for command in commands:
        print(f"\n[INFO] Running {command.name} seed={command.seed}")
        subprocess.run(command.argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
