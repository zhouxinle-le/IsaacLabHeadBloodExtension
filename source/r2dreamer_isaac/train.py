from __future__ import annotations

import argparse
import atexit
import copy
import os
import pathlib
import sys
from datetime import datetime

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train an Isaac task with vendored R2-Dreamer.")
parser.add_argument("--task", type=str, default="Isaac-Ur3-Blood-Pipe-State-Direct-v0", help="Gym task name.")
parser.add_argument(
    "--cfg_entry_point",
    type=str,
    default="dreamer_cfg_entry_point",
    help="Registry key for the Dreamer-family config to load.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of training environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--logdir", type=str, default=None, help="Explicit log directory.")
parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file or its parent run directory.")
parser.add_argument(
    "--agent_device",
    type=str,
    default=None,
    help="Device for the Dreamer learner. Defaults to --device when omitted.",
)
parser.add_argument(
    "--env_device",
    type=str,
    default=None,
    help="Device for the Isaac environment. Defaults to --device when omitted.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, overrides = parser.parse_known_args()
if "vision" in str(args_cli.task).lower():
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml

import head_blood_absorption.tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from r2dreamer_isaac.config import build_runtime_config, to_plain_dict
from r2dreamer_isaac.env_adapter import IsaacR2DreamerEnvAdapter
from r2dreamer_isaac.replay_buffer import IsaacReplayBuffer
from r2dreamer_isaac.trainer import IsaacOnlineTrainer
from r2dreamer_isaac.vendor.r2dreamer import Dreamer, tools


def _resolve_resume_path(resume_arg: str) -> pathlib.Path:
    resume_path = pathlib.Path(resume_arg).expanduser().resolve()
    if resume_path.is_dir():
        candidate = resume_path / "checkpoints" / "latest.pt"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Could not find latest checkpoint under: {candidate}")
    return resume_path


def _build_logdir(config, resume_path: pathlib.Path | None) -> pathlib.Path:
    if resume_path is not None:
        return resume_path.parent.parent
    if config.logdir:
        return pathlib.Path(config.logdir).expanduser().resolve()
    root = pathlib.Path("logs") / "r2dreamer" / str(config.experiment_name)
    return root.resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _dump_config(logdir: pathlib.Path, config, env_cfg) -> None:
    params_dir = logdir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    with (params_dir / "r2dreamer.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_dict(config), handle, sort_keys=False)
    with (params_dir / "env.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(env_cfg))


def _apply_env_cfg_overrides(env_cfg, overrides) -> None:
    if not overrides:
        return
    for name, value in overrides.items():
        if not hasattr(env_cfg, name):
            raise AttributeError(f"Environment config has no field '{name}' for env.cfg_overrides.")
        setattr(env_cfg, name, value)


def main() -> None:
    task_cfg = load_cfg_from_registry(args_cli.task, args_cli.cfg_entry_point)
    env_device = args_cli.env_device or args_cli.device
    agent_device = args_cli.agent_device or args_cli.device
    cli_updates = {
        "device": agent_device,
        "agent_device": agent_device,
        "env_device": env_device,
        "seed": task_cfg.get("seed", 0) if args_cli.seed is None else args_cli.seed,
    }
    if args_cli.num_envs is not None:
        cli_updates.setdefault("env", {})["num_envs"] = args_cli.num_envs
    if args_cli.logdir is not None:
        cli_updates["logdir"] = args_cli.logdir

    config = build_runtime_config(task_cfg=task_cfg, cli_updates=cli_updates, dotlist_overrides=overrides)

    tools.set_seed_everywhere(int(config.seed))
    if bool(config.deterministic_run):
        tools.enable_deterministic_run()

    resume_path = _resolve_resume_path(args_cli.resume) if args_cli.resume else None
    logdir = _build_logdir(config, resume_path)
    logdir.mkdir(parents=True, exist_ok=True)
    console_file = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_file.close())
    print(f"[INFO] Logging experiment in directory: {logdir}")
    print(f"[INFO] Devices: env={config.env_device}, agent={config.agent_device}")

    train_env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(config.env_device),
        num_envs=int(config.env.num_envs),
        use_fabric=not args_cli.disable_fabric,
    )
    _apply_env_cfg_overrides(train_env_cfg, getattr(config.env, "cfg_overrides", {}))
    _dump_config(logdir, config, train_env_cfg)

    train_env = gym.make(args_cli.task, cfg=train_env_cfg)
    train_adapter = IsaacR2DreamerEnvAdapter(train_env)
    eval_adapter = None
    if int(config.trainer.eval_episode_num) > 0:
        print(
            "[WARN] Inline evaluation is disabled for this IsaacLab trainer because "
            "DirectRLEnv only allows one SimulationContext per process."
        )

    agent = Dreamer(config.model, train_adapter.observation_space, train_adapter.action_space).to(config.agent_device)
    replay_buffer = IsaacReplayBuffer(config.buffer)
    logger = tools.Logger(logdir)
    trainer = IsaacOnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_adapter,
        eval_adapter,
        run_config=to_plain_dict(config),
    )

    checkpoint = None
    if resume_path is not None:
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

    trainer.begin(agent, checkpoint=checkpoint)

    train_adapter.close()
    if eval_adapter is not None:
        eval_adapter.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
