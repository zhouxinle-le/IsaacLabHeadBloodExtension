#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DPI = 300
GRID_COLOR = "#E3E3E3"

DEFAULT_STATE_RUNS = (
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-08_21-19-37_seed_0_800k"),
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-09_09-49-21_seed_1_800k"),
)
DEFAULT_VISION_RUNS = (
    Path("logs/skrl/ur3_blood_pipe_vision_direct_wrist/2026-05-10_21-23-12_ppo_torch_seed_0_800k"),
    Path("logs/skrl/ur3_blood_pipe_vision_direct_wrist/2026-05-11_05-10-37_ppo_torch_seed_1_800k"),
)

GROUP_LABELS = {
    "state": "State PPO (RSL-RL)",
    "vision": "Vision Wrist PPO (skrl)",
}
GROUP_COLORS = {
    "state": "#26A7E1",
    "vision": "#13AF68",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    state_tags: tuple[str, ...]
    vision_tags: tuple[str, ...]
    rate: bool = False


@dataclass(frozen=True)
class RawCurve:
    group: str
    run_index: int
    run_dir: Path
    metric: str
    tag: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class AggregateCurve:
    group: str
    metric: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray
    raw_values: np.ndarray


METRICS = (
    MetricSpec(
        key="mean_episode_return",
        title="训练过程累积奖励曲线",
        ylabel="累积奖励",
        state_tags=("Train/mean_reward",),
        vision_tags=("Reward / Total reward (mean)",),
    ),
    MetricSpec(
        key="success_rate",
        title="训练过程成功率曲线",
        ylabel="成功率",
        state_tags=("Episode_Termination/success", "Metrics/success_rate"),
        vision_tags=("Episode_Termination/success", "Metrics/success_rate"),
        rate=True,
    ),
    # MetricSpec(
    #     key="absorbed_ratio",
    #     title="吸取比例",
    #     ylabel="吸取比例",
    #     state_tags=("Metrics/absorbed_ratio_mean",),
    #     vision_tags=("Metrics/absorbed_ratio_mean",),
    #     rate=True,
    # ),
    # MetricSpec(
    #     key="severe_collision_rate",
    #     title="严重碰撞率",
    #     ylabel="严重碰撞率",
    #     state_tags=("Episode_Termination/severe_collision",),
    #     vision_tags=("Episode_Termination/severe_collision",),
    #     rate=True,
    # ),
)


def _configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "微软雅黑",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
        "DejaVu Serif",
    ]
    font_stack = [font for font in preferred_fonts if font in available_fonts] or ["DejaVu Serif"]
    plt.rcParams.update(
        {
            "font.family": font_stack,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
            "axes.labelsize": 7.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged PPO training curves for state observations vs wrist-camera observations."
    )
    parser.add_argument(
        "--state-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_STATE_RUNS),
        help="RSL-RL state-observation PPO run directories.",
    )
    parser.add_argument(
        "--vision-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_VISION_RUNS),
        help="skrl wrist-vision PPO run directories.",
    )
    parser.add_argument("--xmax", type=float, default=800_000.0, help="Real environment step horizon.")
    parser.add_argument("--points", type=int, default=501, help="Interpolation points.")
    parser.add_argument(
        "--smooth-points",
        type=int,
        default=21,
        help="Trailing smoothing window after interpolation to the shared real-step grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/scripts_2/compare/state_vs_vision_ppo_800k"),
        help="Output directory for figures and CSV files.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_int_value(raw_value: str) -> int | None:
    match = re.match(r"\s*(-?\d+)\b", raw_value)
    return int(match.group(1)) if match else None


def _read_section_int(path: Path, section: str, key: str) -> int | None:
    if not path.is_file():
        return None

    in_section = False
    section_indent = 0
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped == f"{section}:":
            in_section = True
            section_indent = indent
            continue
        if in_section and indent <= section_indent:
            in_section = False
        if in_section and stripped.startswith(f"{key}:"):
            return _parse_int_value(stripped.split(":", 1)[1])
    return None


def _read_top_level_int(path: Path, key: str) -> int | None:
    if not path.is_file():
        return None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0 and stripped.startswith(f"{key}:"):
            return _parse_int_value(stripped.split(":", 1)[1])
    return None


def _state_step_scale(run_dir: Path) -> float:
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    num_envs = _read_section_int(env_yaml, "scene", "num_envs") or 4
    num_steps_per_env = _read_top_level_int(agent_yaml, "num_steps_per_env") or 32
    return float(num_envs * num_steps_per_env)


def _vision_step_scale(run_dir: Path) -> float:
    env_yaml = run_dir / "params" / "env.yaml"
    num_envs = _read_section_int(env_yaml, "scene", "num_envs") or 4
    return float(num_envs)


def _load_scalar_curve(run_dir: Path, group: str, run_index: int, spec: MetricSpec, step_scale: float) -> RawCurve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required. Run with the Isaac environment, for example: "
            "conda run -n isaacsim-4.2 python scripts/scripts_2/compare/plot_state_vs_vision_ppo_training.py"
        ) from exc

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    tags_to_try = spec.state_tags if group == "state" else spec.vision_tags
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    tag = next((candidate for candidate in tags_to_try if candidate in available_tags), None)
    if tag is None:
        available = ", ".join(sorted(available_tags))
        raise KeyError(
            f"None of the expected tags were found for {GROUP_LABELS[group]} metric '{spec.key}' in {run_dir}. "
            f"Expected one of: {tags_to_try}. Available tags: {available}"
        )

    events = accumulator.Scalars(tag)
    if not events:
        raise RuntimeError(f"TensorBoard tag has no scalar events: {tag} in {run_dir}")

    x = np.asarray([event.step * step_scale for event in events], dtype=np.float64)
    y = np.asarray([event.value for event in events], dtype=np.float64)
    return RawCurve(group, run_index, run_dir, spec.key, tag, x, y)


def _format_steps(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size <= 2:
        return y
    window = min(window, int(y.size))
    out = np.empty_like(y, dtype=np.float64)
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    for index in range(y.size):
        start = max(0, index - window + 1)
        out[index] = (cumsum[index + 1] - cumsum[start]) / (index - start + 1)
    return out


def _interpolate(curve: RawCurve, grid: np.ndarray) -> np.ndarray:
    mask = np.isfinite(curve.x) & np.isfinite(curve.y)
    x = curve.x[mask]
    y = curve.y[mask]
    if x.size == 0:
        raise RuntimeError(f"Curve has no finite values: {curve.run_dir} tag={curve.tag}")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    return np.interp(grid, unique_x, unique_y)


def _aggregate(curves: list[RawCurve], grid: np.ndarray, smooth_points: int) -> AggregateCurve:
    if not curves:
        raise ValueError("Cannot aggregate an empty curve list.")
    raw_values = np.vstack([_interpolate(curve, grid) for curve in curves])
    values = np.vstack([_smooth(seed_values, smooth_points) for seed_values in raw_values])
    return AggregateCurve(
        group=curves[0].group,
        metric=curves[0].metric,
        x=grid,
        mean=np.mean(values, axis=0),
        std=np.std(values, axis=0),
        values=values,
        raw_values=raw_values,
    )


def _read_all_curves(args: argparse.Namespace) -> dict[str, dict[str, AggregateCurve]]:
    state_runs = [_resolve(path) for path in args.state_runs]
    vision_runs = [_resolve(path) for path in args.vision_runs]
    grid = np.linspace(0.0, float(args.xmax), int(args.points))
    aggregates: dict[str, dict[str, AggregateCurve]] = {}

    for spec in METRICS:
        state_curves = [
            _load_scalar_curve(run_dir, "state", index, spec, _state_step_scale(run_dir))
            for index, run_dir in enumerate(state_runs)
        ]
        vision_curves = [
            _load_scalar_curve(run_dir, "vision", index, spec, _vision_step_scale(run_dir))
            for index, run_dir in enumerate(vision_runs)
        ]
        aggregates[spec.key] = {
            "state": _aggregate(state_curves, grid, args.smooth_points),
            "vision": _aggregate(vision_curves, grid, args.smooth_points),
        }

        print(
            f"[INFO] {spec.key}: "
            f"state tag={state_curves[0].tag}, vision tag={vision_curves[0].tag}, "
            f"state x=[{state_curves[0].x.min():.0f}, {state_curves[0].x.max():.0f}], "
            f"vision x=[{vision_curves[0].x.min():.0f}, {vision_curves[0].x.max():.0f}]"
        )

    return aggregates


def _style_axis(ax, spec: MetricSpec) -> None:
    ax.set_title(spec.title, pad=8, weight="normal")
    ax.set_xlabel("真实环境交互步数(transitions)", labelpad=6)
    ax.set_ylabel(spec.ylabel, labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#222222", top=False, right=False)
    if spec.key == "success_rate":
        ax.set_ylim(-0.03, 1.03)
    elif spec.rate:
        ax.set_ylim(-0.03, 1.03)


def _plot(aggregates: dict[str, dict[str, AggregateCurve]], output_path: Path) -> list[Path]:
    saved_paths: list[Path] = []
    # Combined layouts kept for reference:
    # fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=FIGURE_DPI, sharex=True)
    # fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=FIGURE_DPI, sharex=True)
    for spec in METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2), dpi=FIGURE_DPI)
        for group in ("state", "vision"):
            curve = aggregates[spec.key][group]
            ax.fill_between(
                curve.x,
                curve.mean - curve.std,
                curve.mean + curve.std,
                color=GROUP_COLORS[group],
                alpha=0.08,
                linewidth=0.0,
                zorder=2,
            )
            ax.plot(
                curve.x,
                curve.mean,
                color=GROUP_COLORS[group],
                linewidth=1.2,
                label=GROUP_LABELS[group],
                zorder=3,
            )
        _style_axis(ax, spec)
        legend = ax.legend(
            loc="lower right",
            ncol=1,
            frameon=True,
            facecolor="white",
            edgecolor="#DDDDDD",
            framealpha=0.9,
            fancybox=True,
        )
        legend.get_frame().set_linewidth(0.8)
        fig.tight_layout()
        metric_output = output_path.with_name(f"{output_path.name}_{spec.key}")
        fig.savefig(metric_output.with_suffix(".png"), dpi=FIGURE_DPI)
        fig.savefig(metric_output.with_suffix(".pdf"))
        plt.close(fig)
        saved_paths.append(metric_output.with_suffix(".png"))
    return saved_paths


def _write_csv(aggregates: dict[str, dict[str, AggregateCurve]], output_path: Path) -> None:
    max_state_runs = max(curves["state"].values.shape[0] for curves in aggregates.values())
    max_vision_runs = max(curves["vision"].values.shape[0] for curves in aggregates.values())
    fieldnames = ["metric", "env_steps", "state_mean", "state_std", "vision_mean", "vision_std"]
    fieldnames.extend(f"state_seed_{index}" for index in range(max_state_runs))
    fieldnames.extend(f"vision_seed_{index}" for index in range(max_vision_runs))
    fieldnames.extend(f"state_raw_seed_{index}" for index in range(max_state_runs))
    fieldnames.extend(f"vision_raw_seed_{index}" for index in range(max_vision_runs))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in METRICS:
            state = aggregates[spec.key]["state"]
            vision = aggregates[spec.key]["vision"]
            for index, env_steps in enumerate(state.x):
                row: dict[str, object] = {
                    "metric": spec.key,
                    "env_steps": float(env_steps),
                    "state_mean": float(state.mean[index]),
                    "state_std": float(state.std[index]),
                    "vision_mean": float(vision.mean[index]),
                    "vision_std": float(vision.std[index]),
                }
                for run_index in range(state.values.shape[0]):
                    row[f"state_seed_{run_index}"] = float(state.values[run_index, index])
                for run_index in range(vision.values.shape[0]):
                    row[f"vision_seed_{run_index}"] = float(vision.values[run_index, index])
                for run_index in range(state.raw_values.shape[0]):
                    row[f"state_raw_seed_{run_index}"] = float(state.raw_values[run_index, index])
                for run_index in range(vision.raw_values.shape[0]):
                    row[f"vision_raw_seed_{run_index}"] = float(vision.raw_values[run_index, index])
                writer.writerow(row)


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregates = _read_all_curves(args)
    output_base = output_dir / "state_vs_vision_ppo_training"
    figure_paths = _plot(aggregates, output_base)
    csv_path = output_dir / "state_vs_vision_ppo_training_curves.csv"
    _write_csv(aggregates, csv_path)

    for path in figure_paths:
        print(f"[INFO] Saved: {path}")
        print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
