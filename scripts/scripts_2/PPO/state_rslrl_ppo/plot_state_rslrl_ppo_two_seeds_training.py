#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = (
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-08_21-19-37_seed_0_800k"),
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-09_09-49-21_seed_1_800k"),
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "two_seed_results"
FIGURE_DPI = 300
GRID_COLOR = "#E3E3E3"
LINE_COLOR = "#26A7E1"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    tag: str
    rate: bool = False


@dataclass(frozen=True)
class RawCurve:
    run_index: int
    tag: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class AggregateCurve:
    key: str
    tag: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray
    raw_values: np.ndarray


METRICS = (
    MetricSpec(
        key="train_mean_reward",
        title="训练过程累积奖励曲线",
        ylabel="累积奖励",
        tag="Train/mean_reward",
    ),
    MetricSpec(
        key="episode_termination_success",
        title="训练过程成功率曲线",
        ylabel="成功率",
        tag="Episode_Termination/success",
        rate=True,
    ),
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two-seed State RSL-RL PPO training curves."
    )
    parser.add_argument("--runs", type=Path, nargs="+", default=list(DEFAULT_RUNS), help="RSL-RL run directories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--points", type=int, default=501, help="Interpolation points on the shared x-axis.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="TensorBoard-like exponential smoothing factor after interpolation. 0 disables smoothing.",
    )
    parser.add_argument("--color", type=str, default=LINE_COLOR, help=f"Line color. Default: {LINE_COLOR}.")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _format_steps(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _smooth_tensorboard_like(values: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = float(np.clip(smoothing, 0.0, 0.999))
    if values.size == 0 or smoothing <= 0.0:
        return values.copy()
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = smoothing * smoothed[index - 1] + (1.0 - smoothing) * values[index]
    return smoothed


def _read_curve(run_dir: Path, run_index: int, spec: MetricSpec) -> RawCurve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required. Run with: "
            "conda run -n isaacsim-4.2 python scripts/scripts_2/PPO/state_rslrl_ppo/"
            "plot_state_rslrl_ppo_two_seeds_training.py"
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    if spec.tag not in tags:
        available = ", ".join(sorted(tags))
        raise KeyError(f"Missing TensorBoard tag {spec.tag!r} in {run_dir}. Available tags: {available}")

    events = accumulator.Scalars(spec.tag)
    if not events:
        raise RuntimeError(f"TensorBoard tag has no scalar events: {spec.tag} in {run_dir}")
    x = np.asarray([event.step for event in events], dtype=np.float64)
    y = np.asarray([event.value for event in events], dtype=np.float64)
    order = np.argsort(x)
    return RawCurve(run_index, spec.tag, x[order], y[order])


def _interpolate(curve: RawCurve, grid: np.ndarray) -> np.ndarray:
    unique_x, unique_indices = np.unique(curve.x, return_index=True)
    unique_y = curve.y[unique_indices]
    return np.interp(grid, unique_x, unique_y)


def _aggregate(curves: list[RawCurve], spec: MetricSpec, points: int, smoothing: float) -> AggregateCurve:
    xmax = min(float(curve.x.max()) for curve in curves)
    grid = np.linspace(0.0, xmax, points)
    raw_values = np.vstack([_interpolate(curve, grid) for curve in curves])
    values = np.vstack([_smooth_tensorboard_like(seed_values, smoothing) for seed_values in raw_values])
    return AggregateCurve(
        key=spec.key,
        tag=spec.tag,
        x=grid,
        mean=np.mean(values, axis=0),
        std=np.std(values, axis=0),
        values=values,
        raw_values=raw_values,
    )


def _style_axis(ax, spec: MetricSpec) -> None:
    ax.set_title(spec.title, pad=8, weight="normal")
    ax.set_xlabel("PPO迭代次数 (iteration)", labelpad=6)
    ax.set_ylabel(spec.ylabel, labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#222222", top=False, right=False)
    if spec.rate:
        ax.set_ylim(-0.03, 1.03)


def _plot_curve(curve: AggregateCurve, spec: MetricSpec, output_dir: Path, color: str) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2), dpi=FIGURE_DPI)
    ax.fill_between(
        curve.x,
        curve.mean - curve.std,
        curve.mean + curve.std,
        color=color,
        alpha=0.08,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(curve.x, curve.mean, color=color, linewidth=1.2, zorder=3)
    _style_axis(ax, spec)
    fig.tight_layout()

    output_base = output_dir / spec.key
    fig.savefig(output_base.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)
    return output_base.with_suffix(".png")


def _write_csv(aggregates: list[AggregateCurve], output_path: Path) -> None:
    max_runs = max(curve.values.shape[0] for curve in aggregates)
    fieldnames = ["metric", "tag", "iteration", "mean", "std"]
    fieldnames.extend(f"seed_{index}" for index in range(max_runs))
    fieldnames.extend(f"raw_seed_{index}" for index in range(max_runs))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in aggregates:
            for index, x in enumerate(curve.x):
                row: dict[str, object] = {
                    "metric": curve.key,
                    "tag": curve.tag,
                    "iteration": float(x),
                    "mean": float(curve.mean[index]),
                    "std": float(curve.std[index]),
                }
                for run_index in range(curve.values.shape[0]):
                    row[f"seed_{run_index}"] = float(curve.values[run_index, index])
                    row[f"raw_seed_{run_index}"] = float(curve.raw_values[run_index, index])
                writer.writerow(row)


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    run_dirs = [_resolve(path) for path in args.runs]
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[AggregateCurve] = []
    for spec in METRICS:
        curves = [_read_curve(run_dir, index, spec) for index, run_dir in enumerate(run_dirs)]
        aggregate = _aggregate(curves, spec, args.points, args.smoothing)
        aggregates.append(aggregate)
        print(
            f"[INFO] {spec.key}: tag={spec.tag}, "
            f"x=[{aggregate.x.min():.0f}, {aggregate.x.max():.0f}], runs={len(curves)}"
        )

    saved = [_plot_curve(curve, spec, output_dir, args.color) for curve, spec in zip(aggregates, METRICS)]
    csv_path = output_dir / "state_rslrl_ppo_two_seed_curves.csv"
    _write_csv(aggregates, csv_path)

    print(f"[INFO] Runs: {', '.join(str(path) for path in run_dirs)}")
    print(f"[INFO] Line color: {args.color}")
    for path in saved:
        print(f"[INFO] Saved: {path}")
        print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
