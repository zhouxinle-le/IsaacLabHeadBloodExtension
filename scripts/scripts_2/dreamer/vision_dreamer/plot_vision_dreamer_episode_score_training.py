#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_0_800k"),
    Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-06_09-29-04"),
    Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-12_21-14-29_seed_0_600k"),
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "multi_seed_results"
FIGURE_DPI = 300
GRID_COLOR = "#E3E3E3"
LINE_COLOR = "#E274A9"
TAG = "episode/score"


@dataclass(frozen=True)
class RawCurve:
    run_index: int
    run_dir: Path
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class AggregateCurve:
    tag: str
    run_dirs: tuple[Path, ...]
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray
    raw_values: np.ndarray


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
    parser = argparse.ArgumentParser(description="Plot multi-seed Vision Wrist Dreamer episode score curves.")
    parser.add_argument("--runs", type=Path, nargs="+", default=list(DEFAULT_RUNS), help="Dreamer run directories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--points", type=int, default=501, help="Interpolation points on the shared real environment step axis.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="TensorBoard-like exponential smoothing factor after interpolation. 0 disables smoothing.",
    )
    parser.add_argument(
        "--xmax-steps",
        type=float,
        default=None,
        help="Maximum real environment step on the x-axis. Defaults to the shortest run max step.",
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


def _read_curve(run_dir: Path, run_index: int) -> RawCurve:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Dreamer metrics file not found: {metrics_path}")

    xs: list[float] = []
    values: list[float] = []
    available_keys: set[str] = set()
    for raw_line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        available_keys.update(row.keys())
        if TAG in row:
            if "step" not in row:
                raise KeyError(f"Dreamer metric {TAG!r} has no 'step' field in {metrics_path}")
            xs.append(float(row["step"]))
            values.append(float(row[TAG]))

    if not values:
        available = ", ".join(sorted(available_keys))
        raise KeyError(f"Missing Dreamer metric {TAG!r} in {metrics_path}. Available keys: {available}")

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    order = np.argsort(x)
    return RawCurve(run_index=run_index, run_dir=run_dir, x=x[order], y=y[order])


def _interpolate(curve: RawCurve, grid: np.ndarray) -> np.ndarray:
    unique_x, unique_indices = np.unique(curve.x, return_index=True)
    unique_y = curve.y[unique_indices]
    return np.interp(grid, unique_x, unique_y)


def _aggregate(curves: list[RawCurve], points: int, smoothing: float, xmax_steps: float | None) -> AggregateCurve:
    if not curves:
        raise ValueError("At least one run is required.")
    common_start = max(float(curve.x.min()) for curve in curves)
    shortest_run_steps = min(float(curve.x.max()) for curve in curves)
    xmax = shortest_run_steps if xmax_steps is None else float(xmax_steps)
    if xmax <= 0:
        raise ValueError("--xmax-steps must be positive.")
    if xmax > shortest_run_steps:
        raise ValueError(
            f"--xmax-steps={xmax:g} exceeds the shortest run max step ({shortest_run_steps:g}). "
            "Use a smaller value so every run contributes to the whole mean curve."
        )
    if xmax < common_start:
        raise ValueError(
            f"--xmax-steps={xmax:g} is smaller than the first common logged step ({common_start:g})."
        )
    grid = np.linspace(common_start, xmax, points)
    raw_values = np.vstack([_interpolate(curve, grid) for curve in curves])
    values = np.vstack([_smooth_tensorboard_like(seed_values, smoothing) for seed_values in raw_values])
    return AggregateCurve(
        tag=TAG,
        run_dirs=tuple(curve.run_dir for curve in curves),
        x=grid,
        mean=np.mean(values, axis=0),
        std=np.std(values, axis=0),
        values=values,
        raw_values=raw_values,
    )


def _style_axis(ax) -> None:
    ax.set_title("Dreamer 训练过程累积奖励曲线", pad=8, weight="normal")
    ax.set_xlabel("真实环境交互步数 (step)", labelpad=6)
    ax.set_ylabel("平均累积奖励", labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, axis="both", color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#222222", top=False, right=False)


def _plot_curve(curve: AggregateCurve, output_dir: Path, color: str) -> Path:
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
    _style_axis(ax)
    fig.tight_layout()

    output_base = output_dir / "episode_score"
    fig.savefig(output_base.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)
    return output_base.with_suffix(".png")


def _write_csv(curve: AggregateCurve, output_path: Path) -> None:
    max_runs = curve.values.shape[0]
    fieldnames = ["tag", "env_steps", "mean", "std", "run_dir"]
    fieldnames.extend(f"seed_{index}" for index in range(max_runs))
    fieldnames.extend(f"raw_seed_{index}" for index in range(max_runs))
    fieldnames.extend(f"run_dir_{index}" for index in range(max_runs))
    run_dir_joined = ";".join(str(path) for path in curve.run_dirs)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, x in enumerate(curve.x):
            row: dict[str, object] = {
                "tag": curve.tag,
                "env_steps": float(x),
                "mean": float(curve.mean[index]),
                "std": float(curve.std[index]),
                "run_dir": run_dir_joined,
            }
            for run_index in range(max_runs):
                row[f"seed_{run_index}"] = float(curve.values[run_index, index])
                row[f"raw_seed_{run_index}"] = float(curve.raw_values[run_index, index])
                row[f"run_dir_{run_index}"] = str(curve.run_dirs[run_index])
            writer.writerow(row)


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    run_dirs = [_resolve(path) for path in args.runs]
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = [_read_curve(run_dir, index) for index, run_dir in enumerate(run_dirs)]
    aggregate = _aggregate(curves, args.points, args.smoothing, args.xmax_steps)
    png_path = _plot_curve(aggregate, output_dir, args.color)
    csv_path = output_dir / "vision_dreamer_episode_score_curves.csv"
    _write_csv(aggregate, csv_path)

    print(
        f"[INFO] {TAG}: x=[{aggregate.x.min():.0f}, {aggregate.x.max():.0f}], "
        f"runs={len(curves)}, shortest_run_steps={min(int(curve.x.max()) for curve in curves)}"
    )
    print(f"[INFO] Runs: {', '.join(str(path) for path in run_dirs)}")
    print(f"[INFO] Line color: {args.color}")
    print(f"[INFO] Saved: {png_path}")
    print(f"[INFO] Saved: {png_path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
