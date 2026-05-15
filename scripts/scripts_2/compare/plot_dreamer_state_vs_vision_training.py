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


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DPI = 300
GRID_COLOR = "#E3E3E3"

DEFAULT_STATE_RUNS = (
    Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_0_800k"),
    Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_1_800k"),
    Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/2026-05-14_02-01-49_seed_1_600k"),
)
DEFAULT_VISION_RUNS = (
    Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_0_800k"),
    Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-06_09-29-04"),
)

GROUP_LABELS = {
    "state": "State Dreamer",
    "vision": "Vision Wrist Dreamer",
}
GROUP_COLORS = {
    "state": "#E95412",
    "vision": "#E274A9",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    tags: tuple[str, ...]
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
        title="平均回合回报",
        ylabel="平均回合回报",
        tags=("rollout/interval_episode_score_mean", "rollout/recent_episode_score_mean"),
    ),
    MetricSpec(
        key="success_rate",
        title="成功率",
        ylabel="成功率",
        tags=("rollout/recent_termination_success",),
        rate=True,
    ),
    # MetricSpec(
    #     key="absorbed_ratio",
    #     title="吸取比例",
    #     ylabel="吸取比例",
    #     tags=("Metrics/absorbed_ratio_mean",),
    #     rate=True,
    # ),
    # MetricSpec(
    #     key="severe_collision_rate",
    #     title="严重碰撞率",
    #     ylabel="严重碰撞率",
    #     tags=("rollout/recent_termination_severe_collision",),
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
        description="Plot averaged Dreamer training curves for state observations vs wrist-camera observations."
    )
    parser.add_argument(
        "--state-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_STATE_RUNS),
        help="Dreamer state-observation run directories.",
    )
    parser.add_argument(
        "--vision-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_VISION_RUNS),
        help="Dreamer wrist-vision run directories.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=500_000.0,
        help="Real environment step horizon. Default is 500k to match the vision Dreamer horizon.",
    )
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
        default=Path("scripts/scripts_2/compare/dreamer_state_vs_vision_500k"),
        help="Output directory for figures and CSV files.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_dreamer_curve(run_dir: Path, group: str, run_index: int, spec: MetricSpec) -> RawCurve:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Dreamer metrics file not found: {metrics_path}")

    xs: list[float] = []
    ys: list[float] = []
    selected_tag: str | None = None
    available_tags: set[str] = set()
    for raw_line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        available_tags.update(row.keys())
        if "step" not in row:
            continue
        if selected_tag is None:
            selected_tag = next((candidate for candidate in spec.tags if candidate in row), None)
        if selected_tag is not None and selected_tag in row:
            xs.append(float(row["step"]))
            ys.append(float(row[selected_tag]))

    if selected_tag is None:
        available = ", ".join(sorted(available_tags))
        raise KeyError(
            f"None of the expected Dreamer tags were found for {GROUP_LABELS[group]} metric '{spec.key}' "
            f"in {metrics_path}. Expected one of: {spec.tags}. Available tags: {available}"
        )
    if not xs:
        raise RuntimeError(f"Dreamer tag has no scalar values: {selected_tag} in {metrics_path}")

    return RawCurve(
        group,
        run_index,
        run_dir,
        spec.key,
        selected_tag,
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    )


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
            _load_dreamer_curve(run_dir, "state", index, spec)
            for index, run_dir in enumerate(state_runs)
        ]
        vision_curves = [
            _load_dreamer_curve(run_dir, "vision", index, spec)
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
    if spec.rate:
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
    output_base = output_dir / "dreamer_state_vs_vision_training"
    figure_paths = _plot(aggregates, output_base)
    csv_path = output_dir / "dreamer_state_vs_vision_training_curves.csv"
    _write_csv(aggregates, csv_path)

    for path in figure_paths:
        print(f"[INFO] Saved: {path}")
        print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
