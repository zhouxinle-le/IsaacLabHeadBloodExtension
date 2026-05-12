#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DPI = 300
GRID_COLOR = "#D8D8D8"

COLORS = {
    "PPO": "#D55E00",
    "Dreamer": "#0072B2",
}


@dataclass(frozen=True)
class Curve:
    task_group: str
    algorithm: str
    seed: int
    metric: str
    source_tag: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    title: str
    ylabel: str
    rate: bool
    dreamer_tag: str
    state_ppo_tag: str
    vision_ppo_tag: str
    threshold: float | None = None
    threshold_direction: str = ">="


METRICS = (
    MetricSpec(
        "mean_episode_return",
        "平均回合回报",
        "平均回合回报",
        False,
        "rollout/recent_episode_score_mean",
        "Train/mean_reward",
        "Reward / Total reward (mean)",
        threshold=90.0,
    ),
    MetricSpec(
        "success_rate",
        "成功率",
        "成功率",
        True,
        "rollout/recent_termination_success",
        "Episode_Termination/success",
        "Episode_Termination/success",
        threshold=0.8,
    ),
    MetricSpec(
        "severe_collision_rate",
        "严重碰撞率",
        "严重碰撞率",
        True,
        "rollout/recent_termination_severe_collision",
        "Episode_Termination/severe_collision",
        "Episode_Termination/severe_collision",
    ),
    MetricSpec(
        "absorb_reward",
        "吸取奖励",
        "平均回合奖励",
        False,
        "rollout/recent_reward_absorb_reward",
        "Episode_Reward/absorb_reward",
        "Episode_Reward/absorb_reward",
    ),
)


def _configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Noto Serif CJK SC",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "Times New Roman",
        "DejaVu Serif",
    ]
    font_stack = [font for font in preferred_fonts if font in available_fonts] or ["DejaVu Serif"]
    plt.rcParams.update(
        {
            "font.family": font_stack,
            "axes.unicode_minus": False,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-seed PPO vs Dreamer paper curves.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Training seeds.")
    parser.add_argument("--xmax", type=float, default=800_000.0, help="Real environment step horizon.")
    parser.add_argument("--run-label", type=str, default="800k", help="Suffix used to discover training runs.")
    parser.add_argument("--points", type=int, default=501, help="Interpolation points.")
    parser.add_argument("--smooth-points", type=int, default=5, help="Trailing smoothing window in logged points.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/comparisons/paper_experiments_800k"),
        help="Output directory for figures and CSV files.",
    )
    return parser.parse_args()


def _format_steps(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _style_axis(ax, spec: MetricSpec) -> None:
    ax.set_title(spec.title, pad=8, weight="normal")
    ax.set_xlabel("真实环境交互步数", labelpad=6)
    ax.set_ylabel(spec.ylabel, labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, axis="both", color=GRID_COLOR, linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=3.5, width=0.9, top=False, right=False)
    if spec.rate:
        ax.set_ylim(-0.03, 1.03)


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


def _find_latest(pattern: str) -> Path | None:
    matches = [path for path in REPO_ROOT.glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _read_dreamer(run_dir: Path, task_group: str, seed: int, spec: MetricSpec) -> Curve:
    metrics_path = run_dir / "metrics.jsonl"
    xs: list[float] = []
    ys: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if spec.dreamer_tag in row:
                xs.append(float(row["step"]))
                ys.append(float(row[spec.dreamer_tag]))
    if not xs:
        raise KeyError(f"Dreamer tag not found in {metrics_path}: {spec.dreamer_tag}")
    return Curve(task_group, "Dreamer", seed, spec.metric, spec.dreamer_tag, np.asarray(xs), np.asarray(ys))


def _read_tensorboard(
    run_dir: Path,
    task_group: str,
    algorithm: str,
    seed: int,
    tag: str,
    metric: str,
    step_scale: float,
) -> Curve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required. Run with the isaacsim environment, for example: "
            "conda run -n isaacsim-4.2 python scripts/scripts/plot_paper_experiments.py"
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    if tag not in tags:
        raise KeyError(f"TensorBoard tag not found in {run_dir}: {tag}")
    events = accumulator.Scalars(tag)
    x = np.asarray([event.step * step_scale for event in events], dtype=np.float64)
    y = np.asarray([event.value for event in events], dtype=np.float64)
    return Curve(task_group, algorithm, seed, metric, tag, x, y)


def _discover_curves(seeds: list[int]) -> list[Curve]:
    curves: list[Curve] = []
    for seed in seeds:
        state_rsl = _find_latest(f"logs/rsl_rl/ur3_blood_pipe_state_direct/*seed_{seed}_{args.run_label}")
        state_dreamer = _find_latest(f"logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_{seed}_{args.run_label}")
        vision_skrl = _find_latest(
            f"logs/skrl/ur3_blood_pipe_vision_direct_wrist/*seed_{seed}_{args.run_label}"
        )
        vision_dreamer = _find_latest(
            f"logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/seed_{seed}_{args.run_label}"
        )

        if state_rsl is None:
            print(f"[WARN] Missing state PPO run for seed={seed}")
        if state_dreamer is None:
            print(f"[WARN] Missing state Dreamer run for seed={seed}")
        if vision_skrl is None:
            print(f"[WARN] Missing vision PPO run for seed={seed}")
        if vision_dreamer is None:
            print(f"[WARN] Missing vision Dreamer run for seed={seed}")

        for spec in METRICS:
            if state_rsl is not None:
                curves.append(
                    _read_tensorboard(
                        state_rsl,
                        "state",
                        "PPO",
                        seed,
                        spec.state_ppo_tag,
                        spec.metric,
                        step_scale=4 * 32,
                    )
                )
            if state_dreamer is not None:
                curves.append(_read_dreamer(state_dreamer, "state", seed, spec))
            if vision_skrl is not None:
                curves.append(
                    _read_tensorboard(
                        vision_skrl,
                        "vision",
                        "PPO",
                        seed,
                        spec.vision_ppo_tag,
                        spec.metric,
                        step_scale=4,
                    )
                )
            if vision_dreamer is not None:
                curves.append(_read_dreamer(vision_dreamer, "vision", seed, spec))
    return curves


def _filtered(curve: Curve, smooth_points: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(curve.x) & np.isfinite(curve.y)
    x = curve.x[mask]
    y = curve.y[mask]
    order = np.argsort(x)
    x = x[order]
    y = _smooth(y[order], smooth_points)
    unique_x, unique_indices = np.unique(x, return_index=True)
    return unique_x, y[unique_indices]


def _interpolate(curve: Curve, grid: np.ndarray, smooth_points: int) -> np.ndarray:
    x, y = _filtered(curve, smooth_points)
    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, x, y)


def _curves_for(curves: list[Curve], task_group: str, algorithm: str, metric: str) -> list[Curve]:
    return [
        curve
        for curve in curves
        if curve.task_group == task_group and curve.algorithm == algorithm and curve.metric == metric
    ]


def _plot_task(curves: list[Curve], task_group: str, output_dir: Path, args: argparse.Namespace) -> None:
    grid = np.linspace(0.0, float(args.xmax), int(args.points))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=FIGURE_DPI, sharex=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, panel_label, spec in zip(axes.ravel(), panel_labels, METRICS):
        spec = MetricSpec(
            spec.metric,
            f"{panel_label} {spec.title}",
            spec.ylabel,
            spec.rate,
            spec.dreamer_tag,
            spec.state_ppo_tag,
            spec.vision_ppo_tag,
            spec.threshold,
            spec.threshold_direction,
        )
        for algorithm in ("Dreamer", "PPO"):
            run_curves = _curves_for(curves, task_group, algorithm, spec.metric)
            if not run_curves:
                continue
            samples = np.vstack([_interpolate(curve, grid, args.smooth_points) for curve in run_curves])
            mean = np.nanmean(samples, axis=0)
            std = np.nanstd(samples, axis=0)
            ax.plot(grid, mean, color=COLORS[algorithm], linewidth=1.8, label=f"{algorithm} (n={len(run_curves)})")
            if len(run_curves) > 1:
                ax.fill_between(grid, mean - std, mean + std, color=COLORS[algorithm], alpha=0.18, linewidth=0)
        _style_axis(ax, spec)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=True,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        facecolor="white",
        edgecolor=GRID_COLOR,
        framealpha=1.0,
        fancybox=False,
        fontsize=9,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = output_dir / f"{task_group}_main_comparison"
    fig.savefig(output_path.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _first_crossing(curve: Curve, threshold: float, direction: str, smooth_points: int, xmax: float) -> float:
    x, y = _filtered(curve, smooth_points)
    mask = x <= xmax
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return math.nan
    if direction == ">=":
        indices = np.flatnonzero(y >= threshold)
    else:
        indices = np.flatnonzero(y <= threshold)
    if indices.size == 0:
        return math.nan
    return float(x[int(indices[0])])


def _write_curve_data(curves: list[Curve], output_dir: Path, args: argparse.Namespace) -> None:
    grid = np.linspace(0.0, float(args.xmax), int(args.points))
    output_path = output_dir / "aggregated_curve_data.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_group", "algorithm", "metric", "real_env_step", "mean", "std", "num_seeds"])
        for task_group in ("state", "vision"):
            for algorithm in ("Dreamer", "PPO"):
                for spec in METRICS:
                    run_curves = _curves_for(curves, task_group, algorithm, spec.metric)
                    if not run_curves:
                        continue
                    samples = np.vstack([_interpolate(curve, grid, args.smooth_points) for curve in run_curves])
                    mean = np.nanmean(samples, axis=0)
                    std = np.nanstd(samples, axis=0)
                    for x_value, mean_value, std_value in zip(grid, mean, std):
                        writer.writerow(
                            [
                                task_group,
                                algorithm,
                                spec.metric,
                                f"{x_value:.6g}",
                                f"{mean_value:.8g}",
                                f"{std_value:.8g}",
                                len(run_curves),
                            ]
                        )


def _write_sample_efficiency(curves: list[Curve], output_dir: Path, args: argparse.Namespace) -> None:
    grid = np.linspace(0.0, float(args.xmax), int(args.points))
    output_path = output_dir / "sample_efficiency_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_group",
                "algorithm",
                "metric",
                "num_seeds",
                "auc_mean",
                "auc_std",
                "value_at_horizon_mean",
                "value_at_horizon_std",
                "first_threshold_step_mean",
                "first_threshold_step_std",
                "threshold",
            ]
        )
        for task_group in ("state", "vision"):
            for algorithm in ("Dreamer", "PPO"):
                for spec in METRICS:
                    run_curves = _curves_for(curves, task_group, algorithm, spec.metric)
                    if not run_curves:
                        continue
                    samples = np.vstack([_interpolate(curve, grid, args.smooth_points) for curve in run_curves])
                    auc_values = np.trapz(samples, grid, axis=1) / float(args.xmax)
                    horizon_values = samples[:, -1]
                    crossing_values: list[float] = []
                    if spec.threshold is not None:
                        crossing_values = [
                            _first_crossing(
                                curve,
                                threshold=spec.threshold,
                                direction=spec.threshold_direction,
                                smooth_points=args.smooth_points,
                                xmax=float(args.xmax),
                            )
                            for curve in run_curves
                        ]
                    finite_crossings = np.asarray(crossing_values, dtype=np.float64)
                    finite_crossings = finite_crossings[np.isfinite(finite_crossings)]
                    writer.writerow(
                        [
                            task_group,
                            algorithm,
                            spec.metric,
                            len(run_curves),
                            f"{np.nanmean(auc_values):.8g}",
                            f"{np.nanstd(auc_values):.8g}",
                            f"{np.nanmean(horizon_values):.8g}",
                            f"{np.nanstd(horizon_values):.8g}",
                            f"{np.nanmean(finite_crossings):.8g}" if finite_crossings.size else "",
                            f"{np.nanstd(finite_crossings):.8g}" if finite_crossings.size else "",
                            "" if spec.threshold is None else f"{spec.threshold_direction}{spec.threshold:g}",
                        ]
                    )


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = _discover_curves(args.seeds)
    if not curves:
        raise RuntimeError("No paper experiment runs were found. Run training first.")
    _plot_task(curves, "state", output_dir, args)
    _plot_task(curves, "vision", output_dir, args)
    _write_curve_data(curves, output_dir, args)
    _write_sample_efficiency(curves, output_dir, args)
    print(f"[INFO] Wrote paper figures and tables to: {output_dir}")


if __name__ == "__main__":
    main()
