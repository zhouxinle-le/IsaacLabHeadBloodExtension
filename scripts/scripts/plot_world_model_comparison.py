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
import numpy as np
from matplotlib.ticker import FuncFormatter


FIGURE_DPI = 300
GRID_COLOR = "#D8D8D8"
DEFAULT_DREAMER_RUN = Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-06_09-29-04")
DEFAULT_PPO_RUN = Path("logs/skrl/ur3_blood_pipe_vision_direct_wrist/2026-05-03_11-29-48_ppo_torch")

COLORS = {
    "Dreamer": "#0072B2",
    "PPO": "#D55E00",
}
DISPLAY_NAMES = {
    "Dreamer": "Dreamer v3",
    "PPO": "PPO",
}


def _configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Noto Serif CJK SC",
        "Noto Serif CJK JP",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
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


_configure_plot_style()


@dataclass(frozen=True)
class Curve:
    algorithm: str
    metric: str
    source_tag: str
    x: np.ndarray
    y: np.ndarray


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_dreamer_curve(run_dir: Path, tag: str, metric: str, step_scale: float = 1.0) -> Curve:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Dreamer metrics file not found: {metrics_path}")

    xs: list[float] = []
    ys: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if tag not in row:
                continue
            xs.append(float(row["step"]) * step_scale)
            ys.append(float(row[tag]))

    if not xs:
        raise KeyError(f"Dreamer tag not found in {metrics_path}: {tag}")
    return Curve("Dreamer", metric, tag, np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))


def _read_ppo_curve(run_dir: Path, tag: str, metric: str, step_scale: float) -> Curve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required to read skrl event files. Run this script in the Isaac/skrl environment, "
            "for example: conda run -n isaacsim-4.2 python scripts/scripts/plot_world_model_comparison.py"
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if tag not in scalar_tags:
        raise KeyError(f"PPO tag not found in TensorBoard event file: {tag}")

    events = accumulator.Scalars(tag)
    xs = np.asarray([event.step * step_scale for event in events], dtype=np.float64)
    ys = np.asarray([event.value for event in events], dtype=np.float64)
    return Curve("PPO", metric, tag, xs, ys)


def _infer_ppo_num_envs(ppo_run: Path) -> int:
    env_yaml = ppo_run / "params" / "env.yaml"
    if not env_yaml.is_file():
        return 1

    in_scene = False
    for raw_line in env_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw_line.strip()
        if stripped == "scene:":
            in_scene = True
            continue
        if in_scene and raw_line and not raw_line.startswith(" ") and not raw_line.startswith("\t"):
            break
        if in_scene and stripped.startswith("num_envs:"):
            return int(stripped.split(":", 1)[1].strip())
    return 1


def _filter_curve(curve: Curve, xmax: float | None) -> Curve:
    mask = np.isfinite(curve.x) & np.isfinite(curve.y)
    if xmax is not None:
        mask &= curve.x <= xmax
    x = curve.x[mask]
    y = curve.y[mask]
    order = np.argsort(x)
    return Curve(curve.algorithm, curve.metric, curve.source_tag, x[order], y[order])


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


def _format_steps(value: float, _pos: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _style_axis(ax, ylabel: str, title: str | None = None, rate: bool = False) -> None:
    ax.set_xlabel("真实环境交互步数", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    if title:
        ax.set_title(title, pad=8, weight="normal")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color=GRID_COLOR, linewidth=0.55, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        colors="black",
        labelsize=9,
        length=3.5,
        width=0.9,
        top=False,
        right=False,
    )
    if rate:
        ax.set_ylim(-0.03, 1.03)


def _add_legend(ax, loc: str = "best") -> None:
    legend = ax.legend(
        frameon=True,
        loc=loc,
        facecolor="white",
        edgecolor=GRID_COLOR,
        framealpha=1.0,
        fancybox=False,
        fontsize=9,
    )
    legend.get_frame().set_linewidth(0.8)


def _save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(output_path.with_suffix(".pdf"))


def _plot_pair(
    dreamer: Curve,
    ppo: Curve,
    output_path: Path,
    ylabel: str,
    title: str,
    smooth_points: int,
    rate: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=FIGURE_DPI)
    for curve in (dreamer, ppo):
        ax.plot(
            curve.x,
            _smooth(curve.y, smooth_points),
            color=COLORS[curve.algorithm],
            linewidth=1.9,
            label=DISPLAY_NAMES[curve.algorithm],
        )
    _style_axis(ax, ylabel=ylabel, title=title, rate=rate)
    _add_legend(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_components(
    pairs: list[tuple[str, Curve, Curve]],
    output_path: Path,
    smooth_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=FIGURE_DPI, sharex=True)
    for ax, (label, dreamer, ppo) in zip(axes.ravel(), pairs):
        for curve in (dreamer, ppo):
            ax.plot(
                curve.x,
                _smooth(curve.y, smooth_points),
                color=COLORS[curve.algorithm],
                linewidth=1.7,
                label=DISPLAY_NAMES[curve.algorithm],
            )
        _style_axis(ax, ylabel="平均回合奖励", title=label)
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
    _save_figure(fig, output_path)
    plt.close(fig)


def _plot_paper_main(
    pairs: list[tuple[str, str, Curve, Curve, bool]],
    output_path: Path,
    smooth_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), dpi=FIGURE_DPI, sharex=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, panel_label, (title, ylabel, dreamer, ppo, rate) in zip(axes.ravel(), panel_labels, pairs):
        for curve in (dreamer, ppo):
            ax.plot(
                curve.x,
                _smooth(curve.y, smooth_points),
                color=COLORS[curve.algorithm],
                linewidth=1.7,
                label=DISPLAY_NAMES[curve.algorithm],
            )
        _style_axis(ax, ylabel=ylabel, title=f"{panel_label} {title}", rate=rate)
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
    _save_figure(fig, output_path)
    plt.close(fig)


def _nearest_value(curve: Curve, step: float) -> float:
    if curve.x.size == 0:
        return math.nan
    index = int(np.argmin(np.abs(curve.x - step)))
    return float(curve.y[index])


def _write_curve_csv(curves: list[Curve], output_path: Path, smooth_points: int) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "metric", "source_tag", "real_env_step", "value", "smoothed_value"])
        for curve in curves:
            smoothed = _smooth(curve.y, smooth_points)
            for x, y, ys in zip(curve.x, curve.y, smoothed):
                writer.writerow([curve.algorithm, curve.metric, curve.source_tag, f"{x:.6g}", f"{y:.8g}", f"{ys:.8g}"])


def _write_summary_csv(curves: list[Curve], output_path: Path, compare_step: float) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "compare_real_env_step", "dreamer_value", "ppo_value", "dreamer_minus_ppo"])
        metrics = sorted({curve.metric for curve in curves})
        for metric in metrics:
            dreamer = next((curve for curve in curves if curve.metric == metric and curve.algorithm == "Dreamer"), None)
            ppo = next((curve for curve in curves if curve.metric == metric and curve.algorithm == "PPO"), None)
            if dreamer is None or ppo is None:
                continue
            dreamer_value = _nearest_value(dreamer, compare_step)
            ppo_value = _nearest_value(ppo, compare_step)
            writer.writerow(
                [
                    metric,
                    f"{compare_step:.6g}",
                    f"{dreamer_value:.8g}",
                    f"{ppo_value:.8g}",
                    f"{dreamer_value - ppo_value:.8g}",
                ]
            )


def _build_output_dir(dreamer_run: Path, ppo_run: Path, output_dir: str | None) -> Path:
    if output_dir:
        return _resolve(output_dir)
    return _resolve("logs/comparisons/world_model_vs_ppo") / f"dreamer_{dreamer_run.name}__ppo_{ppo_run.name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sample-efficiency curves for Dreamer/R2-Dreamer and skrl PPO training logs."
    )
    parser.add_argument("--dreamer-run", type=Path, default=DEFAULT_DREAMER_RUN, help="Dreamer run directory.")
    parser.add_argument("--ppo-run", type=Path, default=DEFAULT_PPO_RUN, help="skrl PPO run directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for generated figures and CSV files.")
    parser.add_argument(
        "--ppo-num-envs",
        type=_positive_int,
        default=None,
        help="PPO vectorized environment count. Defaults to params/env.yaml scene.num_envs.",
    )
    parser.add_argument(
        "--dreamer-step-scale",
        type=float,
        default=1.0,
        help="Scale applied to Dreamer logged steps. Keep 1 for this repository's Dreamer trainer.",
    )
    parser.add_argument(
        "--xmax",
        type=str,
        default="common",
        help="X-axis horizon: 'common', 'all', or a numeric real-env-step value.",
    )
    parser.add_argument(
        "--smooth-points",
        type=_positive_int,
        default=5,
        help="Trailing moving-average window in logged points. Use 1 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dreamer_run = _resolve(args.dreamer_run)
    ppo_run = _resolve(args.ppo_run)
    ppo_num_envs = args.ppo_num_envs or _infer_ppo_num_envs(ppo_run)
    ppo_step_scale = float(ppo_num_envs)

    curve_specs = [
        (
            "mean_episode_return",
            "平均回合回报",
            "平均回合回报",
            "rollout/recent_episode_score_mean",
            "Reward / Total reward (mean)",
            False,
        ),
        (
            "success_rate",
            "成功率",
            "成功率",
            "rollout/recent_termination_success",
            "Episode_Termination/success",
            True,
        ),
        (
            "severe_collision_rate",
            "严重碰撞率",
            "严重碰撞率",
            "rollout/recent_termination_severe_collision",
            "Episode_Termination/severe_collision",
            True,
        ),
        (
            "episode_length",
            "平均回合长度",
            "平均回合长度",
            "rollout/recent_episode_length_mean",
            "Episode / Total timesteps (mean)",
            False,
        ),
    ]
    component_specs = [
        ("吸取奖励", "absorb_reward", "rollout/recent_reward_absorb_reward", "Episode_Reward/absorb_reward"),
        (
            "任务完成奖励",
            "task_complete_reward",
            "rollout/recent_reward_task_complete",
            "Episode_Reward/task_complete",
        ),
        (
            "接触警告惩罚",
            "contact_warning_penalty",
            "rollout/recent_reward_contact_warning_penalty",
            "Episode_Reward/contact_warning_penalty",
        ),
        (
            "壁面间隙惩罚",
            "wall_clearance_penalty",
            "rollout/recent_reward_wall_clearance_penalty",
            "Episode_Reward/wall_clearance_penalty",
        ),
    ]

    all_curves: list[Curve] = []
    for metric, _title, _ylabel, dreamer_tag, ppo_tag, _rate in curve_specs:
        all_curves.append(_read_dreamer_curve(dreamer_run, dreamer_tag, metric, args.dreamer_step_scale))
        all_curves.append(_read_ppo_curve(ppo_run, ppo_tag, metric, ppo_step_scale))
    for _label, metric, dreamer_tag, ppo_tag in component_specs:
        all_curves.append(_read_dreamer_curve(dreamer_run, dreamer_tag, metric, args.dreamer_step_scale))
        all_curves.append(_read_ppo_curve(ppo_run, ppo_tag, metric, ppo_step_scale))

    if args.xmax == "common":
        dreamer_max = max(curve.x.max() for curve in all_curves if curve.algorithm == "Dreamer")
        ppo_max = max(curve.x.max() for curve in all_curves if curve.algorithm == "PPO")
        xmax = min(dreamer_max, ppo_max)
    elif args.xmax == "all":
        xmax = None
    else:
        xmax = float(args.xmax)

    filtered_curves = [_filter_curve(curve, xmax) for curve in all_curves]
    curve_by_key = {(curve.algorithm, curve.metric): curve for curve in filtered_curves}

    output_dir = _build_output_dir(dreamer_run, ppo_run, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, ylabel, _dreamer_tag, _ppo_tag, rate in curve_specs:
        _plot_pair(
            curve_by_key[("Dreamer", metric)],
            curve_by_key[("PPO", metric)],
            output_dir / f"{metric}.png",
            ylabel=ylabel,
            title=title,
            smooth_points=args.smooth_points,
            rate=rate,
        )

    component_pairs = [
        (label, curve_by_key[("Dreamer", metric)], curve_by_key[("PPO", metric)])
        for label, metric, _dreamer_tag, _ppo_tag in component_specs
    ]
    _plot_components(component_pairs, output_dir / "reward_components.png", args.smooth_points)

    paper_main_specs = [
        ("平均回合回报", "平均回合回报", "mean_episode_return", False),
        ("成功率", "成功率", "success_rate", True),
        ("吸取奖励", "平均回合奖励", "absorb_reward", False),
        ("严重碰撞率", "严重碰撞率", "severe_collision_rate", True),
    ]
    paper_main_pairs = [
        (
            title,
            ylabel,
            curve_by_key[("Dreamer", metric)],
            curve_by_key[("PPO", metric)],
            rate,
        )
        for title, ylabel, metric, rate in paper_main_specs
    ]
    _plot_paper_main(paper_main_pairs, output_dir / "paper_main_comparison.png", args.smooth_points)

    _write_curve_csv(filtered_curves, output_dir / "plot_data.csv", args.smooth_points)
    compare_step = xmax if xmax is not None else min(
        max(curve.x.max() for curve in filtered_curves if curve.algorithm == "Dreamer"),
        max(curve.x.max() for curve in filtered_curves if curve.algorithm == "PPO"),
    )
    _write_summary_csv(filtered_curves, output_dir / "summary_at_common_horizon.csv", compare_step)

    print(f"[INFO] Dreamer run: {dreamer_run}")
    print(f"[INFO] PPO run: {ppo_run}")
    print(f"[INFO] PPO TensorBoard step scale: {ppo_step_scale:g} real env steps per logged step")
    print(f"[INFO] X horizon: {'all' if xmax is None else f'{xmax:.0f} real env steps'}")
    print(f"[INFO] Wrote figures and CSV files to: {output_dir}")


if __name__ == "__main__":
    main()
