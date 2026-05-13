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
DEFAULT_RUN_DIR = Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-08_21-19-37_seed_0_800k")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
FIGURE_DPI = 300
GRID_COLOR = "#E3E3E3"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    tag: str
    rate: bool = False


@dataclass(frozen=True)
class Curve:
    key: str
    tag: str
    x: np.ndarray
    raw: np.ndarray
    smoothed: np.ndarray


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
        description="Plot smoothed seed-0 State RSL-RL PPO TensorBoard curves."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="RSL-RL run directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="TensorBoard-like exponential smoothing factor. 0 disables smoothing.",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Scale RSL-RL PPO iterations on the x-axis. Defaults to 1.",
    )
    parser.add_argument("--color", type=str, default="#D67AB2", help="Line color, e.g. '#D67AB2'.")
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


def _read_curve(run_dir: Path, spec: MetricSpec, step_scale: float, smoothing: float) -> Curve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required. Run with: "
            "conda run -n isaacsim-4.2 python scripts/scripts_2/PPO/state_rslrl_ppo/"
            "plot_state_rslrl_ppo_seed0_training.py"
        ) from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    if spec.tag not in tags:
        available = ", ".join(sorted(tags))
        raise KeyError(f"Missing TensorBoard tag {spec.tag!r}. Available tags: {available}")

    events = accumulator.Scalars(spec.tag)
    if not events:
        raise RuntimeError(f"TensorBoard tag has no scalar events: {spec.tag}")
    x = np.asarray([event.step * step_scale for event in events], dtype=np.float64)
    raw = np.asarray([event.value for event in events], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    raw = raw[order]
    return Curve(spec.key, spec.tag, x, raw, _smooth_tensorboard_like(raw, smoothing))


def _style_axis(ax, spec: MetricSpec) -> None:
    ax.set_title(spec.title, pad=8, weight="normal")
    ax.set_xlabel("迭代次数 (iteration)", labelpad=6)
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


def _plot_curve(curve: Curve, spec: MetricSpec, output_dir: Path, color: str) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2), dpi=FIGURE_DPI)
    ax.plot(curve.x, curve.smoothed, color=color, linewidth=1.2, zorder=3)
    _style_axis(ax, spec)
    fig.tight_layout()

    output_base = output_dir / spec.key
    fig.savefig(output_base.with_suffix(".png"), dpi=FIGURE_DPI)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)
    return output_base.with_suffix(".png")


def _write_csv(curves: list[Curve], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "tag", "iteration", "raw_value", "smoothed_value"])
        writer.writeheader()
        for curve in curves:
            for x, raw, smoothed in zip(curve.x, curve.raw, curve.smoothed):
                writer.writerow(
                    {
                        "metric": curve.key,
                        "tag": curve.tag,
                        "iteration": float(x),
                        "raw_value": float(raw),
                        "smoothed_value": float(smoothed),
                    }
                )


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_scale = float(args.step_scale)
    curves = [_read_curve(run_dir, spec, step_scale, args.smoothing) for spec in METRICS]

    saved = [_plot_curve(curve, spec, output_dir, args.color) for curve, spec in zip(curves, METRICS)]
    csv_path = output_dir / "state_rslrl_ppo_seed0_curves.csv"
    _write_csv(curves, csv_path)

    print(f"[INFO] Run: {run_dir}")
    print(f"[INFO] Line color: {args.color}")
    print(f"[INFO] RSL-RL iteration scale: {step_scale:g} x-axis units per logged iteration")
    for curve in curves:
        print(f"[INFO] {curve.key}: tag={curve.tag}, x=[{curve.x.min():.0f}, {curve.x.max():.0f}]")
    for path in saved:
        print(f"[INFO] Saved: {path}")
        print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
