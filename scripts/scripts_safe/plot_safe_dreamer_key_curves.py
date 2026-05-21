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


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DPI = 300
GRID_COLOR = "#E1E1E1"

DEFAULT_RUNS = {
    "state_dreamer": Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/2026-05-15_21-32-23"),
    "state_safe": Path("logs/r2dreamer/ur3_blood_pipe_state_safe_dreamer/2026-05-15_09-18-25"),
    "vision_dreamer": Path(
        "logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-12_21-14-29_seed_0_600k"
    ),
    "vision_safe": Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_safe_dreamer/2026-05-16_06-36-51"),
}

COLORS = {
    "Dreamer": "#0072B2",
    "Safe-Dreamer": "#D55E00",
}

ALGORITHM_LABELS = {
    "Dreamer": "Dreamer V3",
    "Safe-Dreamer": "Risk Dreamer",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    tag: str
    title: str
    ylabel: str
    rate: bool = False


@dataclass(frozen=True)
class Curve:
    task: str
    algorithm: str
    metric: str
    tag: str
    x: np.ndarray
    y: np.ndarray
    y_smooth: np.ndarray


METRICS = (
    MetricSpec(
        key="success",
        tag="rollout/recent_termination_success",
        title="成功终止率",
        ylabel="终止率",
        rate=True,
    ),
    MetricSpec(
        key="severe_collision",
        tag="rollout/recent_termination_severe_collision",
        title="碰撞终止率",
        ylabel="终止率",
        rate=True,
    ),
    MetricSpec(
        key="lambda",
        tag="safe_dreamer/lambda",
        title="拉格朗日乘子",
        ylabel="乘子值",
        rate=False,
    ),
)


def _configure_plot_style() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Noto Serif CJK SC",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "Microsoft YaHei",
        "SimSun",
        "Times New Roman",
        "DejaVu Serif",
    ]
    font_stack = [font for font in preferred_fonts if font in available_fonts] or ["DejaVu Serif"]
    plt.rcParams.update(
        {
            "font.family": font_stack,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot key Safe-Dreamer comparison curves: success termination, severe collision termination, "
            "and Lagrangian lambda."
        )
    )
    parser.add_argument("--state-dreamer", type=Path, default=DEFAULT_RUNS["state_dreamer"])
    parser.add_argument("--state-safe", type=Path, default=DEFAULT_RUNS["state_safe"])
    parser.add_argument("--vision-dreamer", type=Path, default=DEFAULT_RUNS["vision_dreamer"])
    parser.add_argument("--vision-safe", type=Path, default=DEFAULT_RUNS["vision_safe"])
    parser.add_argument(
        "--xmax",
        type=float,
        default=510_000.0,
        help="Maximum environment step to plot. Use <=0 to keep full logs.",
    )
    parser.add_argument(
        "--smooth-points",
        type=int,
        default=3,
        help="Trailing moving-average window over logged points. Use 1 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/scripts_safe/safe_dreamer_key_curves"),
        help="Directory for generated figures and CSV.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _format_steps(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size <= 2:
        return y.astype(np.float64, copy=True)
    window = min(window, int(y.size))
    out = np.empty_like(y, dtype=np.float64)
    cumsum = np.cumsum(np.insert(y.astype(np.float64), 0, 0.0))
    for index in range(y.size):
        start = max(0, index - window + 1)
        out[index] = (cumsum[index + 1] - cumsum[start]) / (index - start + 1)
    return out


def _read_curve(
    run_dir: Path,
    task: str,
    algorithm: str,
    spec: MetricSpec,
    smooth_points: int,
    xmax: float,
) -> Curve | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    xs: list[float] = []
    ys: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if spec.tag not in row:
                continue
            step = float(row.get("step", len(xs)))
            if xmax > 0 and step > xmax:
                continue
            xs.append(step)
            ys.append(float(row[spec.tag]))

    if not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    return Curve(task, algorithm, spec.key, spec.tag, x, y, _smooth(y, smooth_points))


def _load_curves(args: argparse.Namespace) -> list[Curve]:
    runs = {
        ("状态观测", "Dreamer"): _resolve(args.state_dreamer),
        ("状态观测", "Safe-Dreamer"): _resolve(args.state_safe),
        ("视觉观测", "Dreamer"): _resolve(args.vision_dreamer),
        ("视觉观测", "Safe-Dreamer"): _resolve(args.vision_safe),
    }

    curves: list[Curve] = []
    for (task, algorithm), run_dir in runs.items():
        for spec in METRICS:
            # Lambda is a Safe-Dreamer-only diagnostic.
            if spec.key == "lambda" and algorithm != "Safe-Dreamer":
                continue
            curve = _read_curve(
                run_dir=run_dir,
                task=task,
                algorithm=algorithm,
                spec=spec,
                smooth_points=int(args.smooth_points),
                xmax=float(args.xmax),
            )
            if curve is not None:
                curves.append(curve)
            else:
                print(f"[WARN] Tag not found: {spec.tag} in {run_dir}")
    return curves


def _style_axis(ax, spec: MetricSpec) -> None:
    ax.set_title(spec.title, pad=7)
    ax.set_xlabel("环境交互步数")
    ax.set_ylabel(spec.ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.0, width=0.8, top=False, right=False)
    if spec.rate:
        ax.set_ylim(-0.03, 1.03)


def _plot_combined(curves: list[Curve], output_dir: Path) -> None:
    tasks = ("状态观测", "视觉观测")

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharex=False)
    for row, task in enumerate(tasks):
        for col, spec in enumerate(METRICS):
            ax = axes[row, col]
            _style_axis(ax, spec)
            if col == 0:
                ax.text(
                    -0.30,
                    0.5,
                    task,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    weight="bold",
                )

            plotted = False
            for algorithm in ("Dreamer", "Safe-Dreamer"):
                curve = next(
                    (
                        item
                        for item in curves
                        if item.task == task and item.algorithm == algorithm and item.metric == spec.key
                    ),
                    None,
                )
                if curve is None:
                    continue
                ax.plot(
                    curve.x,
                    curve.y_smooth,
                    color=COLORS[algorithm],
                    linewidth=1.8,
                    label=ALGORITHM_LABELS.get(algorithm, algorithm),
                )
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "无数据", transform=ax.transAxes, ha="center", va="center")
            else:
                legend_loc = "lower right" if col == 0 else "upper right"
                ax.legend(loc=legend_loc, frameon=True, framealpha=0.86, borderpad=0.35, handlelength=2.2)

    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"safe_dreamer_key_curves.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _write_curve_csv(curves: list[Curve], output_dir: Path) -> None:
    csv_path = output_dir / "safe_dreamer_key_curve_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "algorithm", "metric", "tag", "step", "value", "smoothed_value"],
        )
        writer.writeheader()
        for curve in curves:
            for step, value, smoothed_value in zip(curve.x, curve.y, curve.y_smooth):
                writer.writerow(
                    {
                        "task": curve.task,
                        "algorithm": curve.algorithm,
                        "metric": curve.metric,
                        "tag": curve.tag,
                        "step": float(step),
                        "value": float(value),
                        "smoothed_value": float(smoothed_value),
                    }
                )


def _write_summary(curves: list[Curve], output_dir: Path) -> None:
    summary_path = output_dir / "safe_dreamer_key_curve_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "algorithm", "metric", "tag", "points", "first_step", "first_value", "last_step", "last_value"],
        )
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "task": curve.task,
                    "algorithm": curve.algorithm,
                    "metric": curve.metric,
                    "tag": curve.tag,
                    "points": int(curve.x.size),
                    "first_step": float(curve.x[0]),
                    "first_value": float(curve.y[0]),
                    "last_step": float(curve.x[-1]),
                    "last_value": float(curve.y[-1]),
                }
            )


def main() -> None:
    args = _parse_args()
    _configure_plot_style()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = _load_curves(args)
    if not curves:
        raise RuntimeError("No curves were loaded. Check run paths and metric tags.")

    _plot_combined(curves, output_dir)
    _write_curve_csv(curves, output_dir)
    _write_summary(curves, output_dir)
    print(f"[INFO] Wrote figure: {output_dir / 'safe_dreamer_key_curves.png'}")
    print(f"[INFO] Wrote figure: {output_dir / 'safe_dreamer_key_curves.pdf'}")
    print(f"[INFO] Wrote CSV: {output_dir / 'safe_dreamer_key_curve_points.csv'}")
    print(f"[INFO] Wrote summary: {output_dir / 'safe_dreamer_key_curve_summary.csv'}")


if __name__ == "__main__":
    main()
