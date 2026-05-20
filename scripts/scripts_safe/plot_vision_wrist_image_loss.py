#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
IMAGE_LOSS_TAG = "train/loss/image"
DEFAULT_RUN = Path("logs/r2dreamer/ur3_blood_pipe_vision_wrist_dreamer/2026-05-12_21-14-29_seed_0_600k")


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
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot wrist-vision Dreamer image reconstruction loss.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help="R2-Dreamer run directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/scripts_safe/vision_wrist_image_loss"),
        help="Directory for generated figure and CSV.",
    )
    parser.add_argument("--xmax", type=float, default=100_000.0, help="Maximum step. Use <=0 to keep all points.")
    parser.add_argument("--smooth-points", type=int, default=3, help="Trailing moving-average window.")
    parser.add_argument(
        "--drop-first",
        action="store_true",
        default=False,
        help="Drop the first point when plotting. Useful when the initial loss is much larger than later values.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=False,
        help="Use logarithmic y-axis.",
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


def _read_image_loss(run_dir: Path, xmax: float) -> tuple[np.ndarray, np.ndarray]:
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
            if IMAGE_LOSS_TAG not in row:
                continue
            step = float(row.get("step", len(xs)))
            if xmax > 0 and step > xmax:
                continue
            xs.append(step)
            ys.append(float(row[IMAGE_LOSS_TAG]))
    if not xs:
        raise KeyError(f"Tag not found in {metrics_path}: {IMAGE_LOSS_TAG}")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _plot_image_loss(
    x: np.ndarray,
    y: np.ndarray,
    y_smooth: np.ndarray,
    output_dir: Path,
    log_y: bool,
    xmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, y, color="#9ECAE1", linewidth=1.0, alpha=0.55, label="原始曲线")
    ax.plot(x, y_smooth, color="#0072B2", linewidth=2.0, label="平滑曲线")
    ax.set_title("世界模型图像重建损失", pad=8)
    ax.set_xlabel("环境交互步数")
    ax.set_ylabel("图像重建损失")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    if xmax > 0:
        ax.set_xlim(0.0, xmax)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("图像重建损失（对数坐标）")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=3.0, width=0.8, top=False, right=False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.86, borderpad=0.35, handlelength=2.2)
    fig.tight_layout()
    suffix = "_logy" if log_y else ""
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"vision_wrist_image_loss{suffix}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _write_csv(x: np.ndarray, y: np.ndarray, y_smooth: np.ndarray, output_dir: Path) -> None:
    with (output_dir / "vision_wrist_image_loss_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "image_loss", "smoothed_image_loss"])
        writer.writeheader()
        for step, value, smooth_value in zip(x, y, y_smooth):
            writer.writerow(
                {
                    "step": float(step),
                    "image_loss": float(value),
                    "smoothed_image_loss": float(smooth_value),
                }
            )

    with (output_dir / "vision_wrist_image_loss_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["points", "first_step", "first_value", "last_step", "last_value", "min_value", "max_value"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "points": int(x.size),
                "first_step": float(x[0]),
                "first_value": float(y[0]),
                "last_step": float(x[-1]),
                "last_value": float(y[-1]),
                "min_value": float(y.min()),
                "max_value": float(y.max()),
            }
        )


def main() -> None:
    args = _parse_args()
    _configure_plot_style()
    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = _read_image_loss(run_dir, float(args.xmax))
    if args.drop_first and x.size > 1:
        x = x[1:]
        y = y[1:]
    y_smooth = _smooth(y, int(args.smooth_points))
    _plot_image_loss(x, y, y_smooth, output_dir, bool(args.log_y), float(args.xmax))
    _write_csv(x, y, y_smooth, output_dir)

    suffix = "_logy" if args.log_y else ""
    print(f"[INFO] Wrote figure: {output_dir / f'vision_wrist_image_loss{suffix}.png'}")
    print(f"[INFO] Wrote figure: {output_dir / f'vision_wrist_image_loss{suffix}.pdf'}")
    print(f"[INFO] Wrote CSV: {output_dir / 'vision_wrist_image_loss_points.csv'}")


if __name__ == "__main__":
    main()
