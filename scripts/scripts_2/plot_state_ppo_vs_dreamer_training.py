#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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
GRID_COLOR = "#E3E3E3"

DEFAULT_PPO_RUNS = (
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-08_21-19-37_seed_0_800k"),
    Path("logs/rsl_rl/ur3_blood_pipe_state_direct/2026-05-09_09-49-21_seed_1_800k"),
)
DEFAULT_DREAMER_RUNS = (
    Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_0_800k"),
    Path("logs/r2dreamer/ur3_blood_pipe_state_dreamer/seed_1_800k"),
)

GROUP_LABELS = {
    "ppo": "State PPO (RSL-RL)",
    "dreamer": "State Dreamer",
}
GROUP_COLORS = {
    "ppo": "#26A7E1",
    "dreamer": "#E95412",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    ylabel: str
    ppo_tags: tuple[str, ...]
    dreamer_tags: tuple[str, ...]
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
        ppo_tags=("Train/mean_reward",),
        dreamer_tags=("rollout/interval_episode_score_mean", "rollout/recent_episode_score_mean"),
    ),
    MetricSpec(
        key="success_rate",
        title="成功率",
        ylabel="成功率",
        ppo_tags=("Episode_Termination/success",),
        dreamer_tags=("rollout/recent_termination_success",),
        rate=True,
    ),
    # MetricSpec(
    #     key="absorbed_ratio",
    #     title="吸取比例",
    #     ylabel="吸取比例",
    #     ppo_tags=("Metrics/absorbed_ratio_mean",),
    #     dreamer_tags=("Metrics/absorbed_ratio_mean",),
    #     rate=True,
    # ),
    # MetricSpec(
    #     key="severe_collision_rate",
    #     title="严重碰撞率",
    #     ylabel="严重碰撞率",
    #     ppo_tags=("Episode_Termination/severe_collision",),
    #     dreamer_tags=("rollout/recent_termination_severe_collision",),
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
        description="Plot averaged state-observation PPO vs Dreamer training curves."
    )
    parser.add_argument(
        "--ppo-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_PPO_RUNS),
        help="RSL-RL state-observation PPO run directories.",
    )
    parser.add_argument(
        "--dreamer-runs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_DREAMER_RUNS),
        help="Dreamer state-observation run directories.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=600_000.0,
        help="Real environment step horizon. Default is 600k because the Dreamer runs stop there.",
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
        default=Path("scripts/scripts_2/state_ppo_vs_dreamer_600k"),
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


def _ppo_step_scale(run_dir: Path) -> float:
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    num_envs = _read_section_int(env_yaml, "scene", "num_envs") or 4
    num_steps_per_env = _read_top_level_int(agent_yaml, "num_steps_per_env") or 32
    return float(num_envs * num_steps_per_env)


def _load_ppo_curve(run_dir: Path, run_index: int, spec: MetricSpec, step_scale: float) -> RawCurve:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorboard is required. Run with the Isaac environment, for example: "
            "conda run -n isaacsim-4.2 python scripts/scripts_2/plot_state_ppo_vs_dreamer_training.py"
        ) from exc

    if not run_dir.is_dir():
        raise FileNotFoundError(f"PPO run directory not found: {run_dir}")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    tag = next((candidate for candidate in spec.ppo_tags if candidate in available_tags), None)
    if tag is None:
        available = ", ".join(sorted(available_tags))
        raise KeyError(
            f"None of the expected PPO tags were found for metric '{spec.key}' in {run_dir}. "
            f"Expected one of: {spec.ppo_tags}. Available tags: {available}"
        )

    events = accumulator.Scalars(tag)
    if not events:
        raise RuntimeError(f"PPO TensorBoard tag has no scalar events: {tag} in {run_dir}")

    x = np.asarray([event.step * step_scale for event in events], dtype=np.float64)
    y = np.asarray([event.value for event in events], dtype=np.float64)
    return RawCurve("ppo", run_index, run_dir, spec.key, tag, x, y)


def _load_dreamer_curve(run_dir: Path, run_index: int, spec: MetricSpec) -> RawCurve:
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
            selected_tag = next((candidate for candidate in spec.dreamer_tags if candidate in row), None)
        if selected_tag is not None and selected_tag in row:
            xs.append(float(row["step"]))
            ys.append(float(row[selected_tag]))

    if selected_tag is None:
        available = ", ".join(sorted(available_tags))
        raise KeyError(
            f"None of the expected Dreamer tags were found for metric '{spec.key}' in {metrics_path}. "
            f"Expected one of: {spec.dreamer_tags}. Available tags: {available}"
        )
    if not xs:
        raise RuntimeError(f"Dreamer tag has no scalar values: {selected_tag} in {metrics_path}")

    return RawCurve(
        "dreamer",
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
    ppo_runs = [_resolve(path) for path in args.ppo_runs]
    dreamer_runs = [_resolve(path) for path in args.dreamer_runs]
    grid = np.linspace(0.0, float(args.xmax), int(args.points))
    aggregates: dict[str, dict[str, AggregateCurve]] = {}

    for spec in METRICS:
        ppo_curves = [
            _load_ppo_curve(run_dir, index, spec, _ppo_step_scale(run_dir))
            for index, run_dir in enumerate(ppo_runs)
        ]
        dreamer_curves = [
            _load_dreamer_curve(run_dir, index, spec)
            for index, run_dir in enumerate(dreamer_runs)
        ]
        aggregates[spec.key] = {
            "ppo": _aggregate(ppo_curves, grid, args.smooth_points),
            "dreamer": _aggregate(dreamer_curves, grid, args.smooth_points),
        }

        print(
            f"[INFO] {spec.key}: "
            f"ppo tag={ppo_curves[0].tag}, dreamer tag={dreamer_curves[0].tag}, "
            f"ppo x=[{ppo_curves[0].x.min():.0f}, {ppo_curves[0].x.max():.0f}], "
            f"dreamer x=[{dreamer_curves[0].x.min():.0f}, {dreamer_curves[0].x.max():.0f}]"
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
        for group in ("ppo", "dreamer"):
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
    max_ppo_runs = max(curves["ppo"].values.shape[0] for curves in aggregates.values())
    max_dreamer_runs = max(curves["dreamer"].values.shape[0] for curves in aggregates.values())
    fieldnames = ["metric", "env_steps", "ppo_mean", "ppo_std", "dreamer_mean", "dreamer_std"]
    fieldnames.extend(f"ppo_seed_{index}" for index in range(max_ppo_runs))
    fieldnames.extend(f"dreamer_seed_{index}" for index in range(max_dreamer_runs))
    fieldnames.extend(f"ppo_raw_seed_{index}" for index in range(max_ppo_runs))
    fieldnames.extend(f"dreamer_raw_seed_{index}" for index in range(max_dreamer_runs))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for spec in METRICS:
            ppo = aggregates[spec.key]["ppo"]
            dreamer = aggregates[spec.key]["dreamer"]
            for index, env_steps in enumerate(ppo.x):
                row: dict[str, object] = {
                    "metric": spec.key,
                    "env_steps": float(env_steps),
                    "ppo_mean": float(ppo.mean[index]),
                    "ppo_std": float(ppo.std[index]),
                    "dreamer_mean": float(dreamer.mean[index]),
                    "dreamer_std": float(dreamer.std[index]),
                }
                for run_index in range(ppo.values.shape[0]):
                    row[f"ppo_seed_{run_index}"] = float(ppo.values[run_index, index])
                for run_index in range(dreamer.values.shape[0]):
                    row[f"dreamer_seed_{run_index}"] = float(dreamer.values[run_index, index])
                for run_index in range(ppo.raw_values.shape[0]):
                    row[f"ppo_raw_seed_{run_index}"] = float(ppo.raw_values[run_index, index])
                for run_index in range(dreamer.raw_values.shape[0]):
                    row[f"dreamer_raw_seed_{run_index}"] = float(dreamer.raw_values[run_index, index])
                writer.writerow(row)


def main() -> None:
    _configure_plot_style()
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregates = _read_all_curves(args)
    output_base = output_dir / "state_ppo_vs_dreamer_training"
    figure_paths = _plot(aggregates, output_base)
    csv_path = output_dir / "state_ppo_vs_dreamer_training_curves.csv"
    _write_csv(aggregates, csv_path)

    for path in figure_paths:
        print(f"[INFO] Saved: {path}")
        print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
