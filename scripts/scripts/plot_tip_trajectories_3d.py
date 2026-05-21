#!/usr/bin/env python3
"""Plot saved suction tip trajectories in 3D."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results"

METHOD_COLORS = {
    "state_ppo": "#26A7E1",
    "vision_ppo": "#13AF68",
    "state_dreamer": "#E95412",
    "vision_dreamer": "#E274A9",
}

TRAJECTORY_CMAPS = [
    LinearSegmentedColormap.from_list("trajectory_1_progress", ["#BFEFFF", "#26A7E1", "#064F7A"]),
    LinearSegmentedColormap.from_list("trajectory_2_progress", ["#FFD6C2", "#E95412", "#7A2408"]),
]

TRAJECTORY_BASE_COLORS = ["#26A7E1", "#E95412"]


@dataclass
class Trajectory:
    path: Path
    label: str
    method: str
    run_index: int | None
    seed: int | None
    outcome: str
    outcome_index: int | None
    episode_id: int | None
    steps: list[int]
    time_s: list[float]
    x: list[float]
    y: list[float]
    z: list[float]
    force_n: list[float]

    @property
    def duration_s(self) -> float:
        return self.time_s[-1] if self.time_s else 0.0

    @property
    def path_length_m(self) -> float:
        return sum(
            math.dist(
                (self.x[i - 1], self.y[i - 1], self.z[i - 1]),
                (self.x[i], self.y[i], self.z[i]),
            )
            for i in range(1, len(self.x))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Replay result directory. Defaults to the newest result directory containing trajectories.",
    )
    parser.add_argument(
        "--trajectory-files",
        type=Path,
        nargs="*",
        default=None,
        help="Specific trajectory CSV files. Overrides --result-dir discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures and trajectory stats. Defaults to RESULT_DIR/trajectory_plots.",
    )
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used when a second trajectory has to be randomly selected from the same result directory.",
    )
    parser.add_argument(
        "--axis-padding-frac",
        type=float,
        default=0.08,
        help="Fractional XYZ padding for each grouped subplot.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    font_names = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    for font_path in [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/msttcorefonts/msyh.ttf"),
        Path("/usr/share/fonts/truetype/windows/msyh.ttf"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            font_names.insert(0, font_name)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_names,
            "axes.unicode_minus": False,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.titlesize": 8.0,
        }
    )


def latest_result_dir_with_trajectories() -> Path:
    candidates = []
    for result_dir in DEFAULT_RESULTS_ROOT.iterdir() if DEFAULT_RESULTS_ROOT.exists() else []:
        if result_dir.is_dir() and list((result_dir / "trajectories").glob("*_tip_trajectory.csv")):
            candidates.append(result_dir)
    if not candidates:
        raise FileNotFoundError(f"No trajectory CSV files found under {DEFAULT_RESULTS_ROOT}")
    return sorted(candidates)[-1]


def discover_trajectory_files(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    if args.trajectory_files:
        files = [path.resolve() for path in args.trajectory_files]
        result_dir = args.result_dir.resolve() if args.result_dir else files[0].parents[1]
        return result_dir, files

    result_dir = args.result_dir.resolve() if args.result_dir else latest_result_dir_with_trajectories()
    files = sorted((result_dir / "trajectories").glob("*_tip_trajectory.csv"))
    if not files:
        raise FileNotFoundError(f"No *_tip_trajectory.csv files found in {result_dir / 'trajectories'}")
    return result_dir, files


def infer_method(path: Path) -> str:
    stem = path.stem
    for method in METHOD_COLORS:
        if stem.startswith(method):
            return method
    return "unknown"


def parse_trajectory_name(path: Path) -> dict[str, int | str | None]:
    stem = path.stem.replace("_tip_trajectory", "")
    match = re.match(
        r"^(?P<prefix>.+?)(?:_run(?P<run_index>\d+))?_seed(?P<seed>\d+)_"
        r"(?P<outcome>success|failure)(?P<outcome_index>\d+)_episode(?P<episode_id>\d+)$",
        stem,
    )
    if not match:
        return {
            "run_index": None,
            "seed": None,
            "outcome": "unknown",
            "outcome_index": None,
            "episode_id": None,
        }
    return {
        "run_index": int(match.group("run_index")) if match.group("run_index") is not None else None,
        "seed": int(match.group("seed")),
        "outcome": match.group("outcome"),
        "outcome_index": int(match.group("outcome_index")),
        "episode_id": int(match.group("episode_id")),
    }


def make_label(path: Path) -> str:
    parsed = parse_trajectory_name(path)
    if parsed["outcome_index"] is not None:
        return f"轨迹{parsed['outcome_index']}"
    return path.stem.replace("_tip_trajectory", "")


def read_trajectory(path: Path) -> Trajectory:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"Empty trajectory file: {path}")

    required = {"step", "time_s", "tip_pos_w_x", "tip_pos_w_y", "tip_pos_w_z", "tip_contact_force_n"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise KeyError(f"{path} is missing required columns: {', '.join(missing)}")

    parsed = parse_trajectory_name(path)
    return Trajectory(
        path=path,
        label=make_label(path),
        method=infer_method(path),
        run_index=parsed["run_index"],
        seed=parsed["seed"],
        outcome=str(parsed["outcome"]),
        outcome_index=parsed["outcome_index"],
        episode_id=parsed["episode_id"],
        steps=[int(float(row["step"])) for row in rows],
        time_s=[float(row["time_s"]) for row in rows],
        x=[float(row["tip_pos_w_x"]) for row in rows],
        y=[float(row["tip_pos_w_y"]) for row in rows],
        z=[float(row["tip_pos_w_z"]) for row in rows],
        force_n=[float(row["tip_contact_force_n"]) for row in rows],
    )


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = sum(x_limits) / 2.0
    y_mid = sum(y_limits) / 2.0
    z_mid = sum(z_limits) / 2.0
    radius = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    ) / 2.0
    if radius <= 0.0:
        radius = 0.01

    ax.set_xlim3d(x_mid - radius, x_mid + radius)
    ax.set_ylim3d(y_mid - radius, y_mid + radius)
    ax.set_zlim3d(z_mid - radius, z_mid + radius)


def write_stats_csv(path: Path, trajectories: list[Trajectory]) -> None:
    fieldnames = [
        "file",
        "label",
        "method",
        "run_index",
        "seed",
        "outcome",
        "outcome_index",
        "episode_id",
        "num_points",
        "duration_s",
        "path_length_m",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "z_min_m",
        "z_max_m",
        "tip_contact_force_mean_n",
        "tip_contact_force_max_n",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for traj in trajectories:
            writer.writerow(
                {
                    "file": str(traj.path),
                    "label": traj.label,
                    "method": traj.method,
                    "run_index": traj.run_index,
                    "seed": traj.seed,
                    "outcome": traj.outcome,
                    "outcome_index": traj.outcome_index,
                    "episode_id": traj.episode_id,
                    "num_points": len(traj.x),
                    "duration_s": traj.duration_s,
                    "path_length_m": traj.path_length_m,
                    "x_min_m": min(traj.x),
                    "x_max_m": max(traj.x),
                    "y_min_m": min(traj.y),
                    "y_max_m": max(traj.y),
                    "z_min_m": min(traj.z),
                    "z_max_m": max(traj.z),
                    "tip_contact_force_mean_n": sum(traj.force_n) / len(traj.force_n),
                    "tip_contact_force_max_n": max(traj.force_n),
                }
            )


def _trajectory_sort_key(traj: Trajectory) -> tuple[int, int, str]:
    return traj.outcome_index or 9999, traj.episode_id or 9999, traj.path.name


def _supplement_to_two_trajectories(
    selected: list[Trajectory],
    trajectories: list[Trajectory],
    rng: random.Random,
) -> list[Trajectory]:
    if len(selected) != 1 or len(trajectories) <= 1:
        return selected[:2]

    selected_paths = {traj.path for traj in selected}
    candidates = [traj for traj in trajectories if traj.path not in selected_paths]
    if not candidates:
        return selected[:2]

    extra = rng.choice(candidates)
    print(
        "Supplement trajectory: current rule selected only one curve; "
        f"randomly added {extra.path.name} from the same outcome directory."
    )
    return (selected + [extra])[:2]


def _select_one_trajectory_per_run(
    trajectories: list[Trajectory],
    rng: random.Random,
) -> list[Trajectory]:
    selected: list[Trajectory] = []
    run_indices = sorted({traj.run_index for traj in trajectories if traj.run_index is not None})
    if not run_indices:
        run_indices = sorted({traj.seed for traj in trajectories if traj.seed is not None})

    for run_index in run_indices:
        if any(traj.run_index is not None for traj in trajectories):
            candidates = [traj for traj in trajectories if traj.run_index == run_index]
        else:
            candidates = [traj for traj in trajectories if traj.seed == run_index]
        candidates.sort(key=_trajectory_sort_key)
        if candidates:
            selected.append(candidates[0])
    if selected:
        return _supplement_to_two_trajectories(selected[:2], trajectories, rng)
    return sorted(trajectories, key=_trajectory_sort_key)[:2]


def _trajectory_segments(traj: Trajectory) -> np.ndarray:
    points = _plot_points(traj)
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack([points[:-1], points[1:]], axis=1)


def _plot_points(traj: Trajectory) -> np.ndarray:
    points = np.column_stack([traj.x, traj.y, traj.z])
    if len(points) > 1:
        return points[:-1]
    return points


def _set_equal_data_axes(ax, trajectories: list[Trajectory]) -> None:
    points = np.concatenate([_plot_points(traj) for traj in trajectories], axis=0)
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    ax.auto_scale_xyz(xs, ys, zs)
    set_axes_equal(ax)


def _plot_gradient_trajectory(
    ax,
    traj: Trajectory,
    label: str,
    linestyle: str,
    linewidth: float,
    cmap,
    norm: Normalize,
) -> None:
    segments = _trajectory_segments(traj)
    if len(segments) > 0:
        progress = np.linspace(0.0, 1.0, len(segments))
        collection = Line3DCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
            linestyles=linestyle,
        )
        collection.set_array(progress)
        ax.add_collection3d(collection)
    points = _plot_points(traj)
    if len(points) == 0:
        return
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color=cmap(0.0), marker="o", s=18)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=cmap(1.0), marker="^", s=22)


def _plot_outcome_figure(
    trajectories: list[Trajectory],
    outcome: str,
    output_dir: Path,
    png_dpi: int,
    view_elev: float,
    view_azim: float,
    rng: random.Random,
) -> tuple[Path, Path] | None:
    outcome_trajectories = [traj for traj in trajectories if traj.outcome == outcome]
    if not outcome_trajectories:
        print(f"Skip {outcome} trajectories: no matching trajectory files found.")
        return None

    selected = _select_one_trajectory_per_run(outcome_trajectories, rng)
    if not selected:
        print(f"Skip {outcome} trajectories: no trajectories selected.")
        return None

    title = "成功轨迹" if outcome == "success" else "失败轨迹"
    fig = plt.figure(figsize=(5.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0.0, vmax=1.0)
    styles = ["-", "--"]

    selected = selected[:2]
    for index, traj in enumerate(selected):
        cmap = TRAJECTORY_CMAPS[index % len(TRAJECTORY_CMAPS)]
        _plot_gradient_trajectory(
            ax,
            traj,
            label=f"轨迹{index + 1}",
            linestyle=styles[index % len(styles)],
            linewidth=1.4,
            cmap=cmap,
            norm=norm,
        )

    ax.set_title(title, y=1.00)
    ax.set_xlabel("世界坐标 X (m)")
    ax.set_ylabel("世界坐标 Y (m)")
    ax.set_zlabel("世界坐标 Z (m)")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(True, alpha=0.35)
    _set_equal_data_axes(ax, selected)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_BASE_COLORS[index % len(TRAJECTORY_BASE_COLORS)],
            linestyle=styles[index % len(styles)],
            linewidth=1.4,
            label=f"轨迹{index + 1}",
        )
        for index in range(len(selected))
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.88, 0.88), frameon=True)

    fig.tight_layout()
    png_path = output_dir / f"{outcome}_tip_trajectories_3d.png"
    pdf_path = output_dir / f"{outcome}_tip_trajectories_3d.pdf"
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    for index, traj in enumerate(selected, start=1):
        print(
            f"Selected {outcome} 轨迹{index}: "
            f"run={traj.run_index}, seed={traj.seed}, file={traj.path.name}"
        )
    return png_path, pdf_path


def plot_grouped_trajectories(
    trajectories: list[Trajectory],
    output_dir: Path,
    png_dpi: int,
    view_elev: float,
    view_azim: float,
    axis_padding_frac: float,
    random_seed: int,
) -> list[tuple[Path, Path]]:
    saved_paths: list[tuple[Path, Path]] = []
    rng = random.Random(random_seed)

    for outcome in ("success", "failure"):
        paths = _plot_outcome_figure(
            trajectories,
            outcome=outcome,
            output_dir=output_dir,
            png_dpi=png_dpi,
            view_elev=view_elev,
            view_azim=view_azim,
            rng=rng,
        )
        if paths is not None:
            saved_paths.append(paths)

    unknown_trajectories = [traj for traj in trajectories if traj.outcome not in {"success", "failure"}]
    if unknown_trajectories:
        fig = plt.figure(figsize=(5.6, 4.2))
        ax = fig.add_subplot(111, projection="3d")
        norm = Normalize(vmin=0.0, vmax=1.0)
        for index, traj in enumerate(unknown_trajectories[:2]):
            cmap = TRAJECTORY_CMAPS[index % len(TRAJECTORY_CMAPS)]
            _plot_gradient_trajectory(
                ax,
                traj,
                label=f"轨迹{index + 1}",
                linestyle=["-", "--"][index % 2],
                linewidth=1.4,
                cmap=cmap,
                norm=norm,
            )
        ax.set_title("未分类轨迹")
        ax.set_xlabel("世界坐标 X (m)")
        ax.set_ylabel("世界坐标 Y (m)")
        ax.set_zlabel("世界坐标 Z (m)")
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.grid(True, alpha=0.35)
        _set_equal_data_axes(ax, unknown_trajectories[:2])
        fig.tight_layout()
        png_path = output_dir / "unknown_tip_trajectories_3d.png"
        pdf_path = output_dir / "unknown_tip_trajectories_3d.pdf"
        fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append((png_path, pdf_path))

    return saved_paths


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir, files = discover_trajectory_files(args)
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = [read_trajectory(path) for path in files]
    stats_path = output_dir / "tip_trajectory_stats.csv"
    write_stats_csv(stats_path, trajectories)
    saved_paths = plot_grouped_trajectories(
        trajectories,
        output_dir=output_dir,
        png_dpi=args.png_dpi,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        axis_padding_frac=args.axis_padding_frac,
        random_seed=args.random_seed,
    )

    print(f"Read {len(trajectories)} trajectory file(s) from {result_dir / 'trajectories'}")
    for png_path, pdf_path in saved_paths:
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
    print(f"Saved: {stats_path}")
    print("Coordinate frame: tip_pos_w_x/y/z are world-frame positions in meters.")


if __name__ == "__main__":
    main()
