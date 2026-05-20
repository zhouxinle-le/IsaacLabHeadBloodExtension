#!/usr/bin/env python3
"""Plot 3D suction tip trajectories with an endpoint zoom subplot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from plot_tip_trajectories_3d import (
    TRAJECTORY_BASE_COLORS,
    TRAJECTORY_CMAPS,
    configure_matplotlib,
    discover_trajectory_files,
    read_trajectory,
    set_axes_equal,
    _plot_gradient_trajectory,
    _plot_points,
    _select_one_trajectory_per_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("scripts/scripts_3/results/20260513_185547_vision_ppo_replay"),
        help="Replay result directory containing trajectories/.",
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
        help="Directory for figures. Defaults to RESULT_DIR/trajectory_plots.",
    )
    parser.add_argument(
        "--outcome",
        choices=("success", "failure"),
        default="failure",
        help="Trajectory outcome to plot.",
    )
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    parser.add_argument(
        "--zoom-points",
        type=int,
        default=240,
        help="Number of final trajectory points shown in the zoom subplot.",
    )
    parser.add_argument(
        "--zoom-frac",
        type=float,
        default=0.18,
        help="Fallback fraction of final trajectory points if it is larger than --zoom-points.",
    )
    parser.add_argument(
        "--zoom-padding-frac",
        type=float,
        default=0.18,
        help="Fractional padding around the zoomed endpoint region.",
    )
    return parser.parse_args()


def _outcome_title(outcome: str) -> str:
    return "成功轨迹" if outcome == "success" else "失败轨迹"


def _selected_trajectories(trajectories, outcome: str):
    candidates = [traj for traj in trajectories if traj.outcome == outcome]
    selected = _select_one_trajectory_per_seed(candidates)
    if not selected:
        raise ValueError(f"No {outcome} trajectories found.")
    return selected[:2]


def _endpoint_points(traj, zoom_points: int, zoom_frac: float) -> np.ndarray:
    points = _plot_points(traj)
    if points.size == 0:
        return points
    count_from_points = max(2, int(round(len(points) * max(0.0, zoom_frac))))
    count = max(int(zoom_points), count_from_points)
    count = min(len(points), count)
    return points[-count:]


def _set_axes_from_points(ax, points: np.ndarray, padding_frac: float) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    padding = spans * max(0.0, padding_frac)
    ax.set_xlim(float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    ax.set_ylim(float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    ax.set_zlim(float(mins[2] - padding[2]), float(maxs[2] + padding[2]))
    set_axes_equal(ax)


def _set_full_axes(ax, trajectories) -> None:
    points = np.concatenate([_plot_points(traj) for traj in trajectories], axis=0)
    ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
    set_axes_equal(ax)


def _style_3d_axes(ax) -> None:
    ax.set_xlabel("世界坐标 X (m)")
    ax.set_ylabel("世界坐标 Y (m)")
    ax.set_zlabel("世界坐标 Z (m)")
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=6.2, pad=1)


def plot_zoom_figure(
    trajectories,
    outcome: str,
    output_dir: Path,
    png_dpi: int,
    view_elev: float,
    view_azim: float,
    zoom_points: int,
    zoom_frac: float,
    zoom_padding_frac: float,
) -> tuple[Path, Path]:
    selected = _selected_trajectories(trajectories, outcome)
    norm = Normalize(vmin=0.0, vmax=1.0)
    styles = ["-", "--"]

    fig = plt.figure(figsize=(8.6, 4.0))
    ax_full = fig.add_subplot(121, projection="3d")
    ax_zoom = fig.add_subplot(122, projection="3d")

    zoom_point_sets = []
    for index, traj in enumerate(selected):
        cmap = TRAJECTORY_CMAPS[index % len(TRAJECTORY_CMAPS)]
        linestyle = styles[index % len(styles)]
        _plot_gradient_trajectory(
            ax_full,
            traj,
            label=f"轨迹{index + 1}",
            linestyle=linestyle,
            linewidth=1.25,
            cmap=cmap,
            norm=norm,
        )

        points = _endpoint_points(traj, zoom_points=zoom_points, zoom_frac=zoom_frac)
        zoom_point_sets.append(points)
        color = TRAJECTORY_BASE_COLORS[index % len(TRAJECTORY_BASE_COLORS)]
        ax_zoom.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=f"轨迹{index + 1}",
        )
        ax_zoom.scatter(points[0, 0], points[0, 1], points[0, 2], color=color, marker="o", s=18, alpha=0.75)
        ax_zoom.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, marker="^", s=26, alpha=0.95)

    title = _outcome_title(outcome)
    ax_full.set_title(f"{title}全局视图", y=1.00)
    ax_zoom.set_title("末端密集区域放大", y=1.00)
    for ax in (ax_full, ax_zoom):
        _style_3d_axes(ax)
        ax.view_init(elev=view_elev, azim=view_azim)

    _set_full_axes(ax_full, selected)
    _set_axes_from_points(ax_zoom, np.concatenate(zoom_point_sets, axis=0), zoom_padding_frac)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRAJECTORY_BASE_COLORS[index % len(TRAJECTORY_BASE_COLORS)],
            linestyle=styles[index % len(styles)],
            linewidth=1.5,
            label=f"轨迹{index + 1}",
        )
        for index in range(len(selected))
    ]
    ax_full.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.94, 0.90), frameon=True)
    ax_zoom.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.96, 0.92), frameon=True)

    fig.tight_layout(w_pad=0.6)
    png_path = output_dir / f"{outcome}_tip_trajectories_3d_with_endpoint_zoom.png"
    pdf_path = output_dir / f"{outcome}_tip_trajectories_3d_with_endpoint_zoom.pdf"
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    for index, traj in enumerate(selected, start=1):
        print(f"Selected {outcome} 轨迹{index}: seed={traj.seed}, file={traj.path.name}")
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir, files = discover_trajectory_files(args)
    output_dir = args.output_dir.resolve() if args.output_dir else result_dir / "trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = [read_trajectory(path) for path in files]
    png_path, pdf_path = plot_zoom_figure(
        trajectories=trajectories,
        outcome=args.outcome,
        output_dir=output_dir,
        png_dpi=args.png_dpi,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
        zoom_points=args.zoom_points,
        zoom_frac=args.zoom_frac,
        zoom_padding_frac=args.zoom_padding_frac,
    )
    print(f"Read {len(trajectories)} trajectory file(s) from {result_dir / 'trajectories'}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
