#!/usr/bin/env python3
"""Plot wrist-vision blood particle initialization templates."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = (
    REPO_ROOT
    / "exts/head_blood_absorption/head_blood_absorption/tasks/blood_pipe_vision_ur3/"
    / "ur3_head_blood_pipe_env_wrist.py"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
FIGURE_DPI = 300
PARTICLE_COLOR = "#D13F3F"
GRID_COLOR = "#E3E3E3"


@dataclass(frozen=True)
class TemplateData:
    index: int
    path: Path
    positions: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Environment file containing blood_init_template_files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and CSV output.",
    )
    parser.add_argument("--png-dpi", type=int, default=FIGURE_DPI)
    parser.add_argument("--view-elev", type=float, default=22.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _configure_matplotlib() -> None:
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
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": FIGURE_DPI,
        }
    )


def _extract_template_files(env_file: Path) -> tuple[str, ...]:
    text = env_file.read_text(encoding="utf-8")
    match = re.search(r"blood_init_template_files\s*=\s*\((.*?)\)", text, flags=re.S)
    if match is None:
        raise ValueError(f"Cannot find blood_init_template_files in {env_file}")
    names = tuple(re.findall(r"[\"']([^\"']+)[\"']", match.group(1)))
    if not names:
        raise ValueError(f"blood_init_template_files is empty in {env_file}")
    return names


def _asset_dir_from_env_file(env_file: Path) -> Path:
    # Matches Ur3BloodPipeVisionWristEnvCfg.ASSET_PATH:
    # os.path.join(os.path.dirname(CURRENT_PATH), "blood_pipe_state", "usd_models")
    return env_file.parent.parent / "blood_pipe_state" / "usd_models"


def _load_torch_template(path: Path) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to load .pt templates. Run with: "
            "conda run -n isaacsim-4.2 python scripts/scripts_4/particle_templates/"
            "plot_wrist_particle_templates.py"
        ) from exc

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, dict):
        if "positions" in loaded:
            loaded = loaded["positions"]
        elif "particles_pos" in loaded:
            loaded = loaded["particles_pos"]
        else:
            raise ValueError(f"{path} is a dict but has no 'positions' or 'particles_pos' key.")

    positions = np.asarray(torch.as_tensor(loaded, dtype=torch.float32).cpu().numpy(), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{path} must contain positions with shape (N, 3), got {positions.shape}.")
    if positions.shape[0] <= 0:
        raise ValueError(f"{path} contains no particles.")
    return positions


def _load_templates(env_file: Path) -> list[TemplateData]:
    asset_dir = _asset_dir_from_env_file(env_file)
    templates: list[TemplateData] = []
    for index, name in enumerate(_extract_template_files(env_file), start=1):
        template_path = Path(name)
        if not template_path.suffix:
            template_path = template_path.with_suffix(".pt")
        if not template_path.is_absolute():
            template_path = asset_dir / template_path
        template_path = template_path.resolve()
        if not template_path.is_file():
            raise FileNotFoundError(f"Particle template file not found: {template_path}")
        templates.append(TemplateData(index=index, path=template_path, positions=_load_torch_template(template_path)))
    return templates


def _set_equal_axes(ax, all_positions: np.ndarray) -> None:
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.005)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _plot_one_template(
    template: TemplateData,
    all_positions: np.ndarray,
    output_dir: Path,
    png_dpi: int,
    view_elev: float,
    view_azim: float,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(4.8, 4.0), dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")
    positions = template.positions
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=PARTICLE_COLOR,
        s=200,
        alpha=0.9,
        depthshade=True,
    )
    ax.set_title(f"模板 {template.index}: 总粒子数 {template.count}", pad=2, y=1.00)
    ax.set_xlabel("X (m)", labelpad=5)
    ax.set_ylabel("Y (m)", labelpad=5)
    ax.set_zlabel("Z (m)", labelpad=5)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.8)
    _set_equal_axes(ax, all_positions)

    fig.tight_layout()
    output_base = output_dir / f"wrist_particle_template_{template.index:02d}_{template.count}_particles"
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_templates(
    templates: list[TemplateData],
    output_dir: Path,
    png_dpi: int,
    view_elev: float,
    view_azim: float,
) -> list[tuple[Path, Path]]:
    all_positions = np.concatenate([template.positions for template in templates], axis=0)
    return [
        _plot_one_template(
            template,
            all_positions=all_positions,
            output_dir=output_dir,
            png_dpi=png_dpi,
            view_elev=view_elev,
            view_azim=view_azim,
        )
        for template in templates
    ]


def _write_csv(templates: list[TemplateData], output_path: Path) -> None:
    fieldnames = ["template_index", "template_file", "particle_count", "particle_index", "x", "y", "z"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for template in templates:
            for particle_index, position in enumerate(template.positions):
                writer.writerow(
                    {
                        "template_index": template.index,
                        "template_file": str(template.path),
                        "particle_count": template.count,
                        "particle_index": particle_index,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                    }
                )


def main() -> None:
    _configure_matplotlib()
    args = _parse_args()
    env_file = _resolve(args.env_file)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = _load_templates(env_file)
    figure_paths = _plot_templates(
        templates,
        output_dir=output_dir,
        png_dpi=args.png_dpi,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    csv_path = output_dir / "wrist_particle_templates.csv"
    _write_csv(templates, csv_path)

    print(f"[INFO] Env file: {env_file}")
    for template in templates:
        mins = template.positions.min(axis=0)
        maxs = template.positions.max(axis=0)
        print(
            f"[INFO] Template {template.index}: {template.path.name}, "
            f"particles={template.count}, min={mins.tolist()}, max={maxs.tolist()}"
        )
    for png_path, pdf_path in figure_paths:
        print(f"[INFO] Saved: {png_path}")
        print(f"[INFO] Saved: {pdf_path}")
    print(f"[INFO] Saved: {csv_path}")


if __name__ == "__main__":
    main()
