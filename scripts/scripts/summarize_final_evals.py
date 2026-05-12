#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


FIELDS = (
    "return",
    "episode_length",
    "absorbed_ratio_final",
    "ur3_contact_force_max",
    "tip_goal_error_mean",
    "tip_pipe_clearance_mean",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final evaluation JSON files across paper seeds.")
    parser.add_argument("--input-root", type=Path, default=Path("logs/paper_final_eval"), help="Eval output root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/comparisons/paper_experiments_800k/final_eval"),
        help="Directory for final_policy_seed_summary.csv and final_policy_group_summary.csv.",
    )
    return parser.parse_args()


def _summary_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*/summary.json") if path.is_file())


def _field_mean(summary: dict, field: str) -> float:
    stats = summary.get("aggregates", {}).get(field)
    if not isinstance(stats, dict):
        return float("nan")
    value = stats.get("mean")
    return float(value) if value is not None else float("nan")


def _field_std(summary: dict, field: str) -> float:
    stats = summary.get("aggregates", {}).get(field)
    if not isinstance(stats, dict):
        return float("nan")
    value = stats.get("std")
    return float(value) if value is not None else float("nan")


def _load_rows(input_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in _summary_files(input_root):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        group = summary_path.parent.parent.name
        seed = summary_path.parent.name.replace("seed_", "")
        row: dict[str, object] = {
            "group": group,
            "algorithm": summary.get("algorithm", ""),
            "task": summary.get("task", ""),
            "seed": seed,
            "checkpoint_path": summary.get("checkpoint_path", ""),
            "completed_episodes": summary.get("completed_episodes", 0),
            "success_rate": summary.get("success_rate", ""),
            "severe_collision_rate": summary.get("severe_collision_rate", ""),
            "time_out_rate": summary.get("time_out_rate", ""),
        }
        for field in FIELDS:
            row[f"{field}_mean"] = _field_mean(summary, field)
            row[f"{field}_std"] = _field_std(summary, field)
        rows.append(row)
    return rows


def _write_seed_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_path = output_dir / "final_policy_seed_summary.csv"
    fieldnames = [
        "group",
        "algorithm",
        "task",
        "seed",
        "checkpoint_path",
        "completed_episodes",
        "success_rate",
        "severe_collision_rate",
        "time_out_rate",
    ]
    for field in FIELDS:
        fieldnames.extend([f"{field}_mean", f"{field}_std"])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_group_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
    output_path = output_dir / "final_policy_group_summary.csv"
    groups = sorted({str(row["group"]) for row in rows})
    fieldnames = [
        "group",
        "num_seeds",
        "success_rate_mean",
        "success_rate_std",
        "severe_collision_rate_mean",
        "severe_collision_rate_std",
        "time_out_rate_mean",
        "time_out_rate_std",
    ]
    for field in FIELDS:
        fieldnames.extend([f"{field}_mean_across_seeds", f"{field}_std_across_seeds"])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            group_rows = [row for row in rows if row["group"] == group]
            output_row: dict[str, object] = {"group": group, "num_seeds": len(group_rows)}
            for field in ("success_rate", "severe_collision_rate", "time_out_rate"):
                values = np.asarray([_as_float(row.get(field)) for row in group_rows], dtype=np.float64)
                values = values[np.isfinite(values)]
                output_row[f"{field}_mean"] = float(values.mean()) if values.size else ""
                output_row[f"{field}_std"] = float(values.std()) if values.size else ""
            for field in FIELDS:
                values = np.asarray([_as_float(row.get(f"{field}_mean")) for row in group_rows], dtype=np.float64)
                values = values[np.isfinite(values)]
                output_row[f"{field}_mean_across_seeds"] = float(values.mean()) if values.size else ""
                output_row[f"{field}_std_across_seeds"] = float(values.std()) if values.size else ""
            writer.writerow(output_row)
    return output_path


def main() -> None:
    args = _parse_args()
    input_root = (REPO_ROOT / args.input_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(input_root)
    if not rows:
        raise RuntimeError(f"No summary.json files found under: {input_root}")
    seed_path = _write_seed_summary(rows, output_dir)
    group_path = _write_group_summary(rows, output_dir)
    print(f"[INFO] Saved: {seed_path}")
    print(f"[INFO] Saved: {group_path}")


if __name__ == "__main__":
    main()
