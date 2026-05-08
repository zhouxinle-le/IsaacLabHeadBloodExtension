#!/usr/bin/env python3
"""Extract pipe geometry parameters from a USD model.

This is a standalone inspection tool. It does not import or modify the task
environment. Run it with a Python interpreter that can import ``pxr``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_USD_PATH = (
    "exts/head_blood_absorption/head_blood_absorption/tasks/"
    "blood_pipe_state/usd_models/head_pipe_1_2.usd"
)


def _import_pxr():
    try:
        from pxr import Gf, Usd, UsdGeom
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Failed to import pxr. Run this script with Isaac Sim/Isaac Lab Python, "
            "or set PYTHONPATH/LD_LIBRARY_PATH to an OpenUSD installation."
        ) from exc
    return Gf, Usd, UsdGeom


Gf, Usd, UsdGeom = _import_pxr()


@dataclass(frozen=True)
class PipeLocalMesh:
    points: list[tuple[float, float, float]]
    face_vertex_counts: list[int]
    face_vertex_indices: list[int]


def _find_unique_descendant_by_name(root_prim: Usd.Prim, prim_name: str) -> Usd.Prim:
    matches = [prim for prim in Usd.PrimRange(root_prim) if prim.GetName() == prim_name]
    if len(matches) == 0:
        raise RuntimeError(f"Could not find prim named '{prim_name}' under '{root_prim.GetPath()}'.")
    if len(matches) > 1:
        paths = ", ".join(str(prim.GetPath()) for prim in matches)
        raise RuntimeError(f"Found multiple prims named '{prim_name}': {paths}")
    return matches[0]


def _find_first_descendant_by_names(root_prim: Usd.Prim, names: Iterable[str]) -> Usd.Prim | None:
    wanted = set(names)
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() in wanted:
            return prim
    return None


def _quat_wxyz_from_transform(transform) -> tuple[float, float, float, float]:
    rotation = transform.ExtractRotationQuat()
    imag = rotation.GetImaginary()
    quat = (float(rotation.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
    length = math.sqrt(sum(value * value for value in quat))
    if length <= 1.0e-12:
        raise RuntimeError("Encountered a near-zero quaternion while extracting pipe pose.")
    return tuple(value / length for value in quat)


def _collect_pipe_local_meshes(
    stage: Usd.Stage,
    pipe_prim: Usd.Prim,
) -> tuple[list[PipeLocalMesh], list[tuple[float, float, float]], int]:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pipe_to_world = xform_cache.GetLocalToWorldTransform(pipe_prim)
    pipe_world_to_local = pipe_to_world.GetInverse()

    meshes: list[PipeLocalMesh] = []
    points_pipe: list[tuple[float, float, float]] = []
    mesh_count = 0
    for prim in Usd.PrimRange(pipe_prim):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        if mesh_points is None:
            continue
        face_vertex_counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
        face_vertex_indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
        mesh_count += 1
        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        mesh_points_pipe: list[tuple[float, float, float]] = []
        for point in mesh_points:
            point_w = mesh_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            point_pipe = pipe_world_to_local.Transform(point_w)
            point_tuple = (float(point_pipe[0]), float(point_pipe[1]), float(point_pipe[2]))
            mesh_points_pipe.append(point_tuple)
            points_pipe.append(point_tuple)
        meshes.append(
            PipeLocalMesh(
                points=mesh_points_pipe,
                face_vertex_counts=face_vertex_counts,
                face_vertex_indices=face_vertex_indices,
            )
        )

    if len(points_pipe) == 0:
        raise RuntimeError(f"Pipe prim '{pipe_prim.GetPath()}' does not contain any mesh points.")
    return meshes, points_pipe, mesh_count


def _local_bbox(points: list[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(min(point[axis] for point in points) for axis in range(3)),
        tuple(max(point[axis] for point in points) for axis in range(3)),
    )


def _cluster_sorted_values(values: list[float], tolerance: float) -> list[dict[str, float | int]]:
    if not values:
        return []

    sorted_values = sorted(values)
    clusters: list[list[float]] = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    result = []
    for cluster in clusters:
        result.append(
            {
                "min": min(cluster),
                "max": max(cluster),
                "mean": sum(cluster) / len(cluster),
                "count": len(cluster),
            }
        )
    return result


def _deduplicate_points(
    points: list[tuple[float, float, float]],
    tolerance: float,
) -> list[tuple[float, float, float]]:
    unique: list[tuple[float, float, float]] = []
    tolerance_sq = tolerance * tolerance
    for point in points:
        if any(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 + (point[2] - other[2]) ** 2
            <= tolerance_sq
            for other in unique
        ):
            continue
        unique.append(point)
    return unique


def _intersect_edge_with_z_plane(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    z_plane: float,
    eps: float,
) -> tuple[float, float, float] | None:
    za = point_a[2] - z_plane
    zb = point_b[2] - z_plane
    if abs(za) <= eps and abs(zb) <= eps:
        return None
    if abs(za) <= eps:
        return point_a
    if abs(zb) <= eps:
        return point_b
    if za * zb > 0.0:
        return None

    t = (z_plane - point_a[2]) / (point_b[2] - point_a[2])
    return (
        point_a[0] + t * (point_b[0] - point_a[0]),
        point_a[1] + t * (point_b[1] - point_a[1]),
        z_plane,
    )


def _section_points_from_meshes(
    meshes: list[PipeLocalMesh],
    z_plane: float,
    *,
    eps: float = 1.0e-9,
) -> list[tuple[float, float, float]]:
    section_points: list[tuple[float, float, float]] = []
    for mesh in meshes:
        if not mesh.face_vertex_counts or not mesh.face_vertex_indices:
            continue

        cursor = 0
        for face_vertex_count in mesh.face_vertex_counts:
            face_indices = mesh.face_vertex_indices[cursor : cursor + face_vertex_count]
            cursor += face_vertex_count
            if len(face_indices) < 3:
                continue

            face_points = [mesh.points[index] for index in face_indices]
            face_section_points: list[tuple[float, float, float]] = []
            for index, point_a in enumerate(face_points):
                point_b = face_points[(index + 1) % len(face_points)]
                intersection = _intersect_edge_with_z_plane(point_a, point_b, z_plane, eps)
                if intersection is not None:
                    face_section_points.append(intersection)

            section_points.extend(_deduplicate_points(face_section_points, tolerance=1.0e-8))

    return _deduplicate_points(section_points, tolerance=1.0e-7)


def _estimate_radii(
    meshes: list[PipeLocalMesh],
    points: list[tuple[float, float, float]],
    pipe_min: tuple[float, float, float],
    pipe_max: tuple[float, float, float],
    *,
    z_trim_fraction: float,
    cluster_tolerance: float,
) -> dict[str, object]:
    pipe_length = pipe_max[2] - pipe_min[2]
    z_trim = max(pipe_length * z_trim_fraction, 0.0)
    z_low = pipe_min[2] + z_trim
    z_high = pipe_max[2] - z_trim
    z_plane = 0.5 * (z_low + z_high)

    section_points = _section_points_from_meshes(meshes, z_plane)
    used_fallback_vertices = False
    if len(section_points) < 4:
        section_points = [point for point in points if z_low <= point[2] <= z_high]
        if len(section_points) < 4:
            section_points = points
            used_fallback_vertices = True

    radii = [math.hypot(point[0], point[1]) for point in section_points]
    nonzero_radii = [radius for radius in radii if radius > 1.0e-8]
    if len(nonzero_radii) == 0:
        raise RuntimeError("Could not estimate radii because all pipe-local radii are zero.")

    clusters = _cluster_sorted_values(nonzero_radii, cluster_tolerance)
    min_cluster_count = max(4, int(0.01 * len(nonzero_radii)))
    significant_clusters = [cluster for cluster in clusters if int(cluster["count"]) >= min_cluster_count]
    if len(significant_clusters) == 0:
        significant_clusters = clusters

    inner_cluster = min(significant_clusters, key=lambda cluster: float(cluster["mean"]))
    outer_cluster = max(significant_clusters, key=lambda cluster: float(cluster["mean"]))

    return {
        "radial_sample_count": len(nonzero_radii),
        "radial_z_trim_fraction": z_trim_fraction,
        "radial_z_range_used": (z_low, z_high),
        "radial_z_plane_used": z_plane,
        "radial_used_fallback_vertices": used_fallback_vertices,
        "inner_radius_estimate": float(inner_cluster["mean"]),
        "outer_radius_estimate": float(outer_cluster["mean"]),
        "radius_min": min(nonzero_radii),
        "radius_max": max(nonzero_radii),
        "radius_clusters": significant_clusters,
    }


def _marker_local_position(pipe_prim: Usd.Prim, marker_prim: Usd.Prim) -> tuple[float, float, float]:
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pipe_to_world = xform_cache.GetLocalToWorldTransform(pipe_prim)
    marker_to_world = xform_cache.GetLocalToWorldTransform(marker_prim)
    marker_to_pipe = marker_to_world * pipe_to_world.GetInverse()
    pos = marker_to_pipe.ExtractTranslation()
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _read_authored_numeric_attrs(prim: Usd.Prim) -> dict[str, float]:
    keywords = ("radius", "length", "margin", "diameter", "inner", "outer")
    found: dict[str, float] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not any(keyword in name.lower() for keyword in keywords):
            continue
        if not attr.HasAuthoredValueOpinion():
            continue
        value = attr.Get()
        if isinstance(value, (int, float)):
            found[name] = float(value)
    return found


def extract_pipe_params(args: argparse.Namespace) -> dict[str, object]:
    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file '{args.usd_path}'.")

    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"USD file '{args.usd_path}' does not define a valid default prim.")

    pipe_prim = _find_unique_descendant_by_name(root_prim, args.pipe_prim_name)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_to_world = xform_cache.GetLocalToWorldTransform(root_prim)
    pipe_to_world = xform_cache.GetLocalToWorldTransform(pipe_prim)
    pipe_to_root = pipe_to_world * root_to_world.GetInverse()

    meshes, points_pipe, mesh_count = _collect_pipe_local_meshes(stage, pipe_prim)
    pipe_min, pipe_max = _local_bbox(points_pipe)
    length_from_mesh = pipe_max[2] - pipe_min[2]
    if length_from_mesh <= 1.0e-9:
        raise RuntimeError(f"Pipe prim '{pipe_prim.GetPath()}' has non-positive local z length.")

    radii = _estimate_radii(
        meshes,
        points_pipe,
        pipe_min,
        pipe_max,
        z_trim_fraction=args.radial_z_trim_fraction,
        cluster_tolerance=args.radius_cluster_tolerance,
    )

    marker_names = {
        "inner_radius": args.inner_radius_marker,
        "bottom": args.bottom_marker,
        "top": args.top_marker,
    }
    marker_positions: dict[str, tuple[float, float, float]] = {}
    for label, marker_name in marker_names.items():
        marker_prim = None
        if marker_name:
            marker_prim = _find_unique_descendant_by_name(pipe_prim, marker_name)
        elif label == "inner_radius":
            marker_prim = _find_first_descendant_by_names(
                pipe_prim,
                ("inner_radius_point", "pipe_inner_radius_point", "InnerRadiusPoint"),
            )
        elif label == "bottom":
            marker_prim = _find_first_descendant_by_names(pipe_prim, ("bottom_limit", "pipe_bottom_limit"))
        elif label == "top":
            marker_prim = _find_first_descendant_by_names(pipe_prim, ("top_limit", "pipe_top_limit"))

        if marker_prim is not None:
            marker_positions[label] = _marker_local_position(pipe_prim, marker_prim)

    inner_radius_from_marker = None
    if "inner_radius" in marker_positions:
        marker = marker_positions["inner_radius"]
        inner_radius_from_marker = math.hypot(marker[0], marker[1])

    length_from_markers = None
    if "bottom" in marker_positions and "top" in marker_positions:
        length_from_markers = abs(marker_positions["top"][2] - marker_positions["bottom"][2])

    return {
        "usd_path": str(args.usd_path),
        "default_prim": str(root_prim.GetPath()),
        "pipe_prim_path": str(pipe_prim.GetPath()),
        "pipe_link_local_pos": tuple(float(value) for value in pipe_to_root.ExtractTranslation()),
        "pipe_link_local_quat": _quat_wxyz_from_transform(pipe_to_root),
        "pipe_local_min": pipe_min,
        "pipe_local_max": pipe_max,
        "pipe_length_from_mesh": length_from_mesh,
        "pipe_length_from_markers": length_from_markers,
        "pipe_inner_radius_from_marker": inner_radius_from_marker,
        "pipe_inner_radius_estimate": radii["inner_radius_estimate"],
        "pipe_outer_radius_estimate": radii["outer_radius_estimate"],
        "radius_min": radii["radius_min"],
        "radius_max": radii["radius_max"],
        "radius_clusters": radii["radius_clusters"],
        "radial_sample_count": radii["radial_sample_count"],
        "radial_z_range_used": radii["radial_z_range_used"],
        "radial_z_plane_used": radii["radial_z_plane_used"],
        "radial_used_fallback_vertices": radii["radial_used_fallback_vertices"],
        "mesh_count": mesh_count,
        "point_count": len(points_pipe),
        "marker_positions_pipe_local": marker_positions,
        "authored_numeric_attrs": _read_authored_numeric_attrs(pipe_prim),
    }


def _format_tuple(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.12g}" for value in values) + ")"


def print_report(params: dict[str, object]) -> None:
    print("Pipe USD parameter report")
    print("=========================")
    print(f"USD: {params['usd_path']}")
    print(f"Pipe prim: {params['pipe_prim_path']}")
    print(f"Meshes / points: {params['mesh_count']} / {params['point_count']}")
    print()
    print("Strict mesh-derived values:")
    print(f"  pipe_link_local_pos  = {_format_tuple(params['pipe_link_local_pos'])}")
    print(f"  pipe_link_local_quat = {_format_tuple(params['pipe_link_local_quat'])}")
    print(f"  pipe_local_min       = {_format_tuple(params['pipe_local_min'])}")
    print(f"  pipe_local_max       = {_format_tuple(params['pipe_local_max'])}")
    print(f"  pipe_length          = {float(params['pipe_length_from_mesh']):.12g}")
    print()
    print("Radius estimates from pipe-local z-plane mesh section:")
    print(f"  z_range_used          = {_format_tuple(params['radial_z_range_used'])}")
    print(f"  z_plane_used          = {float(params['radial_z_plane_used']):.12g}")
    print(f"  section_sample_count  = {int(params['radial_sample_count'])}")
    print(f"  used_vertex_fallback  = {bool(params['radial_used_fallback_vertices'])}")
    print(f"  inner_radius_estimate = {float(params['pipe_inner_radius_estimate']):.12g}")
    print(f"  outer_radius_estimate = {float(params['pipe_outer_radius_estimate']):.12g}")
    print(f"  radius_min/max        = {float(params['radius_min']):.12g} / {float(params['radius_max']):.12g}")
    print("  significant radius clusters:")
    for cluster in params["radius_clusters"]:
        print(
            "    "
            f"mean={float(cluster['mean']):.12g}, "
            f"min={float(cluster['min']):.12g}, "
            f"max={float(cluster['max']):.12g}, "
            f"count={int(cluster['count'])}"
        )
    print()

    if params["pipe_inner_radius_from_marker"] is not None or params["pipe_length_from_markers"] is not None:
        print("Marker-derived values:")
        if params["pipe_inner_radius_from_marker"] is not None:
            print(f"  pipe_inner_radius = {float(params['pipe_inner_radius_from_marker']):.12g}")
        if params["pipe_length_from_markers"] is not None:
            print(f"  pipe_length = {float(params['pipe_length_from_markers']):.12g}")
        print()

    if params["authored_numeric_attrs"]:
        print("Authored numeric attributes with geometry-like names:")
        for name, value in sorted(params["authored_numeric_attrs"].items()):
            print(f"  {name} = {value:.12g}")
        print()

    inner_radius = params["pipe_inner_radius_from_marker"] or params["pipe_inner_radius_estimate"]
    pipe_length = params["pipe_length_from_markers"] or params["pipe_length_from_mesh"]
    print("Config snippet candidate:")
    print(f"    pipe_link_local_pos = {_format_tuple(params['pipe_link_local_pos'])}")
    print(f"    pipe_link_local_quat = {_format_tuple(params['pipe_link_local_quat'])}")
    print(f"    pipe_length = {float(pipe_length):.12g}")
    print(f"    pipe_inner_radius = {float(inner_radius):.12g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "usd_path",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_USD_PATH),
        help=f"Pipe USD path. Defaults to {DEFAULT_USD_PATH}",
    )
    parser.add_argument("--pipe-prim-name", default="pipe_Link", help="Name of the pipe link prim to inspect.")
    parser.add_argument(
        "--inner-radius-marker",
        default=None,
        help="Optional marker prim name whose pipe-local xy distance defines inner radius.",
    )
    parser.add_argument(
        "--bottom-marker",
        default=None,
        help="Optional marker prim name whose pipe-local z defines the bottom limit.",
    )
    parser.add_argument(
        "--top-marker",
        default=None,
        help="Optional marker prim name whose pipe-local z defines the top limit.",
    )
    parser.add_argument(
        "--radial-z-trim-fraction",
        type=float,
        default=0.20,
        help="Trim this fraction of pipe length near each end, then use the midpoint as the section plane.",
    )
    parser.add_argument(
        "--radius-cluster-tolerance",
        type=float,
        default=2.5e-4,
        help="Absolute tolerance in meters for grouping mesh vertex radii.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report.")
    args = parser.parse_args()

    params = extract_pipe_params(args)
    if args.json:
        print(json.dumps(params, indent=2, sort_keys=True))
    else:
        print_report(params)


if __name__ == "__main__":
    main()
