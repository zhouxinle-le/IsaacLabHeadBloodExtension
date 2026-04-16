from __future__ import annotations

import numpy as np


def rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_vec = quat_wxyz[1:]
    uv = np.cross(quat_vec, vec)
    uuv = np.cross(quat_vec, uv)
    return vec + 2.0 * (quat_wxyz[0] * uv + uuv)


def compute_tip_pose_numpy(
    tip_body_pos_w: np.ndarray,
    tip_local_offset: np.ndarray,
    tip_local_axis: np.ndarray,
    body_quat_w: np.ndarray | None = None,
    use_body_quat_for_tip_dir: bool = True,
    env_origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tip_pos = np.asarray(tip_body_pos_w, dtype=np.float32) + np.asarray(tip_local_offset, dtype=np.float32)
    tip_dir = np.asarray(tip_local_axis, dtype=np.float32).copy()

    if body_quat_w is not None:
        quat_wxyz = np.asarray(body_quat_w, dtype=np.float32)
        tip_pos = np.asarray(tip_body_pos_w, dtype=np.float32) + rotate_vector_by_quat(quat_wxyz, tip_local_offset)
        if use_body_quat_for_tip_dir:
            tip_dir = rotate_vector_by_quat(quat_wxyz, tip_local_axis)

    tip_dir = tip_dir / (np.linalg.norm(tip_dir) + 1.0e-9)
    if env_origin is not None:
        tip_pos = tip_pos - np.asarray(env_origin, dtype=np.float32)

    return tip_pos.astype(np.float32, copy=False), tip_dir.astype(np.float32, copy=False)


def compute_particle_relation(
    particles_pos: np.ndarray,
    tip_pos: np.ndarray,
    tip_dir: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    relative_positions = np.asarray(particles_pos, dtype=np.float32) - np.asarray(tip_pos, dtype=np.float32)
    distances = np.linalg.norm(relative_positions, axis=1)
    axial_depth = np.dot(relative_positions, np.asarray(tip_dir, dtype=np.float32))
    radial_offset = relative_positions - np.outer(axial_depth, tip_dir)
    radial_distance = np.linalg.norm(radial_offset, axis=1)
    return relative_positions, distances, axial_depth, radial_distance


def compute_cone_and_inlet_masks(
    distances: np.ndarray,
    axial_depth: np.ndarray,
    radial_distance: np.ndarray,
    valid_mask: np.ndarray,
    suction_radius: float,
    cos_theta: float,
    inlet_depth: float,
    inlet_radius: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    cos_alpha = axial_depth / (np.asarray(distances, dtype=np.float32) + float(epsilon))
    in_cone = (distances < suction_radius) & (cos_alpha >= cos_theta) & valid_mask
    in_inlet = (
        (axial_depth > 0.0)
        & (axial_depth < inlet_depth)
        & (radial_distance < inlet_radius)
        & valid_mask
    )
    return in_cone, in_inlet
