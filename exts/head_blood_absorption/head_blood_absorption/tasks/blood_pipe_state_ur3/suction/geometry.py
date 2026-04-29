from __future__ import annotations

import numpy as np


def normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    return quat / (np.linalg.norm(quat) + 1.0e-9)


def invert_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat_wxyz)
    return np.array((quat[0], -quat[1], -quat[2], -quat[3]), dtype=np.float32)


def multiply_quat(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lhs = normalize_quat(lhs_wxyz)
    rhs = normalize_quat(rhs_wxyz)
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return normalize_quat(
        np.array(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dtype=np.float32,
        )
    )


def rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_wxyz = normalize_quat(quat_wxyz)
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


def compute_pipe_frame_pose(cfg) -> tuple[np.ndarray, np.ndarray]:
    root_pos = np.asarray(tuple(cfg.pipe.init_state.pos), dtype=np.float32)
    root_quat = normalize_quat(np.asarray(tuple(cfg.pipe.init_state.rot), dtype=np.float32))
    link_pos = np.asarray(tuple(cfg.pipe_link_local_pos), dtype=np.float32)
    link_quat = normalize_quat(np.asarray(tuple(cfg.pipe_link_local_quat), dtype=np.float32))
    pipe_pos = root_pos + rotate_vector_by_quat(root_quat, link_pos)
    pipe_quat = multiply_quat(root_quat, link_quat)
    return pipe_pos.astype(np.float32, copy=False), pipe_quat.astype(np.float32, copy=False)


def pipe_to_env_local(points_pipe: np.ndarray, cfg) -> np.ndarray:
    pipe_pos, pipe_quat = compute_pipe_frame_pose(cfg)
    points = np.asarray(points_pipe, dtype=np.float32)
    flat = points.reshape(-1, 3)
    rotated = np.stack([rotate_vector_by_quat(pipe_quat, point) for point in flat], axis=0)
    return (rotated + pipe_pos).reshape(points.shape).astype(np.float32, copy=False)


def env_local_to_pipe(points_env_local: np.ndarray, cfg) -> np.ndarray:
    pipe_pos, pipe_quat = compute_pipe_frame_pose(cfg)
    inv_pipe_quat = invert_quat(pipe_quat)
    points = np.asarray(points_env_local, dtype=np.float32)
    flat = (points.reshape(-1, 3) - pipe_pos).astype(np.float32, copy=False)
    rotated = np.stack([rotate_vector_by_quat(inv_pipe_quat, point) for point in flat], axis=0)
    return rotated.reshape(points.shape).astype(np.float32, copy=False)


def compute_pipe_membership(points_env_local: np.ndarray, cfg, radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    pipe_pos = env_local_to_pipe(points_env_local, cfg)
    radial_distance = np.linalg.norm(pipe_pos[:, :2], axis=1)
    max_radius = float(cfg.pipe_blood_valid_radius if radius is None else radius)
    z_margin = float(getattr(cfg, "pipe_blood_axis_margin", 0.0))
    valid_mask = (
        (pipe_pos[:, 2] >= z_margin)
        & (pipe_pos[:, 2] <= float(cfg.pipe_length) - z_margin)
        & (radial_distance <= max_radius)
    )
    return valid_mask, pipe_pos


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
