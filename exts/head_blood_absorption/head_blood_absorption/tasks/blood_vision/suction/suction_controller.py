from __future__ import annotations

import math

import numpy as np

from .geometry import compute_cone_and_inlet_masks, compute_particle_relation


class SuctionControllerNoTimer:
    """Particle suction and transfer logic."""

    def __init__(self, cfg, num_envs: int):
        self.cfg = cfg
        self._num_envs = int(num_envs)
        self._tip_body_idx: int | None = None

    def reset(self, env_ids) -> None:
        return None

    def step(
        self,
        tip_pos_local_np: np.ndarray,
        tip_dir_w_np: np.ndarray,
        liquid,
        glass2_pos_np: np.ndarray,
        env_origins_np: np.ndarray,
        apply_suction_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        num_envs = self._num_envs

        absorbed_delta = np.zeros((num_envs,), dtype=np.float32)
        min_dist = np.full((num_envs,), float(self.cfg.suction_cone_range), dtype=np.float32)
        inlet_count = np.zeros((num_envs,), dtype=np.float32)
        cone_count = np.zeros((num_envs,), dtype=np.float32)
        
        # New stats for ParticleTaskTracker
        blood_centroid_w = np.zeros((num_envs, 3), dtype=np.float32)
        valid_in_cone_ratio = np.zeros((num_envs,), dtype=np.float32)
        valid_in_inlet_ratio = np.zeros((num_envs,), dtype=np.float32)

        dt = float(self.cfg.sim.dt * self.cfg.decimation)
        suction_radius = float(self.cfg.suction_cone_range)
        cos_theta = math.cos(math.radians(float(self.cfg.suction_cone_half_angle_deg)))
        epsilon = max(float(getattr(self.cfg, "suction_epsilon", 1e-6)), 1e-12)
        force_scale = float(getattr(self.cfg, "suction_force_scale", 1.0))
        particle_mass = max(float(self.cfg.liquidCfg.particle_mass), epsilon)

        for env_idx in range(num_envs):
            apply_suction_env = apply_suction_mask[env_idx]
            tip_pos = tip_pos_local_np[env_idx]
            tip_dir = tip_dir_w_np[env_idx]
            
            # Default empty centroid to current tip pos world
            blood_centroid_w[env_idx] = tip_pos + env_origins_np[env_idx]

            particles_pos, particles_vel = liquid.read_particles(env_idx)
            if len(particles_pos) == 0:
                continue

            valid_mask = particles_pos[:, self.cfg.height_axis] >= float(self.cfg.height_limit)
            if not valid_mask.any():
                continue

            valid_positions = particles_pos[valid_mask]
            centroid_local = valid_positions.mean(axis=0)
            blood_centroid_w[env_idx] = centroid_local + env_origins_np[env_idx]
            valid_count = max(valid_positions.shape[0], 1)

            relative_positions, distances, axial_depth, radial_distance = compute_particle_relation(
                particles_pos,
                tip_pos,
                tip_dir,
                epsilon,
            )
            min_dist[env_idx] = float(distances.min())
            in_cone, in_inlet = compute_cone_and_inlet_masks(
                distances=distances,
                axial_depth=axial_depth,
                radial_distance=radial_distance,
                valid_mask=valid_mask,
                suction_radius=suction_radius,
                cos_theta=cos_theta,
                inlet_depth=float(self.cfg.inlet_depth),
                inlet_radius=float(self.cfg.inlet_radius),
                epsilon=epsilon,
            )
            if apply_suction_env:
                in_cone = self._apply_manual_suction(
                    particles_vel=particles_vel,
                    relative_positions=relative_positions,
                    distances=distances,
                    in_cone=in_cone,
                    force_scale=force_scale,
                    particle_mass=particle_mass,
                    epsilon=epsilon,
                    dt=dt,
                )
                
            remove_mask = in_inlet

            in_inlet_sum = float(in_inlet.sum())
            in_cone_sum = float(in_cone.sum())
            
            inlet_count[env_idx] = in_inlet_sum
            cone_count[env_idx] = in_cone_sum
            valid_in_cone_ratio[env_idx] = in_cone_sum / float(valid_count)
            valid_in_inlet_ratio[env_idx] = in_inlet_sum / float(valid_count)

            if apply_suction_env and remove_mask.any():
                absorbed_delta[env_idx] = float(
                    self._transfer_particles(
                        env_idx=env_idx,
                        ready_to_transfer=remove_mask,
                        particles_pos=particles_pos,
                        particles_vel=particles_vel,
                        glass2_pos=glass2_pos_np[env_idx],
                    )
                )

            if apply_suction_env:
                self._limit_particle_speed(particles_vel)
                liquid.write_particles(env_idx, particles_pos, particles_vel)

        return {
            "absorbed_delta": absorbed_delta,
            "min_dist": min_dist,
            "inlet_count": inlet_count,
            "cone_count": cone_count,
            "blood_centroid_w": blood_centroid_w,
            "valid_in_cone_ratio": valid_in_cone_ratio,
            "valid_in_inlet_ratio": valid_in_inlet_ratio,
        }

    def _apply_manual_suction(
        self,
        particles_vel,
        relative_positions,
        distances,
        in_cone,
        force_scale,
        particle_mass,
        epsilon,
        dt,
    ) -> np.ndarray:
        if force_scale <= 0.0 or not in_cone.any():
            return np.zeros(len(particles_vel), dtype=bool)

        idx = np.where(in_cone)[0]
        dist = distances[idx]
        direction = -relative_positions[idx] / (dist[:, None] + epsilon)
        accel_mag = force_scale / particle_mass * (float(self.cfg.suction_cone_range) - dist) ** 2
        accel_mag = np.maximum(accel_mag, 0.0)
        particles_vel[idx] += direction * accel_mag[:, None] * dt
        return in_cone

    def _transfer_particles(
        self,
        env_idx: int,
        ready_to_transfer: np.ndarray,
        particles_pos: np.ndarray,
        particles_vel: np.ndarray,
        glass2_pos: np.ndarray,
    ) -> int:
        indices = np.where(ready_to_transfer)[0]
        random_offsets = np.random.uniform(-0.015, 0.015, (len(indices), 3))
        random_offsets[:, 2] = np.abs(random_offsets[:, 2]) + 0.01

        particles_pos[indices] = (
            glass2_pos + np.array([0.0, 0.0, float(self.cfg.glass2_particle_height)]) + random_offsets
        )
        particles_vel[indices] = np.array([0.0, 0.0, float(self.cfg.outflow_speed)], dtype=np.float32)
        return len(indices)

    def _limit_particle_speed(self, particles_vel: np.ndarray) -> None:
        vel_norm = np.linalg.norm(particles_vel, axis=1, keepdims=True) + 1e-9
        too_fast = vel_norm[:, 0] > float(self.cfg.max_particle_speed)
        if too_fast.any():
            particles_vel[too_fast] = (
                particles_vel[too_fast] / vel_norm[too_fast]
            ) * float(self.cfg.max_particle_speed)
