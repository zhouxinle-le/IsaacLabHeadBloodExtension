from __future__ import annotations

import os

import math
import numpy as np
import re
import torch

import carb
import carb.settings
import omni.isaac.lab.sim as sim_utils
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.physx import acquire_physx_interface
from pxr import Gf, Usd, UsdGeom

from .fluid_object import FluidObject, FluidObjectCfg
from .suction import SuctionControllerNoTimer
from .suction.geometry import pipe_to_env_local
from .task_state import ParticleRewardInputs, ParticleTaskState, ParticleTaskTracker


@configclass
class Ur3BloodPipeAbsorptionEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10
    decimation = 2
    action_space = 3
    state_space = 0
    observation_space = 23

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 300,
        render_interval=2,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_particle_contacts=2**22,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=3.0,
        replicate_physics=False,
    )

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    ASSET_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "blood_pipe_state", "usd_models")

    # spawn_pos_pipe = Gf.Vec3f(0.0, 0.37, 0.0) + Gf.Vec3f(0.0, 0.0, 0.08) + Gf.Vec3f(-0.041581, 0.0, 0.0)
    spawn_pos_pipe = Gf.Vec3f(0.0, 0.37, 0.0) + Gf.Vec3f(0.0, 0.0, 0.09) + Gf.Vec3f(-0.039522, 0.0, 0.0)
    pipe_link_local_pos = (0.0398244, -0.033366002, 0.026559601)
    pipe_link_local_quat = (0.69636397, 0.12278932, 0.12278932, -0.69636397)
    pipe_model_auto_sync = True
    pipe_link_prim_name = "pipe_Link"
    pipe_reference_mesh_length = 0.058088833
    pipe_axis_local = (0.0, 0.0, 1.0)
    pipe_length = 0.058
    pipe_inner_radius = 0.015
    pipe_tool_clearance_margin = 0.003
    pipe_blood_valid_radius = 0.011
    pipe_blood_axis_margin = 0.006
    pipe_blood_template_z_counts = (15, 21) # (15, 21, 28)
    pipe_blood_template_z_start = 0.010
    pipe_blood_template_z_end = 0.045
    pipe_wall_clearance_penalty_weight = 0.05
    spawn_pos_fluid = spawn_pos_pipe + Gf.Vec3f(0.039522, -0.052623, 0.07751)
    spawn_pos_glass2 = Gf.Vec3f(0.0, 0.70, 0.01)
    glass2_particle_height = 0.03

    pipe = AssetBaseCfg(
        prim_path="/World/envs/env_.*/HeadPipe",
        init_state=AssetBaseCfg.InitialStateCfg(pos=spawn_pos_pipe, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_PATH}/head_pipe_1_2.usd",
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    table_setup = UsdFileCfg(
        usd_path=f"{ASSET_PATH}/table.usd",
        scale=(1.0, 1.0, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_pos = Gf.Vec3f(0.0, 0.0, 0.457)
    table_height_offset = 0.914
    ur3_init_pos = (0.1, -0.285, 1.20)
    ur3_base_block_size = (0.30, 0.30, 1.20-0.914)
    ur3_base_block_color = (0.32, 0.32, 0.32)

    ur3_base_block = RigidObjectCfg(
        prim_path="/World/envs/env_.*/UR3BaseBlock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ur3_init_pos[0], ur3_init_pos[1], table_height_offset + 0.5 * ur3_base_block_size[2]),
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=ur3_base_block_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=ur3_base_block_color,
                roughness=0.9,
            ),
        ),
    )

    glass2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Glass2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_glass2, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_PATH}/Tall_Glass.usd",
            semantic_tags=[("class", "Glass2")],
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    liquidCfg = FluidObjectCfg()
    liquidCfg.numParticlesX = 3
    liquidCfg.numParticlesY = 3
    liquidCfg.numParticlesZ = 28
    liquidCfg.density = 1060.0
    liquidCfg.particle_mass = 0.001
    liquidCfg.particleSpacing = 0.004
    liquidCfg.viscosity = 3.5
    blood_init_pos_list = ()
    save_blood_init_template_enabled = False
    save_blood_init_template_name = "pipe_particle_init_pos_28"
    save_blood_init_template_after_steps = 240

    ur3_robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/UR3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/ur3_with_suction.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": -1.589,
                "shoulder_lift_joint": -0.634,
                "elbow_joint": 1.075,
                "wrist_1_joint": 3.889,
                "wrist_2_joint": -1.589,
                "wrist_3_joint": 0.148,
            },
            pos=ur3_init_pos,
        ),
        actuators={
            "ur3": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                effort_limit=12000,
                stiffness=800.0,
                damping=40.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    ik_joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    pipe_action_scale_radial = 0.0008
    pipe_action_scale_axial = 0.0012

    ur3_tip_body_name = "suction_tip"
    ur3_collision_body_names_expr = (
        "(shoulder_link|upper_arm_link|forearm_link|wrist_1_link|wrist_2_link|wrist_3_link|"
        "gripper_base_link|gripper_extension_link|suction_tip)"
    )
    ur3_tip_local_offset = (0.0, -0.00423, 0.0)
    ur3_tip_local_axis = (0.0, -1.0, 0.0)
    tip_contact_force_threshold = 0.5
    suction_cone_half_angle_deg = 60.0
    suction_cone_range = 0.07
    suction_force_scale = 0.02
    suction_epsilon = 1e-6
    inlet_radius = 0.003  # 0.008
    inlet_depth = 0.004
    use_body_quat_for_tip_dir = True
    outflow_speed = 0.02
    max_particle_speed = 0.4

    reward_absorb_weight = 75.0
    centroid_progress_weight = 100.0
    centroid_progress_clip = 0.02
    reward_action_weight = 0.02
    reward_time_penalty = 0.01
    reward_task_complete = 25.0
    reward_collision_force_weight = 0.20
    reward_joint_limit_penalty = 10.0
    absorbed_delta_ema_alpha = 0.2
    severe_contact_force_threshold = 2.0
    severe_contact_patience = 2

    blood_success_ratio = 0.98
    joint_limit_termination_tolerance = 1e-3


class Ur3BloodPipeAbsorptionEnv(DirectRLEnv):
    """低维观测的 UR3 粒子吸取环境."""

    cfg: Ur3BloodPipeAbsorptionEnvCfg

    def __init__(self, cfg: Ur3BloodPipeAbsorptionEnvCfg, render_mode: str | None = None, **kwargs):
        self._tip_contact_sensor = None
        self._ur3_contact_sensors: dict[str, ContactSensor] = {}
        self._ur3_collision_body_names: list[str] = []
        self._ee_jacobi_idx: int | None = None
        self._capture_blood_template_enabled = bool(cfg.save_blood_init_template_enabled)
        self._blood_template_capture_saved = False
        self._capture_blood_reset_state: tuple[np.ndarray, np.ndarray] | None = None
        super().__init__(cfg, render_mode, **kwargs)

        self._init_scene_runtime_state()
        self._init_particle_task_state()
        self._init_episode_stats()
        self._init_robot_control_state()
        self._init_observation_cache()

    def _init_scene_runtime_state(self) -> None:
        self._suction_controller = SuctionControllerNoTimer(cfg=self.cfg, num_envs=self.num_envs)
        self._task_state_dirty = True
        self._reward_cache_dirty = True
        self._done_cache_dirty = True
        self._observation_pending = True

    def _init_particle_task_state(self) -> None:
        self._particle_task_tracker = ParticleTaskTracker(cfg=self.cfg, num_envs=self.num_envs, device=self.device)

    def _build_episode_reward_sums(self) -> dict[str, torch.Tensor]:
        return {
            "absorb_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "centroid_progress_reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "action_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "collision_force_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "wall_clearance_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "joint_limit_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "time_penalty": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "task_complete": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        }

    def _init_episode_stats(self) -> None:
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._severe_contact_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reward_sums = self._build_episode_reward_sums()
        self._episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_joint_limit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_severe_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward_cache = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._terminated_cache = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._truncated_cache = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_robot_control_state(self) -> None:
        self._raw_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)

        self._max_particle_count = float(
            self.cfg.liquidCfg.numParticlesX * self.cfg.liquidCfg.numParticlesY * self.cfg.liquidCfg.numParticlesZ
        )
        self._initial_particle_count = torch.full(
            (self.num_envs,),
            self._max_particle_count,
            dtype=torch.float32,
            device=self.device,
        )
        self._success_threshold = self._initial_particle_count * float(self.cfg.blood_success_ratio)
        self._blood_template_index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self._joint_lower_limits = self._ur3.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self._joint_upper_limits = self._ur3.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        joint_names = list(self._ur3.data.joint_names)
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        self._ik_joint_ids = [joint_name_to_idx[name] for name in self.cfg.ik_joint_names]
        self._ik_joint_lower_limits = self._joint_lower_limits[self._ik_joint_ids]
        self._ik_joint_upper_limits = self._joint_upper_limits[self._ik_joint_ids]
        self._joint_pos_des = self._ur3.data.default_joint_pos[:, self._ik_joint_ids].clone()

        self._ur3_body_name_to_idx, self._ur3_body_name_to_path = self._build_ur3_body_lookup()
        self._tip_body_idx = self._resolve_required_body_idx(self.cfg.ur3_tip_body_name)
        self._resolve_ur3_collision_body_selection()
        self._register_ur3_contact_sensors()
        self._tip_local_offset = torch.tensor(self.cfg.ur3_tip_local_offset, dtype=torch.float32, device=self.device)
        self._tip_local_axis = torch.tensor(self.cfg.ur3_tip_local_axis, dtype=torch.float32, device=self.device)
        self._tip_local_axis = self._tip_local_axis / torch.linalg.vector_norm(self._tip_local_axis).clamp_min(1.0e-9)

        ik_controller_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
        )
        self._ik_controller = DifferentialIKController(ik_controller_cfg, num_envs=self.num_envs, device=self.device)
        self._ur3_entity_cfg = SceneEntityCfg(
            "ur3",
            joint_names=list(self.cfg.ik_joint_names),
            body_names=[self.cfg.ur3_tip_body_name],
        )
        self._ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), dtype=torch.float32, device=self.device)
        self._ee_goal_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._init_pipe_frame_state()
        self._pipe_action_scale = torch.tensor(
            (
                float(self.cfg.pipe_action_scale_radial),
                float(self.cfg.pipe_action_scale_radial),
                float(self.cfg.pipe_action_scale_axial),
            ),
            dtype=torch.float32,
            device=self.device,
        )

    def _init_observation_cache(self) -> None:
        self._obs_state = torch.zeros(
            (self.num_envs, int(self.cfg.observation_space)),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def _particle_state(self) -> ParticleTaskState:
        return self._particle_task_tracker.state

    @staticmethod
    def _gf_vec3_to_tensor(vec: Gf.Vec3f, device: torch.device | str) -> torch.Tensor:
        return torch.tensor((float(vec[0]), float(vec[1]), float(vec[2])), dtype=torch.float32, device=device)

    @staticmethod
    def _normalize_quat_tuple(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(value * value for value in quat_wxyz))
        if norm <= 1.0e-9:
            raise ValueError("Cannot normalize zero-length pipe link quaternion from USD.")
        return tuple(float(value / norm) for value in quat_wxyz)

    @staticmethod
    def _find_unique_descendant_by_name(root_prim: Usd.Prim, prim_name: str) -> Usd.Prim:
        matches = [prim for prim in Usd.PrimRange(root_prim) if prim.GetName() == prim_name]
        if len(matches) != 1:
            paths = ", ".join(str(prim.GetPath()) for prim in matches)
            detail = f" Found matches: {paths}." if paths else ""
            raise RuntimeError(
                f"Expected exactly one prim named '{prim_name}' under '{root_prim.GetPath()}'."
                + detail
            )
        return matches[0]

    @staticmethod
    def _read_pipe_model_metadata(usd_path: str, pipe_link_prim_name: str) -> dict[str, object]:
        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open pipe USD model at '{usd_path}'.")

        root_prim = stage.GetDefaultPrim()
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Pipe USD model '{usd_path}' does not define a valid default prim.")

        pipe_prim = Ur3BloodPipeAbsorptionEnv._find_unique_descendant_by_name(root_prim, pipe_link_prim_name)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        root_to_world = xform_cache.GetLocalToWorldTransform(root_prim)
        pipe_to_world = xform_cache.GetLocalToWorldTransform(pipe_prim)
        pipe_to_root = pipe_to_world * root_to_world.GetInverse()
        pipe_world_to_local = pipe_to_world.GetInverse()

        translation = pipe_to_root.ExtractTranslation()
        rotation = pipe_to_root.ExtractRotationQuat()
        rotation_imag = rotation.GetImaginary()
        pipe_link_quat = Ur3BloodPipeAbsorptionEnv._normalize_quat_tuple(
            (
                float(rotation.GetReal()),
                float(rotation_imag[0]),
                float(rotation_imag[1]),
                float(rotation_imag[2]),
            )
        )

        points_pipe: list[tuple[float, float, float]] = []
        mesh_count = 0
        for prim in Usd.PrimRange(pipe_prim):
            if prim.GetTypeName() != "Mesh":
                continue
            mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if mesh_points is None:
                continue
            mesh_count += 1
            mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
            for point in mesh_points:
                point_w = mesh_to_world.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                point_pipe = pipe_world_to_local.Transform(point_w)
                points_pipe.append((float(point_pipe[0]), float(point_pipe[1]), float(point_pipe[2])))

        if len(points_pipe) == 0:
            raise RuntimeError(
                f"Pipe link prim '{pipe_prim.GetPath()}' in '{usd_path}' does not contain mesh points."
            )

        pipe_min = tuple(min(point[axis] for point in points_pipe) for axis in range(3))
        pipe_max = tuple(max(point[axis] for point in points_pipe) for axis in range(3))
        pipe_length = float(pipe_max[2] - pipe_min[2])
        if pipe_length <= 1.0e-9:
            raise RuntimeError(
                f"Pipe link prim '{pipe_prim.GetPath()}' in '{usd_path}' has non-positive z-axis length."
            )

        return {
            "pipe_prim_path": str(pipe_prim.GetPath()),
            "pipe_link_local_pos": (float(translation[0]), float(translation[1]), float(translation[2])),
            "pipe_link_local_quat": pipe_link_quat,
            "pipe_local_min": pipe_min,
            "pipe_local_max": pipe_max,
            "pipe_length": pipe_length,
            "mesh_count": mesh_count,
            "point_count": len(points_pipe),
        }

    def _sync_pipe_model_parameters_from_usd(self) -> None:
        if getattr(self, "_pipe_model_parameters_synced", False):
            return

        if not bool(getattr(self.cfg, "pipe_model_auto_sync", True)):
            return

        usd_path = str(self.cfg.pipe.spawn.usd_path)
        metadata = self._read_pipe_model_metadata(usd_path, str(self.cfg.pipe_link_prim_name))
        reference_length = max(float(self.cfg.pipe_reference_mesh_length), 1.0e-9)
        model_scale = float(metadata["pipe_length"]) / reference_length

        self.cfg.pipe_link_local_pos = metadata["pipe_link_local_pos"]
        self.cfg.pipe_link_local_quat = metadata["pipe_link_local_quat"]
        self.cfg.pipe_length = float(metadata["pipe_length"])
        self.cfg.pipe_inner_radius = 0.015 * model_scale
        self.cfg.pipe_blood_valid_radius = 0.011 * model_scale
        self.cfg.pipe_blood_axis_margin = 0.003 * model_scale
        self.cfg.pipe_blood_template_z_start = 0.010 * model_scale
        self.cfg.pipe_blood_template_z_end = 0.045 * model_scale
        self._pipe_model_parameters_synced = True

        message = (
            "Synced pipe model parameters from "
            f"'{usd_path}' prim '{metadata['pipe_prim_path']}': "
            f"length={self.cfg.pipe_length:.9f}, scale={model_scale:.6f}, "
            f"inner_radius={self.cfg.pipe_inner_radius:.9f}, "
            f"blood_valid_radius={self.cfg.pipe_blood_valid_radius:.9f}."
        )
        print(f"[INFO] {message}", flush=True)
        carb.log_info(message)

    @staticmethod
    def _normalize_quat_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
        return quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-9)

    @staticmethod
    def _invert_quat_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
        quat = Ur3BloodPipeAbsorptionEnv._normalize_quat_torch(quat_wxyz)
        return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)

    @staticmethod
    def _multiply_quat_torch(lhs_wxyz: torch.Tensor, rhs_wxyz: torch.Tensor) -> torch.Tensor:
        lhs = Ur3BloodPipeAbsorptionEnv._normalize_quat_torch(lhs_wxyz)
        rhs = Ur3BloodPipeAbsorptionEnv._normalize_quat_torch(rhs_wxyz)
        lw, lx, ly, lz = lhs.unbind(dim=-1)
        rw, rx, ry, rz = rhs.unbind(dim=-1)
        result = torch.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            dim=-1,
        )
        return Ur3BloodPipeAbsorptionEnv._normalize_quat_torch(result)

    def _init_pipe_frame_state(self) -> None:
        pipe_root_pos = torch.tensor(tuple(self.cfg.pipe.init_state.pos), dtype=torch.float32, device=self.device)
        pipe_root_quat = torch.tensor(tuple(self.cfg.pipe.init_state.rot), dtype=torch.float32, device=self.device)
        pipe_link_pos = torch.tensor(self.cfg.pipe_link_local_pos, dtype=torch.float32, device=self.device)
        pipe_link_quat = torch.tensor(self.cfg.pipe_link_local_quat, dtype=torch.float32, device=self.device)

        self._pipe_root_pos_local = pipe_root_pos
        self._pipe_root_quat = self._normalize_quat_torch(pipe_root_quat)
        self._pipe_link_local_pos = pipe_link_pos
        self._pipe_link_local_quat = self._normalize_quat_torch(pipe_link_quat)
        self._pipe_pos_local = self._pipe_root_pos_local + self._quat_rotate_torch(
            self._pipe_root_quat.unsqueeze(0),
            self._pipe_link_local_pos.unsqueeze(0),
        )[0]
        self._pipe_quat_w = self._multiply_quat_torch(self._pipe_root_quat, self._pipe_link_local_quat)
        self._pipe_inv_quat_w = self._invert_quat_torch(self._pipe_quat_w)

    def _world_to_pipe_pos(self, pos_w: torch.Tensor) -> torch.Tensor:
        pipe_pos_w = self.scene.env_origins + self._pipe_pos_local.unsqueeze(0)
        rel_w = pos_w - pipe_pos_w
        inv_quat = self._pipe_inv_quat_w.unsqueeze(0).expand(pos_w.shape[0], -1)
        return self._quat_rotate_torch(inv_quat, rel_w)

    def _pipe_to_world_pos(self, pos_pipe: torch.Tensor) -> torch.Tensor:
        pipe_pos_w = self.scene.env_origins + self._pipe_pos_local.unsqueeze(0)
        pipe_quat = self._pipe_quat_w.unsqueeze(0).expand(pos_pipe.shape[0], -1)
        return pipe_pos_w + self._quat_rotate_torch(pipe_quat, pos_pipe)

    def _world_dir_to_pipe(self, dir_w: torch.Tensor) -> torch.Tensor:
        inv_quat = self._pipe_inv_quat_w.unsqueeze(0).expand(dir_w.shape[0], -1)
        return self._quat_rotate_torch(inv_quat, dir_w)

    def _clamp_pipe_position(self, pos_pipe: torch.Tensor) -> torch.Tensor:
        clamped = pos_pipe.clone()
        radius = max(float(self.cfg.pipe_inner_radius) - float(self.cfg.pipe_tool_clearance_margin), 1.0e-6)
        radial = torch.linalg.vector_norm(clamped[:, :2], dim=1, keepdim=True)
        scale = torch.clamp(radius / radial.clamp_min(1.0e-9), max=1.0)
        clamped[:, :2] = clamped[:, :2] * scale
        z_low = float(self.cfg.pipe_blood_axis_margin)
        z_high = max(float(self.cfg.pipe_length) - float(self.cfg.pipe_blood_axis_margin), z_low)
        clamped[:, 2] = torch.clamp(clamped[:, 2], min=z_low, max=z_high)
        return clamped

    def _compute_pipe_clearance(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_pipe = self._world_to_pipe_pos(pos_w)
        radial = torch.linalg.vector_norm(pos_pipe[:, :2], dim=1)
        clearance = float(self.cfg.pipe_inner_radius) - radial
        return pos_pipe, radial, clearance

    def _get_blood_template_capture_path(self) -> str:
        template_name = str(self.cfg.save_blood_init_template_name)
        if not template_name.endswith(".pt"):
            template_name = f"{template_name}.pt"
        return os.path.join(self.cfg.ASSET_PATH, template_name)

    def _maybe_save_blood_template(self) -> None:
        if not self._capture_blood_template_enabled or self._blood_template_capture_saved:
            return

        settle_steps = max(int(self.cfg.save_blood_init_template_after_steps), 0)
        if int(self._step_count[0].item()) < settle_steps:
            return

        particles_pos, _ = self.liquid.read_particles(0)
        if len(particles_pos) <= 0:
            raise RuntimeError("Cannot save blood template because env_0 has no particles.")

        save_path = self._get_blood_template_capture_path()
        torch.save(torch.tensor(np.asarray(particles_pos, dtype=np.float32)), save_path)
        self._blood_template_capture_saved = True
        message = (
            f"Saved blood template with {len(particles_pos)} particles to '{save_path}' "
            f"after {int(self._step_count[0].item())} control steps."
        )
        print(f"[INFO] {message}", flush=True)
        carb.log_info(message)

    def _initialize_blood_reset_source(self) -> None:
        if self._capture_blood_template_enabled:
            particles_pos, particles_vel = self.liquid.read_particles(0)
            self._capture_blood_reset_state = (
                np.asarray(particles_pos, dtype=np.float32).copy(),
                np.asarray(particles_vel, dtype=np.float32).copy(),
            )
            return

        self._load_blood_init_templates()

    def _load_blood_init_templates(self) -> None:
        max_particle_capacity = int(
            self.cfg.liquidCfg.numParticlesX * self.cfg.liquidCfg.numParticlesY * self.cfg.liquidCfg.numParticlesZ
        )
        blood_init_pos_list: list[np.ndarray] = []
        blood_init_vel_list: list[np.ndarray] = []
        blood_init_counts: list[int] = []

        spacing = float(self.cfg.liquidCfg.particleSpacing)
        xy_values = np.linspace(-spacing, spacing, int(self.cfg.liquidCfg.numParticlesX), dtype=np.float32)
        z_start = float(self.cfg.pipe_blood_template_z_start)
        z_end = float(self.cfg.pipe_blood_template_z_end)
        z_low = max(z_start, float(self.cfg.pipe_blood_axis_margin))
        z_high = min(z_end, float(self.cfg.pipe_length) - float(self.cfg.pipe_blood_axis_margin))
        if z_high <= z_low:
            raise ValueError("Pipe blood template z range is empty.")

        for template_idx, z_count in enumerate(tuple(self.cfg.pipe_blood_template_z_counts)):
            z_values = np.linspace(z_low, z_high, int(z_count), dtype=np.float32)
            positions_pipe = []
            for z_value in z_values:
                for x_value in xy_values:
                    for y_value in xy_values:
                        if math.hypot(float(x_value), float(y_value)) <= float(self.cfg.pipe_blood_valid_radius):
                            positions_pipe.append((x_value, y_value, z_value))

            positions_pipe_np = np.asarray(positions_pipe, dtype=np.float32)
            positions = pipe_to_env_local(positions_pipe_np, self.cfg)
            if positions.shape[0] > max_particle_capacity:
                raise ValueError(
                    f"Pipe blood template {template_idx} has {positions.shape[0]} particles, "
                    f"which exceeds the configured capacity of {max_particle_capacity}."
                )

            velocities = np.zeros_like(positions, dtype=np.float32)
            blood_init_pos_list.append(positions.copy())
            blood_init_vel_list.append(velocities)
            blood_init_counts.append(int(positions.shape[0]))

        self._blood_init_pos_list = blood_init_pos_list
        self._blood_init_vel_list = blood_init_vel_list
        self._blood_init_counts = tuple(blood_init_counts)

    def _reset_pipe_and_blood(self, env_ids: torch.Tensor) -> None:
        env_count = int(env_ids.numel())
        if env_count <= 0:
            return

        pipe_positions = torch.tensor(
            tuple(self.cfg.pipe.init_state.pos), dtype=torch.float32, device=self.device
        ).repeat(env_count, 1)
        pipe_positions += self.scene.env_origins[env_ids]
        pipe_orientations = torch.tensor(
            tuple(self.cfg.pipe.init_state.rot), dtype=torch.float32, device=self.device
        ).repeat(env_count, 1)
        env_id_list = env_ids.detach().cpu().tolist()
        self._pipe.set_world_poses(pipe_positions, pipe_orientations, env_id_list)

        if self._capture_blood_template_enabled:
            if self._capture_blood_reset_state is None:
                raise RuntimeError("Blood template capture mode is enabled but no capture reset state is available.")

            reset_pos, reset_vel = self._capture_blood_reset_state
            self.liquid.reset_particles(env_id_list, reset_pos, reset_vel)
            particle_count = float(reset_pos.shape[0])
            self._initial_particle_count[env_ids] = particle_count
            self._success_threshold[env_ids] = particle_count * float(self.cfg.blood_success_ratio)
            self._blood_template_index[env_ids] = -1
            return

        num_templates = len(self._blood_init_pos_list)
        if num_templates <= 0:
            raise RuntimeError("No blood templates are loaded for blood_pipe_state resets.")

        sampled_template_indices = torch.randint(0, num_templates, (env_count,), device=self.device)
        sampled_particle_counts = torch.empty((env_count,), dtype=torch.float32, device=self.device)

        for local_idx, env_id in enumerate(env_id_list):
            template_idx = int(sampled_template_indices[local_idx].item())
            self.liquid.set_particles_position(
                self._blood_init_pos_list[template_idx],
                self._blood_init_vel_list[template_idx],
                env_id,
            )
            sampled_particle_counts[local_idx] = float(self._blood_init_counts[template_idx])

        self._initial_particle_count[env_ids] = sampled_particle_counts
        self._success_threshold[env_ids] = sampled_particle_counts * float(self.cfg.blood_success_ratio)
        self._blood_template_index[env_ids] = sampled_template_indices

    def _set_task_state_dirty(self) -> None:
        self._task_state_dirty = True
        self._reward_cache_dirty = True
        self._done_cache_dirty = True
        self._observation_pending = True

    def _resolve_ik_handles(self) -> None:
        self._ur3_entity_cfg.resolve(self.scene)
        self._ee_jacobi_idx = self._ur3_entity_cfg.body_ids[0] - 1

    def _setup_scene(self):
        self._sync_pipe_model_parameters_from_usd()

        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)

        sim_context = SimulationContext()
        sim_context.set_render_mode(sim_context.RenderMode.FULL_RENDERING)
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/disableContactProcessing", False)
        settings.set("/rtx/translucency/enabled", True)

        self.cfg.table_setup.func(
            prim_path="/World/envs/env_0/Table",
            cfg=self.cfg.table_setup,
            translation=self.cfg.table_pos,
        )

        self._ur3_base_block = RigidObject(self.cfg.ur3_base_block)
        self.scene.rigid_objects["ur3_base_block"] = self._ur3_base_block

        lift = Gf.Vec3f(0.0, 0.0, self.cfg.table_height_offset)
        spawn_pos_pipe = self.cfg.spawn_pos_pipe + lift
        spawn_pos_fluid = self.cfg.spawn_pos_fluid + lift
        spawn_pos_glass2 = self.cfg.spawn_pos_glass2

        self.cfg.pipe.init_state.pos = spawn_pos_pipe
        self.cfg.pipe.spawn.func(
            self.cfg.pipe.prim_path,
            self.cfg.pipe.spawn,
            translation=self.cfg.pipe.init_state.pos,
            orientation=self.cfg.pipe.init_state.rot,
        )
        self._pipe = XFormPrimView(self.cfg.pipe.prim_path, reset_xform_properties=False)
        self.scene.extras["pipe"] = self._pipe

        self.liquid = FluidObject(cfg=self.cfg.liquidCfg, lower_pos=spawn_pos_fluid)
        self.liquid.spawn_fluid_direct()
        self._initialize_blood_reset_source()

        self.cfg.glass2.init_state.pos = spawn_pos_glass2
        self._glass2 = RigidObject(self.cfg.glass2)
        self.scene.rigid_objects["glass2"] = self._glass2

        self._ur3 = Articulation(self.cfg.ur3_robot)
        self.scene.articulations["ur3"] = self._ur3

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self._ee_jacobi_idx is None:
            self._resolve_ik_handles()

        if self._capture_blood_template_enabled:
            self._raw_actions.zero_()
            tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
            self._ee_goal_pos_w[:] = tip_pos_w
            self._joint_pos_des[:] = self._ur3.data.default_joint_pos[:, self._ik_joint_ids]
            self._set_task_state_dirty()
            return

        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        delta_pos_pipe = self._raw_actions * self._pipe_action_scale.unsqueeze(0)

        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        tip_quat_w = self._compute_tip_quat_w()
        uninitialized_goal = torch.linalg.vector_norm(self._ee_goal_pos_w, dim=-1) < 1.0e-6
        self._ee_goal_pos_w[uninitialized_goal] = tip_pos_w[uninitialized_goal]
        ee_goal_pos_pipe = self._world_to_pipe_pos(self._ee_goal_pos_w)
        ee_goal_pos_pipe = self._clamp_pipe_position(ee_goal_pos_pipe + delta_pos_pipe)
        self._ee_goal_pos_w[:] = self._pipe_to_world_pos(ee_goal_pos_pipe)

        root_pose_w = self._ur3.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            tip_pos_w,
            tip_quat_w,
        )
        ee_goal_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            self._ee_goal_pos_w,
            tip_quat_w,
        )

        self._ik_commands[:] = ee_goal_pos_b
        self._ik_controller.set_command(self._ik_commands, ee_quat=ee_quat_b)

        jacobian = self._ur3.root_physx_view.get_jacobians()[:, self._ee_jacobi_idx, :, self._ik_joint_ids]
        joint_pos = self._ur3.data.joint_pos[:, self._ik_joint_ids]
        self._joint_pos_des[:] = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self._joint_pos_des[:] = torch.clamp(self._joint_pos_des, self._ik_joint_lower_limits, self._ik_joint_upper_limits)

        self._set_task_state_dirty()

    def _apply_action(self):
        self._ur3.set_joint_position_target(self._joint_pos_des, joint_ids=self._ik_joint_ids)

    def _register_ur3_contact_sensors(self) -> None:
        tip_body_path = self._ur3_body_name_to_path.get(self.cfg.ur3_tip_body_name)
        tip_contact_cfg = ContactSensorCfg(
            prim_path=tip_body_path,
            update_period=0.0,
            history_length=1,
            track_air_time=True,
            force_threshold=float(self.cfg.tip_contact_force_threshold),
            debug_vis=False,
        )
        self._tip_contact_sensor = ContactSensor(cfg=tip_contact_cfg)
        self.scene.sensors["tip_contact"] = self._tip_contact_sensor

        self._ur3_contact_sensors = {}
        for body_name in self._ur3_collision_body_names:
            body_path = self._ur3_body_name_to_path.get(body_name)
            if body_path is None:
                raise RuntimeError(f"Failed to resolve collision sensor path for UR3 body '{body_name}'.")
            sensor_name = f"ur3_contact_{body_name}"
            ur3_contact_cfg = ContactSensorCfg(
                prim_path=body_path,
                update_period=0.0,
                history_length=1,
                force_threshold=float(self.cfg.severe_contact_force_threshold),
                debug_vis=False,
            )
            sensor = ContactSensor(cfg=ur3_contact_cfg)
            self._ur3_contact_sensors[body_name] = sensor
            self.scene.sensors[sensor_name] = sensor

        if self.sim.is_playing():
            self._tip_contact_sensor._initialize_callback(None)
            for sensor in self._ur3_contact_sensors.values():
                sensor._initialize_callback(None)

    def _resolve_ur3_collision_body_selection(self) -> None:
        pattern = re.compile(self.cfg.ur3_collision_body_names_expr)
        body_names = [name for name in self._ur3.body_names if pattern.fullmatch(name)]
        if len(body_names) == 0:
            available = ", ".join(self._ur3.body_names)
            raise RuntimeError(
                "Failed to match any UR3 collision bodies with pattern "
                f"'{self.cfg.ur3_collision_body_names_expr}'. Available bodies: {available}"
            )
        self._ur3_collision_body_names = body_names
        carb.log_info(
            "UR3 severe collision monitoring bodies: "
            + ", ".join(self._ur3_collision_body_names)
        )

    def _get_tip_contact_force(self) -> torch.Tensor:
        if self._tip_contact_sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        net_forces_w = self.scene["tip_contact"].data.net_forces_w
        contact_force = torch.linalg.vector_norm(net_forces_w, dim=-1)
        if contact_force.ndim > 1:
            contact_force = torch.amax(contact_force, dim=1)
        return contact_force.to(dtype=torch.float32)

    def _get_ur3_contact_force(self) -> torch.Tensor:
        if len(self._ur3_contact_sensors) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        per_body_contact_force = []
        for sensor in self._ur3_contact_sensors.values():
            net_forces_w = sensor.data.net_forces_w
            contact_force = torch.linalg.vector_norm(net_forces_w, dim=-1)
            if contact_force.ndim > 1:
                contact_force = torch.amax(contact_force, dim=1)
            per_body_contact_force.append(contact_force.to(dtype=torch.float32))

        return torch.amax(torch.stack(per_body_contact_force, dim=0), dim=0)

    def _build_ur3_body_lookup(self) -> tuple[dict[str, int], dict[str, str]]:
        env_zero_ns = self.scene.env_regex_ns.replace(".*", "0")
        body_name_to_idx: dict[str, int] = {}
        body_name_to_path: dict[str, str] = {}

        for body_idx, (body_name, body_path) in enumerate(zip(self._ur3.body_names, self._ur3.root_physx_view.link_paths[0])):
            body_name_to_idx[body_name] = body_idx
            if body_path.startswith(env_zero_ns):
                body_name_to_path[body_name] = body_path.replace(env_zero_ns, self.scene.env_regex_ns, 1)
            else:
                body_name_to_path[body_name] = body_path

        return body_name_to_idx, body_name_to_path

    def _resolve_required_body_idx(self, body_name: str) -> int:
        if body_name not in self._ur3_body_name_to_idx:
            available = ", ".join(self._ur3.body_names)
            raise RuntimeError(f"Required UR3 body '{body_name}' not found. Available bodies: {available}")
        return self._ur3_body_name_to_idx[body_name]

    @staticmethod
    def _quat_rotate_torch(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        quat_vec = quat_wxyz[..., 1:]
        uv = torch.linalg.cross(quat_vec, vec, dim=-1)
        uuv = torch.linalg.cross(quat_vec, uv, dim=-1)
        return vec + 2.0 * (quat_wxyz[..., :1] * uv + uuv)

    def _compute_tip_quat_w(self) -> torch.Tensor:
        body_quat_w = getattr(self._ur3.data, "body_quat_w", None)
        if body_quat_w is None:
            return self._ur3.data.root_state_w[:, 3:7]
        return body_quat_w[:, self._tip_body_idx]

    def _compute_tip_pose_and_direction_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        tip_body_pos_w = self._ur3.data.body_pos_w[:, self._tip_body_idx]
        local_offset = self._tip_local_offset.unsqueeze(0).expand(self.num_envs, -1)
        local_axis = self._tip_local_axis.unsqueeze(0).expand(self.num_envs, -1)

        body_quat_w = getattr(self._ur3.data, "body_quat_w", None)
        if body_quat_w is None:
            tip_pos_w = tip_body_pos_w + local_offset
            tip_dir_w = local_axis
        else:
            tip_body_quat_w = body_quat_w[:, self._tip_body_idx]
            tip_pos_w = tip_body_pos_w + self._quat_rotate_torch(tip_body_quat_w, local_offset)
            if self.cfg.use_body_quat_for_tip_dir:
                tip_dir_w = self._quat_rotate_torch(tip_body_quat_w, local_axis)
            else:
                tip_dir_w = local_axis

        tip_dir_w = tip_dir_w / torch.linalg.vector_norm(tip_dir_w, dim=-1, keepdim=True).clamp_min(1.0e-9)
        return tip_pos_w, tip_dir_w

    def _refresh_post_step_task_state(self) -> None:
        if not self._task_state_dirty:
            return

        tip_pos_w, tip_dir_w = self._compute_tip_pose_and_direction_w()
        # 准备传入底层 NumPy 接口的坐标数组
        env_origins_np = self.scene.env_origins.detach().cpu().numpy()
        tip_pos_local_np = (tip_pos_w - self.scene.env_origins).detach().cpu().numpy()
        tip_dir_w_np = tip_dir_w.detach().cpu().numpy()
        if self._capture_blood_template_enabled:
            apply_suction_mask_np = np.zeros((self.num_envs,), dtype=bool)
        else:
            apply_suction_mask_np = (self._step_count > 0).detach().cpu().numpy()
        glass2_pos_np = (self._glass2.data.root_pos_w - self.scene.env_origins).detach().cpu().numpy()

        # 一步运算，拿到吸血结果和当前统计指标
        particle_stats = self._suction_controller.step(
            tip_pos_local_np=tip_pos_local_np,
            tip_dir_w_np=tip_dir_w_np,
            liquid=self.liquid,
            glass2_pos_np=glass2_pos_np,
            env_origins_np=env_origins_np,
            apply_suction_mask=apply_suction_mask_np,
        )

        # Tracker 纯粹更新 Torch 张量和时序奖励指标
        self._particle_task_tracker.refresh(
            tip_pos_w=tip_pos_w,
            step_count=self._step_count,
            particle_stats=particle_stats,
        )
        self._task_state_dirty = False

    def _build_pipe_pose_features(self, pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_pipe, _, clearance = self._compute_pipe_clearance(pos_w)
        pipe_length = max(float(self.cfg.pipe_length), 1.0e-6)
        pipe_radius = max(float(self.cfg.pipe_inner_radius), 1.0e-6)
        features = torch.stack(
            (
                torch.clamp(2.0 * pos_pipe[:, 2] / pipe_length - 1.0, -1.0, 1.0),
                torch.clamp(pos_pipe[:, 0] / pipe_radius, -1.0, 1.0),
                torch.clamp(pos_pipe[:, 1] / pipe_radius, -1.0, 1.0),
                torch.clamp(clearance / pipe_radius, -1.0, 1.0),
            ),
            dim=1,
        )
        return features, pos_pipe

    def _build_observation_from_task_state(
        self,
        tip_pos_w: torch.Tensor,
        tip_dir_w: torch.Tensor,
        contact_force: torch.Tensor,
    ) -> torch.Tensor:
        task_state = self._particle_state
        tip_pose_features, tip_pos_pipe = self._build_pipe_pose_features(tip_pos_w)
        blood_pose_features, blood_pos_pipe = self._build_pipe_pose_features(task_state.blood_centroid)
        tip_dir_pipe = self._world_dir_to_pipe(tip_dir_w)
        tip_dir_pipe = tip_dir_pipe / torch.linalg.vector_norm(tip_dir_pipe, dim=1, keepdim=True).clamp_min(1.0e-9)

        pipe_scale = torch.tensor(
            (
                max(float(self.cfg.pipe_inner_radius), 1.0e-6),
                max(float(self.cfg.pipe_inner_radius), 1.0e-6),
                max(float(self.cfg.pipe_length), 1.0e-6),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        blood_centroid_rel_normalized = torch.clamp((blood_pos_pipe - tip_pos_pipe) / pipe_scale, -1.0, 1.0)
        goal_pos_pipe = self._world_to_pipe_pos(self._ee_goal_pos_w)
        goal_error_normalized = torch.clamp((goal_pos_pipe - tip_pos_pipe) / pipe_scale, -1.0, 1.0)

        absorbed_ratio = torch.clamp(
            task_state.absorbed_count / self._initial_particle_count.clamp_min(1.0),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        absorbed_delta_ema = torch.tanh(task_state.absorbed_delta_ema).unsqueeze(1)
        valid_in_cone_ratio = torch.clamp(task_state.valid_in_cone_ratio, min=0.0, max=1.0).unsqueeze(1)
        valid_in_inlet_ratio = torch.clamp(task_state.valid_in_inlet_ratio, min=0.0, max=1.0).unsqueeze(1)
        contact_ratio = torch.clamp(
            contact_force / max(float(self.cfg.severe_contact_force_threshold), 1.0e-6),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)
        step_ratio = torch.clamp(
            self._step_count.to(dtype=torch.float32) / max(float(self.max_episode_length), 1.0),
            min=0.0,
            max=1.0,
        ).unsqueeze(1)

        return torch.cat(
            (
                tip_pose_features,
                tip_dir_pipe,
                blood_pose_features,
                blood_centroid_rel_normalized,
                goal_error_normalized,
                valid_in_cone_ratio,
                valid_in_inlet_ratio,
                absorbed_ratio,
                absorbed_delta_ema,
                contact_ratio,
                step_ratio,
            ),
            dim=1,
        )

    def _update_low_dim_observation(self) -> None:
        self._refresh_post_step_task_state()

        tip_pos_w, tip_dir_w = self._compute_tip_pose_and_direction_w()
        reset_goal_mask = self._step_count <= 0
        if torch.any(reset_goal_mask):
            tip_pos_pipe = self._world_to_pipe_pos(tip_pos_w)
            clamped_tip_goal_w = self._pipe_to_world_pos(self._clamp_pipe_position(tip_pos_pipe))
            self._ee_goal_pos_w[reset_goal_mask] = clamped_tip_goal_w[reset_goal_mask]
        tip_contact_force = self._get_tip_contact_force()
        self._obs_state[:] = self._build_observation_from_task_state(tip_pos_w, tip_dir_w, tip_contact_force)

    def _compute_termination_flags(
        self, contact_force: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        task_state = self._particle_state
        ik_joint_pos = self._ur3.data.joint_pos[:, self._ik_joint_ids]
        tolerance = float(self.cfg.joint_limit_termination_tolerance)
        joint_limit_reached = torch.any(
            (ik_joint_pos <= self._ik_joint_lower_limits + tolerance)
            | (ik_joint_pos >= self._ik_joint_upper_limits - tolerance),
            dim=1,
        )

        severe_contact = contact_force > float(self.cfg.severe_contact_force_threshold)
        next_counter = torch.where(
            severe_contact,
            self._severe_contact_counter + 1,
            torch.zeros_like(self._severe_contact_counter),
        )
        severe_collision = next_counter >= int(self.cfg.severe_contact_patience)
        absorption_complete = task_state.absorbed_count >= self._success_threshold
        success = absorption_complete & (~joint_limit_reached) & (~severe_collision)

        terminated = success | severe_collision
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        flags = {
            "success": success,
            "joint_limit": joint_limit_reached,
            "severe_collision": severe_collision,
            "absorption_complete": absorption_complete,
            "time_out": truncated,
        }
        return terminated, truncated, next_counter, flags

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._done_cache_dirty:
            return self._terminated_cache.clone(), self._truncated_cache.clone()

        if self._capture_blood_template_enabled:
            self._refresh_post_step_task_state()
            self._severe_contact_counter.zero_()
            self._episode_success[:] = False
            self._episode_joint_limit[:] = False
            self._episode_severe_collision[:] = False
            self._episode_time_out[:] = False
            self._terminated_cache[:] = False
            self._truncated_cache[:] = False
            self._done_cache_dirty = False
            return self._terminated_cache.clone(), self._truncated_cache.clone()

        self._refresh_post_step_task_state()

        ur3_contact_force = self._get_ur3_contact_force()
        terminated, truncated, next_counter, flags = self._compute_termination_flags(ur3_contact_force)
        self._severe_contact_counter[:] = next_counter
        self._episode_success[:] = flags["success"]
        self._episode_joint_limit[:] = flags["joint_limit"]
        self._episode_severe_collision[:] = flags["severe_collision"]
        self._episode_time_out[:] = flags["time_out"]
        self._terminated_cache[:] = terminated
        self._truncated_cache[:] = truncated
        self._done_cache_dirty = False
        return terminated.clone(), truncated.clone()

    def _compute_reward_terms(self, reward_inputs: ParticleRewardInputs) -> dict[str, torch.Tensor]:
        task_state = self._particle_state

        success_threshold = self._success_threshold.clamp_min(1.0)
        absorbed_frac_delta = task_state.absorbed_delta / success_threshold
        absorb_reward = self.cfg.reward_absorb_weight * absorbed_frac_delta

        centroid_progress = torch.clamp(
            task_state.prev_blood_centroid_distance - task_state.blood_centroid_distance,
            min=-float(self.cfg.centroid_progress_clip),
            max=float(self.cfg.centroid_progress_clip),
        )
        centroid_progress_reward = self.cfg.centroid_progress_weight * centroid_progress

        action_penalty = self.cfg.reward_action_weight * torch.sum(reward_inputs.raw_actions**2, dim=1)

        safe_contact_force = torch.log1p(torch.clamp(reward_inputs.contact_force, min=0.0))
        safe_contact_threshold = math.log1p(float(self.cfg.tip_contact_force_threshold))
        collision_force_penalty = self.cfg.reward_collision_force_weight * torch.clamp(
            safe_contact_force - safe_contact_threshold,
            min=0.0,
        )
        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        _, _, tip_clearance = self._compute_pipe_clearance(tip_pos_w)
        clearance_margin = max(float(self.cfg.pipe_tool_clearance_margin), 1.0e-6)
        wall_clearance_penalty = float(self.cfg.pipe_wall_clearance_penalty_weight) * torch.clamp(
            (clearance_margin - tip_clearance) / clearance_margin,
            min=0.0,
        )
        joint_limit_penalty = float(self.cfg.reward_joint_limit_penalty) * (
            self._episode_joint_limit & (~self._episode_success)
        ).float()

        time_penalty = torch.full(
            (self.num_envs,),
            float(self.cfg.reward_time_penalty),
            dtype=torch.float32,
            device=self.device,
        )
        absorption_complete = task_state.absorbed_count >= self._success_threshold
        task_complete = self.cfg.reward_task_complete * absorption_complete.float()
        total_reward = (
            task_complete
            + absorb_reward
            + centroid_progress_reward
            - action_penalty
            - collision_force_penalty
            - wall_clearance_penalty
            - joint_limit_penalty
            - time_penalty
        ).float()

        return {
            "absorb_reward": absorb_reward,
            "centroid_progress_reward": centroid_progress_reward,
            "action_penalty": action_penalty,
            "collision_force_penalty": collision_force_penalty,
            "wall_clearance_penalty": wall_clearance_penalty,
            "joint_limit_penalty": joint_limit_penalty,
            "time_penalty": time_penalty,
            "task_complete": task_complete,
            "total_reward": total_reward,
        }

    def _get_rewards(self) -> torch.Tensor:
        if not self._reward_cache_dirty:
            return self._reward_cache.clone()

        self._refresh_post_step_task_state()

        raw_contact_force = self._get_tip_contact_force()
        ur3_contact_force = self._get_ur3_contact_force()
        reward_terms = self._compute_reward_terms(
            ParticleRewardInputs(
                raw_actions=self._raw_actions,
                contact_force=raw_contact_force,
            )
        )
        task_state = self._particle_state

        self._episode_reward_sums["absorb_reward"] += reward_terms["absorb_reward"]
        self._episode_reward_sums["centroid_progress_reward"] += reward_terms["centroid_progress_reward"]
        self._episode_reward_sums["action_penalty"] -= reward_terms["action_penalty"]
        self._episode_reward_sums["collision_force_penalty"] -= reward_terms["collision_force_penalty"]
        self._episode_reward_sums["wall_clearance_penalty"] -= reward_terms["wall_clearance_penalty"]
        self._episode_reward_sums["joint_limit_penalty"] -= reward_terms["joint_limit_penalty"]
        self._episode_reward_sums["time_penalty"] -= reward_terms["time_penalty"]
        self._episode_reward_sums["task_complete"] += reward_terms["task_complete"]

        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        _, _, tip_clearance = self._compute_pipe_clearance(tip_pos_w)
        absorption_complete = task_state.absorbed_count >= self._success_threshold
        self.extras["log"] = {
            "Metrics/absorbed_count": task_state.absorbed_count.mean(),
            "Metrics/absorbed_delta": task_state.absorbed_delta.mean(),
            "Metrics/absorbed_ratio_mean": torch.clamp(
                task_state.absorbed_count / self._initial_particle_count.clamp_min(1.0),
                min=0.0,
                max=1.0,
            ).mean(),
            "Metrics/blood_centroid_distance": task_state.blood_centroid_distance.mean(),
            "Metrics/initial_particle_count": self._initial_particle_count.mean(),
            "Metrics/success_threshold": self._success_threshold.mean(),
            "Metrics/blood_template_index": self._blood_template_index.to(dtype=torch.float32).mean(),
            "Metrics/valid_in_cone_ratio": task_state.valid_in_cone_ratio.mean(),
            "Metrics/valid_in_inlet_ratio": task_state.valid_in_inlet_ratio.mean(),
            "Metrics/absorbed_delta_ema": task_state.absorbed_delta_ema.mean(),
            "Metrics/raw_contact_force_mean": raw_contact_force.mean(),
            "Metrics/raw_contact_force_max": raw_contact_force.max(),
            "Metrics/ur3_contact_force_mean": ur3_contact_force.mean(),
            "Metrics/ur3_contact_force_max": ur3_contact_force.max(),
            "Metrics/tip_pipe_clearance_mean": tip_clearance.mean(),
            "Metrics/tip_pipe_clearance_min": tip_clearance.min(),
            "Metrics/absorption_complete_rate": absorption_complete.float().mean(),
            "Metrics/success_rate": absorption_complete.float().mean(),
        }

        self._reward_cache[:] = reward_terms["total_reward"]
        self._reward_cache_dirty = False
        return self._reward_cache.clone()

    def _flush_episode_logs(self, finished_env_ids: torch.Tensor) -> None:
        if finished_env_ids.numel() <= 0:
            return

        if "log" not in self.extras:
            self.extras["log"] = {}

        actual_steps = self._step_count[finished_env_ids].float().clamp_min(1.0)
        reward_logs = {}
        for key, values in self._episode_reward_sums.items():
            reward_logs[f"Episode_Reward/{key}"] = (values[finished_env_ids] / actual_steps).mean()

        success_mask = self._episode_success[finished_env_ids]
        joint_limit_mask = (~success_mask) & self._episode_joint_limit[finished_env_ids]
        severe_collision_mask = (
            (~success_mask)
            & (~joint_limit_mask)
            & self._episode_severe_collision[finished_env_ids]
        )
        time_out_mask = (
            (~success_mask)
            & (~joint_limit_mask)
            & (~severe_collision_mask)
            & self._episode_time_out[finished_env_ids]
        )
        termination_logs = {
            "Episode_Termination/success": success_mask.float().mean(),
            "Episode_Termination/joint_limit": joint_limit_mask.float().mean(),
            "Episode_Termination/severe_collision": severe_collision_mask.float().mean(),
            "Episode_Termination/time_out": time_out_mask.float().mean(),
        }
        self.extras["log"].update(reward_logs)
        self.extras["log"].update(termination_logs)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if self._ee_jacobi_idx is None:
            self._resolve_ik_handles()
        self._ik_controller.reset(env_ids=env_ids)

        finished_mask = (
            self._episode_success[env_ids]
            | self._episode_joint_limit[env_ids]
            | self._episode_severe_collision[env_ids]
            | self._episode_time_out[env_ids]
        )
        self._flush_episode_logs(env_ids[finished_mask])
        if self._capture_blood_template_enabled:
            self._blood_template_capture_saved = False

        root_state = self._ur3.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._ur3.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._ur3.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        joint_pos = self._ur3.data.default_joint_pos[env_ids]
        joint_vel = self._ur3.data.default_joint_vel[env_ids]
        self._ur3.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._ur3.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._ur3.reset(env_ids=env_ids)
        self._joint_pos_des[env_ids] = joint_pos[:, self._ik_joint_ids]

        self._reset_pipe_and_blood(env_ids)

        self._suction_controller.reset(env_ids.tolist())
        self._step_count[env_ids] = 0
        self._severe_contact_counter[env_ids] = 0
        for values in self._episode_reward_sums.values():
            values[env_ids] = 0.0
        self._episode_success[env_ids] = False
        self._episode_joint_limit[env_ids] = False
        self._episode_severe_collision[env_ids] = False
        self._episode_time_out[env_ids] = False
        self._raw_actions[env_ids] = 0.0
        self._ik_commands[env_ids] = 0.0
        self._obs_state[env_ids] = 0.0

        tip_pos_w, _ = self._compute_tip_pose_and_direction_w()
        tip_pos_pipe = self._world_to_pipe_pos(tip_pos_w)
        self._ee_goal_pos_w[env_ids] = self._pipe_to_world_pos(self._clamp_pipe_position(tip_pos_pipe))[env_ids]
        self._particle_task_tracker.reset(env_ids, tip_pos_w)
        self._set_task_state_dirty()

    def _get_observations(self) -> dict:
        if self._observation_pending:
            self._update_low_dim_observation()

            self._step_count += 1
            self._maybe_save_blood_template()
            self._observation_pending = False

        return {"policy": self._obs_state.clone()}
