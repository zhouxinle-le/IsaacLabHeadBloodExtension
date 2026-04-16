from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import omni.kit.commands
from omni.isaac.core.utils.stage import get_current_stage
from omni.physx.scripts import particleUtils, physicsUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdShade, Vt


class FluidObjectCfg:

    # Number of particles along the hoorizontal and vertical axes (for direct spawn)
    numParticlesX: int
    numParticlesY: int
    numParticlesZ: int
    particleSpacing: float

    # Fluid properties
    particle_mass: float
    density: float
    viscosity: float


class FluidObject:

    cfg: FluidObjectCfg

    def __init__(self, cfg: FluidObjectCfg, lower_pos: Gf.Vec3f):
        self.cfg = cfg
        self.lower_pos = lower_pos  # Lower position for the spawn

        # Scene infos (Default values)
        self.stage = get_current_stage()
        self.default_prim = UsdGeom.Xform.Define(
            self.stage, Sdf.Path("/World")
        ).GetPrim()
        self.stage.SetDefaultPrim(self.default_prim)
        self.default_prim_path = self.stage.GetDefaultPrim().GetPath()
        self.scenePath = Sdf.Path("/physicsScene")
        self._particle_paths: dict[int, Sdf.Path] = {}
        self._particle_prims: dict[int, UsdGeom.Points] = {}
        self._initial_particles_pos: np.ndarray | None = None
        self._initial_particles_vel: np.ndarray | None = None

    def spawn_fluid_direct(self, env_index: int = 0):

        # Particle System
        self.particleSystemPath = self.default_prim_path.AppendChild("particleSystem")

        # Particle points
        self.particlesPath = Sdf.Path(f"/World/envs/env_{env_index}/particles")
        self._particle_paths[env_index] = self.particlesPath

        # solver iterations
        self._solverPositionIterations = 4
        physxAPI = PhysxSchema.PhysxSceneAPI.Apply(
            self.stage.GetPrimAtPath(self.scenePath)
        )
        physxAPI.CreateSolverTypeAttr("TGS")

        # particle params
        restOffset = self.cfg.particleSpacing * 0.9
        fluidRestOffset = restOffset * 0.6
        particleContactOffset = restOffset + 0.001
        particle_system = particleUtils.add_physx_particle_system(
            stage=self.stage,
            particle_system_path=self.particleSystemPath,
            simulation_owner=self.scenePath,
            contact_offset=restOffset * 1.1,
            rest_offset=0.001,
            particle_contact_offset=particleContactOffset,
            solid_rest_offset=0.0,
            fluid_rest_offset=fluidRestOffset,
            solver_position_iterations=self._solverPositionIterations,
        )

        mtl_created = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniSurfacePresets.mdl",
            mtl_name="OmniSurface_Blood",
            mtl_created_list=mtl_created,
        )

        # 获取材质和 Shader 路径
        pbd_particle_material_path = mtl_created[0]
        shader_path = f"{pbd_particle_material_path}/Shader"

        # 1. 获取 Prim 并包装成 Shader 对象
        shader_prim = self.stage.GetPrimAtPath(shader_path)
        shader = UsdShade.Shader(shader_prim)

        # # Base
        # shader.CreateInput("diffuse_reflection_weight", Sdf.ValueTypeNames.Float).Set(1.0)
        # shader.CreateInput("diffuse_reflection_color",  Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.43, 0.0, 0.0))

        # # Specular
        # shader.CreateInput("specular_reflection_weight",    Sdf.ValueTypeNames.Float).Set(1.0)
        # shader.CreateInput("specular_reflection_roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        # shader.CreateInput("specular_reflection_color",     Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))

        # Subsurface
        # shader.CreateInput("enable_diffuse_transmission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(0.2)
        # shader.CreateInput("subsurface_scale",            Sdf.ValueTypeNames.Float).Set(0.1)
        # shader.CreateInput("subsurface_transmission_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.43, 0.0, 0.0))

        omni.kit.commands.execute(
            "BindMaterial",
            prim_path=self.particleSystemPath,
            material_path=pbd_particle_material_path,
        )

        # Create a pbd particle material and set it on the particle system
        particleUtils.add_pbd_particle_material(
            self.stage,
            pbd_particle_material_path,
            cohesion=10,
            viscosity=self.cfg.viscosity,
            surface_tension=0.74,
            friction=0.1,
        )
        physicsUtils.add_physics_material_to_prim(
            self.stage, particle_system.GetPrim(), pbd_particle_material_path
        )

        particle_system.CreateMaxVelocityAttr().Set(200)

        # add particle anisotropy
        anisotropyAPI = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(
            particle_system.GetPrim()
        )
        anisotropyAPI.CreateParticleAnisotropyEnabledAttr().Set(True)
        aniso_scale = 5.0
        anisotropyAPI.CreateScaleAttr().Set(aniso_scale)
        anisotropyAPI.CreateMinAttr().Set(1.0)
        anisotropyAPI.CreateMaxAttr().Set(2.0)

        # add particle smoothing
        smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(
            particle_system.GetPrim()
        )
        smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
        smoothingAPI.CreateStrengthAttr().Set(0.5)

        # apply isosurface params
        isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(
            particle_system.GetPrim()
        )
        isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
        isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
        isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
        isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
        isosurfaceAPI.CreateGridSpacingAttr().Set(fluidRestOffset * 1.5)
        isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluidRestOffset * 1.6)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
        isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluidRestOffset * 2)

        isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(1)

        primVarsApi = UsdGeom.PrimvarsAPI(particle_system)
        primVarsApi.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

        self.stage.SetInterpolationType(Usd.InterpolationTypeHeld)

        # Create Grid
        gridSpacing = self.cfg.particleSpacing + 0.001
        lower = self.lower_pos + Gf.Vec3f(
            -gridSpacing * self.cfg.numParticlesX / 2,
            -gridSpacing * self.cfg.numParticlesY / 2,
            0,
        )  # Translate lower corner
        positions, velocities = particleUtils.create_particles_grid(
            lower,
            gridSpacing,
            self.cfg.numParticlesX,
            self.cfg.numParticlesY,
            self.cfg.numParticlesZ,
        )

        widths = [self.cfg.particleSpacing] * len(positions)

        self.particlesPrim = particleUtils.add_physx_particleset_points(
            stage=self.stage,
            path=self.particlesPath,
            positions_list=Vt.Vec3fArray(positions),
            velocities_list=Vt.Vec3fArray(velocities),
            widths_list=widths,
            particle_system_path=self.particleSystemPath,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=self.cfg.particle_mass,
            density=self.cfg.density,
        )

        # 粒子可见性设置：invisible 隐藏粒子点，只显示等值面
        visibility_attribute = self.particlesPrim.GetVisibilityAttr()
        visibility_attribute.Set("invisible")
        self._particle_prims[env_index] = UsdGeom.Points(
            self.stage.GetPrimAtPath(self.particlesPath)
        )

    @property
    def has_initial_state(self) -> bool:
        return (
            self._initial_particles_pos is not None
            and self._initial_particles_vel is not None
        )

    def _get_particles_path(self, env_id: int) -> Sdf.Path:
        if env_id not in self._particle_paths:
            self._particle_paths[env_id] = self.default_prim_path.AppendPath(
                f"envs/env_{env_id}/particles"
            )
        return self._particle_paths[env_id]

    def _get_particles_prim(self, env_id: int) -> UsdGeom.Points:
        if env_id not in self._particle_prims:
            self._particle_prims[env_id] = UsdGeom.Points(
                self.stage.GetPrimAtPath(self._get_particles_path(env_id))
            )
        return self._particle_prims[env_id]

    def read_particles(self, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        particles = self._get_particles_prim(env_id)
        particles_pos = np.array(particles.GetPointsAttr().Get())
        particles_vel = np.array(particles.GetVelocitiesAttr().Get())
        return particles_pos, particles_vel

    def write_particles(
        self, env_id: int, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        particles = self._get_particles_prim(env_id)
        particles.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(positions, dtype=np.float32))
        )
        particles.GetVelocitiesAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(velocities, dtype=np.float32))
        )

    def capture_initial_state(self, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
        positions, velocities = self.read_particles(env_id)
        self._initial_particles_pos = positions.copy()
        self._initial_particles_vel = velocities.copy()
        return self._initial_particles_pos.copy(), self._initial_particles_vel.copy()

    def get_initial_state(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.has_initial_state:
            return None
        return self._initial_particles_pos.copy(), self._initial_particles_vel.copy()

    def reset_particles(
        self,
        env_ids: Iterable[int],
        positions: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        if positions is None or velocities is None:
            if not self.has_initial_state:
                return
            positions = self._initial_particles_pos
            velocities = self._initial_particles_vel

        for env_id in env_ids:
            self.write_particles(
                int(env_id), np.asarray(positions).copy(), np.asarray(velocities).copy()
            )

    def get_particles_position(self, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.read_particles(env_id)

    def set_particles_position(
        self, particles_pos: np.ndarray, particles_vel: np.ndarray, env_id: int
    ):
        self.write_particles(env_id, particles_pos, particles_vel)
