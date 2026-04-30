from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    ISAACLAB_SCENE_AVAILABLE = True
except Exception:
    ISAACLAB_SCENE_AVAILABLE = False


@dataclass
class SceneHandles:
    drone: str = "drone_handle"
    shuttlecock: str = "shuttlecock_handle"
    court: str = "court_handle"
    net: str = "net_handle"


@dataclass
class SceneEntityConfigs:
    drone_cfg: Any
    shuttlecock_cfg: Any
    net_cfg: Any
    drone_kind: str
    resolved_air_usd_path: str


DEFAULT_DRONE_ASSET_KIND = "articulation"
DEFAULT_DRONE_USD_PATH = "F:/eai/isaaclab/badminton_intercept_project_1/assets/X152b/model.usd"
_DRONE_ASSET_KIND_ALIASES = {
    "rigid": "rigid_object",
    "rigid_object": "rigid_object",
    "articulation": "articulation",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_drone_asset_kind(cfg: Any) -> str:
    configured = str(getattr(cfg, "drone_asset_kind", DEFAULT_DRONE_ASSET_KIND)).strip().lower()
    if not configured:
        configured = DEFAULT_DRONE_ASSET_KIND
    drone_kind = _DRONE_ASSET_KIND_ALIASES.get(configured)
    if drone_kind is None:
        valid = ", ".join(sorted(_DRONE_ASSET_KIND_ALIASES))
        raise ValueError(f"Unsupported drone_asset_kind={configured!r}. Expected one of: {valid}.")
    return drone_kind


def _resolve_air_usd_path(cfg: Any) -> str:
    override = os.environ.get("BADMINTON_AIR_USD")
    configured = str(getattr(cfg, "drone_usd_path", DEFAULT_DRONE_USD_PATH)).strip()
    if not configured:
        configured = DEFAULT_DRONE_USD_PATH
    if override:
        candidate = Path(override)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            "BADMINTON_AIR_USD does not point to an existing USD file: "
            f"{override}"
        )

    candidate_paths: list[Path] = []
    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            candidate_paths.append(configured_path)
        else:
            candidate_paths.append(_PROJECT_ROOT / configured_path)

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    # Keep this strict for early error surfacing in IsaacLab mode.
    tried = ", ".join(str(path) for path in candidate_paths) or "<empty>"
    raise FileNotFoundError(
        "Drone USD not found. Set InterceptEnvCfg.drone_usd_path or BADMINTON_AIR_USD to a valid path. "
        f"Tried: {tried}"
    )


def spawn_drone_with_racket(asset_path: str) -> str:
    """Create drone actor from the USD path (fallback placeholder)."""
    return f"drone({asset_path})"


def build_scene(config: dict[str, Any]) -> SceneHandles | SceneEntityConfigs:
    """Build IsaacLab scene entity configs or fallback string handles."""
    cfg = config.get("cfg")

    if not ISAACLAB_SCENE_AVAILABLE:
        return SceneHandles()

    air_usd_path = _resolve_air_usd_path(cfg)
    drone_kind = _resolve_drone_asset_kind(cfg)

    if drone_kind == "rigid_object":
        drone_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Drone",
            spawn=sim_utils.UsdFileCfg(
                usd_path=air_usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(getattr(cfg, "drone_init_pos", (2.0, 0.0, 1.2))),
                rot=tuple(getattr(cfg, "drone_init_rot", (1.0, 0.0, 0.0, 0.0))),
            ),
        )
    else:
        drone_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Drone",
            spawn=sim_utils.UsdFileCfg(
                usd_path=air_usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(getattr(cfg, "drone_init_pos", (2.0, 0.0, 1.2))),
                rot=tuple(getattr(cfg, "drone_init_rot", (1.0, 0.0, 0.0, 0.0))),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )

    shuttlecock_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ShuttlecockProxy",
        spawn=sim_utils.SphereCfg(
            radius=float(getattr(cfg, "shuttle_radius_m", 0.03)),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.7,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=float(getattr(cfg, "shuttle_mass_kg", 0.005))
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(getattr(cfg, "shuttle_init_pos", (-2.0, 0.0, 1.6))),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    net_width = float(getattr(cfg, "net_width_m", 6.5))
    net_height = float(getattr(getattr(cfg, "arena", object()), "net_height", 1.55))
    net_thickness = float(getattr(cfg, "net_thickness_m", 0.05))

    net_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Net",
        spawn=sim_utils.CuboidCfg(
            size=(net_thickness, net_width, net_height),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, net_height * 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    return SceneEntityConfigs(
        drone_cfg=drone_cfg,
        shuttlecock_cfg=shuttlecock_cfg,
        net_cfg=net_cfg,
        drone_kind=drone_kind,
        resolved_air_usd_path=air_usd_path,
    )
