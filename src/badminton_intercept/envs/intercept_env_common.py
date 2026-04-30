from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:
    torch = None

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sensors import ContactSensor
    from isaaclab.sim.schemas import activate_contact_sensors
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    ISAACLAB_RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback for skeleton stage
    ISAACLAB_RUNTIME_AVAILABLE = False
    sim_utils = None
    Articulation = None
    RigidObject = None
    ContactSensor = None
    activate_contact_sensors = None
    GroundPlaneCfg = None
    spawn_ground_plane = None

    class DirectRLEnv:
        def __init__(self, cfg, render_mode=None):
            self.cfg = cfg
            self.render_mode = render_mode


@dataclass
class StepOutput:
    observations: dict[str, Any]
    rewards: Any
    dones: Any
    infos: dict[str, Any]
