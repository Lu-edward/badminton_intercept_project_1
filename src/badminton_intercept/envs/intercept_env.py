from __future__ import annotations

from badminton_intercept.envs.intercept_env_common import DirectRLEnv, ISAACLAB_RUNTIME_AVAILABLE, StepOutput
from badminton_intercept.envs.intercept_env_control import InterceptEnvControlMixin
from badminton_intercept.envs.intercept_env_mdp import InterceptEnvMDPMixin
from badminton_intercept.envs.intercept_env_reset import InterceptEnvResetMixin
from badminton_intercept.envs.intercept_env_runtime import InterceptEnvRuntimeMixin
from badminton_intercept.envs.intercept_env_scene import InterceptEnvSceneMixin
from badminton_intercept.envs.intercept_env_state import InterceptEnvStateMixin


class InterceptEnv(
    InterceptEnvResetMixin,
    InterceptEnvMDPMixin,
    InterceptEnvControlMixin,
    InterceptEnvRuntimeMixin,
    InterceptEnvSceneMixin,
    InterceptEnvStateMixin,
    DirectRLEnv,
):
    """Interception task environment with IsaacLab scene wiring and local fallback mode."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._configure_policy_modes(cfg)
        self._initialize_env_state(cfg)

        if ISAACLAB_RUNTIME_AVAILABLE:
            super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        else:
            super().__init__(cfg=cfg, render_mode=render_mode)
            self.scene = self._setup_scene()

        if self._launch_sampler_mode == "trajectory_library":
            self._ensure_launch_libraries_loaded()


__all__ = ["InterceptEnv", "StepOutput"]
