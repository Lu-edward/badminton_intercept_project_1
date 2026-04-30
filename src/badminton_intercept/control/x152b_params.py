from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class X152bParams:
    """Physical and controller constants for the AirGym X152b model.usd asset."""

    mass_kg: float = 0.641
    inertia_diag: tuple[float, float, float] = (0.040545, 0.040545, 0.040288)

    rotor_positions_m: tuple[tuple[float, float, float], ...] = (
        (0.05374, -0.05374, 0.024),
        (-0.05374, 0.05374, 0.024),
        (0.05374, 0.05374, 0.024),
        (-0.05374, -0.05374, 0.024),
    )
    rotor_directions: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)

    px4_mixer_roll_scale: tuple[float, float, float, float] = (-0.707107, 0.707107, 0.707107, -0.707107)
    px4_mixer_pitch_scale: tuple[float, float, float, float] = (-0.707107, 0.707107, -0.707107, 0.707107)
    px4_mixer_yaw_scale: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)
    px4_mixer_thrust_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    airgym_thrust_scale_n: float = 9.59
    airgym_torque_thrust_ratio_m: float = 0.2
    airgym_hover_thrust_norm: float = 0.1635

    rate_p_gain: tuple[float, float, float] = (0.5, 0.5, 0.2)
    rate_i_gain: tuple[float, float, float] = (0.08, 0.08, 0.05)
    rate_d_gain: tuple[float, float, float] = (0.001, 0.001, 0.0)
    rate_integral_limit: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rate_control_limit: tuple[float, float, float] = (1.0, 1.0, 1.0)


DEFAULT_X152B_PARAMS = X152bParams()
