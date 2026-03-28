from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AirYamlParams:
    mass_kg: float
    inertia_diag: tuple[float, float, float]
    arm_lengths_m: tuple[float, float, float, float]
    rotor_angles_rad: tuple[float, float, float, float]
    directions: tuple[float, float, float, float]
    force_constants: tuple[float, float, float, float]
    moment_constants: tuple[float, float, float, float]
    max_rot_vel_rad_s: tuple[float, float, float, float]
    motor_time_constant_s: float
    noise_scale: float


def _to_float_tuple(values, length: int, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)):
        return fallback
    vals = tuple(float(v) for v in values[:length])
    if len(vals) < length:
        vals = vals + fallback[len(vals):]
    return vals


def load_air_yaml_params(yaml_path: str | Path | None) -> AirYamlParams:
    """Load physical parameters from air.yaml with safe fallbacks."""
    fallback = AirYamlParams(
        mass_kg=0.9505,
        inertia_diag=(0.005290248336341343, 0.005623796273825697, 0.007804622804278718),
        arm_lengths_m=(0.125, 0.125, 0.125, 0.125),
        rotor_angles_rad=(0.78539816, 2.35619449, 3.92699082, 5.49778714),
        directions=(-1.0, 1.0, -1.0, 1.0),
        force_constants=(5.39899e-6, 5.39899e-6, 5.39899e-6, 5.39899e-6),
        moment_constants=(1.52164e-7, 1.52164e-7, 1.52164e-7, 1.52164e-7),
        max_rot_vel_rad_s=(1356.6, 1356.6, 1356.6, 1356.6),
        motor_time_constant_s=0.027159047085755388,
        noise_scale=0.0,
    )

    if yaml_path is None:
        return fallback

    path = Path(yaml_path)
    if not path.exists():
        return fallback

    try:
        import yaml  # type: ignore
    except Exception:
        return fallback

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    inertia = data.get("inertia", {})
    rotor = data.get("rotor_configuration", {})

    return AirYamlParams(
        mass_kg=float(data.get("mass", fallback.mass_kg)),
        inertia_diag=(
            float(inertia.get("xx", fallback.inertia_diag[0])),
            float(inertia.get("yy", fallback.inertia_diag[1])),
            float(inertia.get("zz", fallback.inertia_diag[2])),
        ),
        arm_lengths_m=_to_float_tuple(rotor.get("arm_lengths"), 4, fallback.arm_lengths_m),
        rotor_angles_rad=_to_float_tuple(rotor.get("rotor_angles"), 4, fallback.rotor_angles_rad),
        directions=_to_float_tuple(rotor.get("directions"), 4, fallback.directions),
        force_constants=_to_float_tuple(rotor.get("force_constants"), 4, fallback.force_constants),
        moment_constants=_to_float_tuple(rotor.get("moment_constants"), 4, fallback.moment_constants),
        max_rot_vel_rad_s=_to_float_tuple(rotor.get("max_rotation_velocities"), 4, fallback.max_rot_vel_rad_s),
        motor_time_constant_s=float(rotor.get("time_constant", fallback.motor_time_constant_s)),
        noise_scale=float(rotor.get("noise_scale", fallback.noise_scale)),
    )
