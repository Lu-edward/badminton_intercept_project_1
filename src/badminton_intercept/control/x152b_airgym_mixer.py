from __future__ import annotations

from badminton_intercept.control.x152b_params import DEFAULT_X152B_PARAMS, X152bParams

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _tensor(values, *, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype)


def mix_px4_quad_x_to_pwm(
    normalized_control: "torch.Tensor",
    params: X152bParams = DEFAULT_X152B_PARAMS,
    clamp: bool = True,
) -> "torch.Tensor":
    """Mix normalized [roll, pitch, yaw, thrust] commands to X152b PWM values."""
    if torch is None:
        raise RuntimeError("torch is required for mix_px4_quad_x_to_pwm")
    if normalized_control.ndim != 2 or normalized_control.shape[-1] != 4:
        raise ValueError(f"Expected normalized_control shape (N,4), got {tuple(normalized_control.shape)}")

    device = normalized_control.device
    dtype = normalized_control.dtype
    scales = torch.stack(
        [
            _tensor(params.px4_mixer_roll_scale, device=device, dtype=dtype),
            _tensor(params.px4_mixer_pitch_scale, device=device, dtype=dtype),
            _tensor(params.px4_mixer_yaw_scale, device=device, dtype=dtype),
            _tensor(params.px4_mixer_thrust_scale, device=device, dtype=dtype),
        ],
        dim=0,
    )
    pwm = normalized_control @ scales
    if clamp:
        pwm = pwm.clamp(0.0, 1.0)
    return pwm


def airgym_pwm_to_rotor_force(
    rotor_pwm: "torch.Tensor",
    params: X152bParams = DEFAULT_X152B_PARAMS,
) -> "torch.Tensor":
    """Convert AirGym-style PWM to per-rotor force with F = PWM * 9.59 N."""
    if torch is None:
        raise RuntimeError("torch is required for airgym_pwm_to_rotor_force")
    if rotor_pwm.ndim != 2 or rotor_pwm.shape[-1] != 4:
        raise ValueError(f"Expected rotor_pwm shape (N,4), got {tuple(rotor_pwm.shape)}")
    return rotor_pwm.clamp(0.0, 1.0) * float(params.airgym_thrust_scale_n)


def airgym_rotor_force_to_body_wrench(
    rotor_force_n: "torch.Tensor",
    params: X152bParams = DEFAULT_X152B_PARAMS,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Aggregate X152b rotor forces into body-frame force and torque."""
    if torch is None:
        raise RuntimeError("torch is required for airgym_rotor_force_to_body_wrench")
    if rotor_force_n.ndim != 2 or rotor_force_n.shape[-1] != 4:
        raise ValueError(f"Expected rotor_force_n shape (N,4), got {tuple(rotor_force_n.shape)}")

    device = rotor_force_n.device
    dtype = rotor_force_n.dtype
    positions = _tensor(params.rotor_positions_m, device=device, dtype=dtype)
    directions = _tensor(params.rotor_directions, device=device, dtype=dtype)

    tau_x = (positions[:, 1].unsqueeze(0) * rotor_force_n).sum(dim=-1)
    tau_y = (-positions[:, 0].unsqueeze(0) * rotor_force_n).sum(dim=-1)
    tau_z = (directions.unsqueeze(0) * rotor_force_n).sum(dim=-1) * float(params.airgym_torque_thrust_ratio_m)

    body_force = torch.zeros((rotor_force_n.shape[0], 3), device=device, dtype=dtype)
    body_force[:, 2] = rotor_force_n.sum(dim=-1)
    body_torque = torch.stack([tau_x, tau_y, tau_z], dim=-1)
    return body_force, body_torque


def mix_px4_quad_x_to_airgym_wrench(
    normalized_control: "torch.Tensor",
    params: X152bParams = DEFAULT_X152B_PARAMS,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Return body wrench plus intermediate PWM and rotor force for a PX4 control vector."""
    rotor_pwm = mix_px4_quad_x_to_pwm(normalized_control, params=params, clamp=True)
    rotor_force_n = airgym_pwm_to_rotor_force(rotor_pwm, params=params)
    body_force, body_torque = airgym_rotor_force_to_body_wrench(rotor_force_n, params=params)
    return body_force, body_torque, rotor_pwm, rotor_force_n
