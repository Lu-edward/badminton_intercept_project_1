from __future__ import annotations

from dataclasses import dataclass

from badminton_intercept.control.x152b_airgym_mixer import mix_px4_quad_x_to_airgym_wrench
from badminton_intercept.control.x152b_params import DEFAULT_X152B_PARAMS, X152bParams

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class X152bCtbrControlOutput:
    body_force: "torch.Tensor"
    body_torque: "torch.Tensor"
    rotor_pwm: "torch.Tensor"
    rotor_force_n: "torch.Tensor"
    target_body_rate_rad_s: "torch.Tensor"
    rate_control: "torch.Tensor"
    rate_integral: "torch.Tensor"
    filtered_body_rate_rad_s: "torch.Tensor"


def _gain_tensor(values, *, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)


def compute_x152b_rate_control(
    target_body_rate_rad_s: "torch.Tensor",
    body_rate_rad_s: "torch.Tensor",
    prev_filtered_body_rate_rad_s: "torch.Tensor",
    rate_integral: "torch.Tensor",
    dt: float,
    params: X152bParams = DEFAULT_X152B_PARAMS,
    lpf_alpha: float = 0.803307,
    lpf_beta: float = 0.196693,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """PX4-style rate PID that returns normalized roll/pitch/yaw controls."""
    if torch is None:
        raise RuntimeError("torch is required for compute_x152b_rate_control")
    if target_body_rate_rad_s.shape != body_rate_rad_s.shape or target_body_rate_rad_s.shape[-1] != 3:
        raise ValueError("target_body_rate_rad_s and body_rate_rad_s must both have shape (N,3)")

    dt = max(float(dt), 1.0e-4)
    device = body_rate_rad_s.device
    dtype = body_rate_rad_s.dtype
    p_gain = _gain_tensor(params.rate_p_gain, device=device, dtype=dtype)
    i_gain = _gain_tensor(params.rate_i_gain, device=device, dtype=dtype)
    d_gain = _gain_tensor(params.rate_d_gain, device=device, dtype=dtype)
    integral_limit = _gain_tensor(params.rate_integral_limit, device=device, dtype=dtype)
    control_limit = _gain_tensor(params.rate_control_limit, device=device, dtype=dtype)

    rate_error = target_body_rate_rad_s - body_rate_rad_s
    filtered_rate = float(lpf_alpha) * body_rate_rad_s + float(lpf_beta) * prev_filtered_body_rate_rad_s
    rate_deriv = (filtered_rate - prev_filtered_body_rate_rad_s) / dt
    rate_integral = (rate_integral + rate_error * dt).clamp(-integral_limit, integral_limit)
    rate_control = p_gain * rate_error + i_gain * rate_integral - d_gain * rate_deriv
    rate_control = rate_control.clamp(-control_limit, control_limit)
    return rate_control, rate_integral, filtered_rate


def compute_x152b_ctbr_wrench(
    actions: "torch.Tensor",
    body_rate_rad_s: "torch.Tensor",
    prev_filtered_body_rate_rad_s: "torch.Tensor",
    rate_integral: "torch.Tensor",
    dt: float,
    params: X152bParams = DEFAULT_X152B_PARAMS,
    rate_scale_rad_s: float = 3.141592653589793,
    thrust_min: float = 0.0,
    thrust_max: float = 1.0,
    body_rate_axis_sign: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> X152bCtbrControlOutput:
    """Compute body-frame wrench for X152b from normalized CTBR policy actions."""
    if torch is None:
        raise RuntimeError("torch is required for compute_x152b_ctbr_wrench")
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError(f"Expected actions shape (N,4), got {tuple(actions.shape)}")
    if body_rate_rad_s.ndim != 2 or body_rate_rad_s.shape[-1] != 3:
        raise ValueError(f"Expected body_rate_rad_s shape (N,3), got {tuple(body_rate_rad_s.shape)}")
    if actions.shape[0] != body_rate_rad_s.shape[0]:
        raise ValueError("actions and body_rate_rad_s must have the same batch dimension")

    actions = torch.tanh(actions)
    axis_sign = torch.tensor(body_rate_axis_sign, device=actions.device, dtype=actions.dtype).unsqueeze(0)
    target_body_rate_rad_s = actions[:, 0:3] * float(rate_scale_rad_s) * axis_sign
    thrust_cmd = float(thrust_min) + ((actions[:, 3] + 1.0) * 0.5) * (float(thrust_max) - float(thrust_min))
    thrust_cmd = thrust_cmd.clamp(min=float(thrust_min), max=float(thrust_max))

    rate_control, rate_integral, filtered_rate = compute_x152b_rate_control(
        target_body_rate_rad_s=target_body_rate_rad_s,
        body_rate_rad_s=body_rate_rad_s,
        prev_filtered_body_rate_rad_s=prev_filtered_body_rate_rad_s,
        rate_integral=rate_integral,
        dt=dt,
        params=params,
    )
    normalized_control = torch.cat([rate_control, thrust_cmd.unsqueeze(-1)], dim=-1)
    body_force, body_torque, rotor_pwm, rotor_force_n = mix_px4_quad_x_to_airgym_wrench(
        normalized_control,
        params=params,
    )
    return X152bCtbrControlOutput(
        body_force=body_force,
        body_torque=body_torque,
        rotor_pwm=rotor_pwm,
        rotor_force_n=rotor_force_n,
        target_body_rate_rad_s=target_body_rate_rad_s,
        rate_control=rate_control,
        rate_integral=rate_integral,
        filtered_body_rate_rad_s=filtered_rate,
    )
