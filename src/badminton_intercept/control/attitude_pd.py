from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def compute_rate_pid_angacc(
    target_body_rate_rad_s: "torch.Tensor",
    current_body_rate_rad_s: "torch.Tensor",
    prev_filtered_body_rate_rad_s: "torch.Tensor",
    integ_error: "torch.Tensor",
    dt: float,
    p_gain: "torch.Tensor",
    i_gain: "torch.Tensor",
    d_gain: "torch.Tensor",
    k_gain: "torch.Tensor",
    integ_limit: "torch.Tensor",
    lpf_alpha: float = 0.803307,
    lpf_beta: float = 0.196693,
):
    """Flightmare-style body-rate PID mapping to desired angular acceleration."""
    if torch is None:
        raise RuntimeError("torch is required for compute_rate_pid_angacc")

    dt = max(float(dt), 1.0e-4)
    rate_error = target_body_rate_rad_s - current_body_rate_rad_s
    filtered_rate = float(lpf_alpha) * current_body_rate_rad_s + float(lpf_beta) * prev_filtered_body_rate_rad_s
    rate_deriv = (filtered_rate - prev_filtered_body_rate_rad_s) / dt

    # FM integral coefficient: 1 - (e / (2.5*pi))^2, clipped to >= 0.
    angacc_des = k_gain * (p_gain * rate_error + i_gain * integ_error - d_gain * rate_deriv)
    int_coef = (1.0 - (rate_error / (2.5 * torch.pi)) ** 2).clamp_min(0.0)
    # Discrete-time integral update should include dt to avoid excessive wind-up.
    integ_error = (integ_error + rate_error * int_coef * dt).clamp(-integ_limit, integ_limit)
    return angacc_des, integ_error, filtered_rate


def angacc_to_torque(angacc_des_rad_s2: "torch.Tensor", inertia_diag: "torch.Tensor"):
    """Convert desired angular acceleration to body torque via diagonal inertia."""
    if torch is None:
        raise RuntimeError("torch is required for angacc_to_torque")
    return angacc_des_rad_s2 * inertia_diag
