from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def decode_ctbr_action(
    actions: "torch.Tensor",
    rate_scale_rad_s: float = 3.141592653589793,
    thrust_scale: float = 15.0,
):
    """Decode normalized policy action [-1, 1] to FM-style CTBR targets.

    Args:
        actions: Tensor of shape (N, 4): [wx, wy, wz, thrust] in [-1, 1].
        rate_scale_rad_s: Body-rate scaling factor (default pi rad/s).
        thrust_scale: Dimensionless thrust scaling factor (default 15.0).

    Returns:
        target_body_rate_rad_s: Tensor (N, 3).
        target_thrust_ref: Tensor (N,) where F_des = target_thrust_ref * mass.
    """
    if torch is None:
        raise RuntimeError("torch is required for decode_ctbr_action")
    if actions.ndim != 2 or actions.shape[-1] != 4:
        raise ValueError(f"Expected actions shape (N,4), got {tuple(actions.shape)}")

    actions = torch.tanh(actions)
    target_body_rate_rad_s = actions[:, 0:3] * float(rate_scale_rad_s)
    target_thrust_ref = ((actions[:, 3] + 1.0) * 0.5) * float(thrust_scale)
    return target_body_rate_rad_s, target_thrust_ref
