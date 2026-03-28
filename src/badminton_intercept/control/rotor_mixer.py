from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def build_mixer_matrix(
    arm_lengths: "torch.Tensor",
    rotor_angles: "torch.Tensor",
    directions: "torch.Tensor",
    force_constants: "torch.Tensor",
    moment_constants: "torch.Tensor",
):
    """Build mapping from [ang_acc_x, ang_acc_y, ang_acc_z, thrust] to per-rotor thrust."""
    if torch is None:
        raise RuntimeError("torch is required for build_mixer_matrix")

    if arm_lengths.numel() == 1:
        arm_lengths = arm_lengths.repeat(rotor_angles.shape[0])

    yaw_coeff = -directions * (moment_constants / force_constants.clamp_min(1.0e-9))
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            yaw_coeff,
            torch.ones_like(rotor_angles),
        ],
        dim=0,
    )
    return A.T @ torch.linalg.inv(A @ A.T)


def mix_angacc_thrust_to_rotor_thrust(
    mixer_matrix: "torch.Tensor",
    angacc_des_rad_s2: "torch.Tensor",
    collective_thrust_n: "torch.Tensor",
):
    """Mix desired [angular acceleration, collective thrust] to per-rotor thrust."""
    if torch is None:
        raise RuntimeError("torch is required for mix_angacc_thrust_to_rotor_thrust")

    if collective_thrust_n.ndim == 1:
        collective_thrust_n = collective_thrust_n.unsqueeze(-1)
    target = torch.cat([angacc_des_rad_s2, collective_thrust_n], dim=-1)
    return target @ mixer_matrix.T


def rotor_thrust_to_body_wrench(
    rotor_thrust_n: "torch.Tensor",
    arm_lengths: "torch.Tensor",
    rotor_angles: "torch.Tensor",
    directions: "torch.Tensor",
    force_constants: "torch.Tensor",
    moment_constants: "torch.Tensor",
):
    """Aggregate rotor thrusts into body-frame force and torque."""
    if torch is None:
        raise RuntimeError("torch is required for rotor_thrust_to_body_wrench")

    if arm_lengths.numel() == 1:
        arm_lengths = arm_lengths.repeat(rotor_angles.shape[0])

    x = arm_lengths * torch.cos(rotor_angles)
    y = arm_lengths * torch.sin(rotor_angles)
    yaw_coeff = -directions * (moment_constants / force_constants.clamp_min(1.0e-9))

    tau_x = (y.unsqueeze(0) * rotor_thrust_n).sum(dim=-1)
    tau_y = (-x.unsqueeze(0) * rotor_thrust_n).sum(dim=-1)
    tau_z = (yaw_coeff.unsqueeze(0) * rotor_thrust_n).sum(dim=-1)

    body_force = torch.zeros((rotor_thrust_n.shape[0], 3), device=rotor_thrust_n.device, dtype=rotor_thrust_n.dtype)
    body_force[:, 2] = rotor_thrust_n.sum(dim=-1)

    body_torque = torch.stack([tau_x, tau_y, tau_z], dim=-1)
    return body_force, body_torque
