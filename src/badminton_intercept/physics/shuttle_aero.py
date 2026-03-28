from typing import Iterable, Sequence

try:
    import torch
except Exception:
    torch = None


def compute_drag_force(velocity_xyz: Sequence[float], drag_scale: float = 1.0) -> list[float]:
    """Quadratic drag placeholder for shuttlecock proxy."""
    vx, vy, vz = velocity_xyz
    speed_sq = (vx * vx) + (vy * vy) + (vz * vz)
    coeff = -drag_scale * speed_sq
    return [coeff * vx, coeff * vy, coeff * vz]


def batch_drag_forces(velocities: Iterable[Sequence[float]], drag_scale: float = 1.0) -> list[list[float]]:
    return [compute_drag_force(v, drag_scale=drag_scale) for v in velocities]


def compute_drag_force_tensor(
    velocity_w: "torch.Tensor",
    mass_kg: "torch.Tensor",
    drag_length_m: "torch.Tensor",
    eps: float = 1.0e-6,
) -> "torch.Tensor":
    """Compute F_drag = -m * |v| * v / L in world frame."""
    if torch is None:
        raise RuntimeError("Torch is required for tensor drag computation.")

    speed = torch.linalg.norm(velocity_w, dim=-1, keepdim=True)
    mass = mass_kg.unsqueeze(-1)
    drag_len = drag_length_m.unsqueeze(-1).clamp_min(eps)
    return -(mass * speed * velocity_w) / drag_len


def predict_shuttle_trajectory(
    pos_start: "torch.Tensor",
    vel_start: "torch.Tensor",
    mass_kg: "torch.Tensor",
    drag_length_m: "torch.Tensor",
    dt: float = 0.001,
    max_time: float = 5.0,
    gravity: float = 9.81,
) -> "torch.Tensor":
    """Predict shuttlecock trajectory using numerical integration with drag.

    Args:
        pos_start: Initial position (num_envs, 3)
        vel_start: Initial velocity (num_envs, 3)
        mass_kg: Mass of shuttlecock (num_envs,)
        drag_length_m: Drag length parameter L (num_envs,)
        dt: Time step for integration
        max_time: Maximum simulation time
        gravity: Gravitational acceleration

    Returns:
        positions: Trajectory positions (num_steps, num_envs, 3)
    """
    if torch is None:
        raise RuntimeError("Torch is required for trajectory prediction.")

    num_steps = int(max_time / dt)
    num_envs = pos_start.shape[0]
    device = pos_start.device

    pos = pos_start.clone()
    vel = vel_start.clone()

    positions = torch.zeros(num_steps, num_envs, 3, device=device)
    positions[0] = pos

    drag_len = drag_length_m.unsqueeze(-1).clamp_min(1e-6)
    gravity_vec = torch.tensor([0.0, 0.0, -gravity], device=device).unsqueeze(0)

    for step in range(1, num_steps):
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True).clamp_min(1e-6)
        drag_acc = -(speed * vel) / drag_len
        acc = drag_acc + gravity_vec

        vel = vel + acc * dt
        pos = pos + vel * dt

        positions[step] = pos

    return positions


def evaluate_shuttle_trajectory_constraints(
    positions: "torch.Tensor",
    net_x: float = 0.0,
    ground_z: float = 0.0,
    eps: float = 1.0e-6,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Vectorized trajectory constraint evaluation.

    Args:
        positions: Trajectory positions with shape (num_steps, num_envs, 3).
        net_x: Net x-position in the trajectory frame.
        ground_z: Ground z-position in the trajectory frame.
        eps: Numerical stability epsilon.

    Returns:
        z_at_net: Interpolated z at first x crossing (num_envs,), NaN if no crossing.
        has_cross: Whether first crossing exists (num_envs,).
        landing_xy: Interpolated x/y at first ground hit (num_envs, 2), NaN if no landing.
        has_landing: Whether first landing exists (num_envs,).
    """
    if torch is None:
        raise RuntimeError("Torch is required for trajectory constraint evaluation.")

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (T, N, 3), got {tuple(positions.shape)}")

    num_steps, num_envs, _ = positions.shape
    if num_steps < 2:
        device = positions.device
        nan_z = torch.full((num_envs,), float("nan"), device=device)
        nan_xy = torch.full((num_envs, 2), float("nan"), device=device)
        false_mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        return nan_z, false_mask, nan_xy, false_mask

    device = positions.device
    env_idx = torch.arange(num_envs, device=device)

    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]

    # First crossing from left court (x<0) to right court (x>=0).
    cross_mask = (x[:-1] < net_x) & (x[1:] >= net_x)
    has_cross = cross_mask.any(dim=0)
    first_cross_idx = cross_mask.to(torch.int32).argmax(dim=0).clamp(0, num_steps - 2)
    x0 = x[first_cross_idx, env_idx]
    x1 = x[first_cross_idx + 1, env_idx]
    z0 = z[first_cross_idx, env_idx]
    z1 = z[first_cross_idx + 1, env_idx]
    denom_cross = x1 - x0
    denom_cross = torch.where(denom_cross.abs() < eps, torch.full_like(denom_cross, eps), denom_cross)
    alpha_cross = ((net_x - x0) / denom_cross).clamp(0.0, 1.0)
    z_at_net = z0 + alpha_cross * (z1 - z0)
    z_at_net = torch.where(has_cross, z_at_net, torch.full_like(z_at_net, float("nan")))

    # First landing from above ground to below ground.
    land_mask = (z[:-1] > ground_z) & (z[1:] <= ground_z)
    has_landing = land_mask.any(dim=0)
    first_land_idx = land_mask.to(torch.int32).argmax(dim=0).clamp(0, num_steps - 2)
    z0_l = z[first_land_idx, env_idx]
    z1_l = z[first_land_idx + 1, env_idx]
    x0_l = x[first_land_idx, env_idx]
    x1_l = x[first_land_idx + 1, env_idx]
    y0_l = y[first_land_idx, env_idx]
    y1_l = y[first_land_idx + 1, env_idx]
    denom_land = z1_l - z0_l
    denom_land = torch.where(
        denom_land.abs() < eps,
        torch.where(denom_land >= 0.0, torch.full_like(denom_land, eps), torch.full_like(denom_land, -eps)),
        denom_land,
    )
    alpha_land = ((ground_z - z0_l) / denom_land).clamp(0.0, 1.0)
    x_land = x0_l + alpha_land * (x1_l - x0_l)
    y_land = y0_l + alpha_land * (y1_l - y0_l)
    landing_xy = torch.stack([x_land, y_land], dim=-1)
    landing_xy = torch.where(
        has_landing.unsqueeze(-1),
        landing_xy,
        torch.full_like(landing_xy, float("nan")),
    )

    return z_at_net, has_cross, landing_xy, has_landing


def compute_net_crossing(
    positions: "torch.Tensor",
    net_x: float = 0.0,
    net_height: float = 1.55,
) -> "torch.Tensor":
    """Compute height at net crossing for each trajectory.

    Args:
        positions: Trajectory positions (num_steps, num_envs, 3)
        net_x: X coordinate of the net
        net_height: Height of the net

    Returns:
        z_at_net: Z coordinate at net crossing (num_envs,), NaN if never crosses
    """
    z_at_net, _has_cross, _landing_xy, _has_landing = evaluate_shuttle_trajectory_constraints(
        positions=positions,
        net_x=net_x,
        ground_z=0.0,
    )
    return z_at_net


def compute_landing_point(
    positions: "torch.Tensor",
    ground_z: float = 0.0,
) -> "torch.Tensor":
    """Compute landing position for each trajectory.

    Args:
        positions: Trajectory positions (num_steps, num_envs, 3)
        ground_z: Z coordinate of ground

    Returns:
        landing_pos: Landing positions (num_envs, 3), NaN if never lands
    """
    _z_at_net, _has_cross, landing_xy, has_landing = evaluate_shuttle_trajectory_constraints(
        positions=positions,
        net_x=0.0,
        ground_z=ground_z,
    )
    landing_pos = torch.full((positions.shape[1], 3), float("nan"), device=positions.device)
    landing_pos[:, 0:2] = landing_xy
    landing_pos[:, 2] = torch.where(has_landing, torch.full_like(landing_pos[:, 2], ground_z), landing_pos[:, 2])
    return landing_pos
