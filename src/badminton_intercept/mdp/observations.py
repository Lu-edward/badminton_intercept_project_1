from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def _quat_to_rotmat_wxyz(quat_wxyz: "torch.Tensor") -> "torch.Tensor":
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def build_actor_critic_observations(
    drone_pos=None,
    drone_quat_w=None,
    drone_lin_vel=None,
    drone_ang_vel=None,
    ball_pos=None,
    ball_lin_vel=None,
    hit_time_s=None,
    hit_point_rel=None,
    privileged: dict | None = None,
    add_actor_noise: bool = False,
    noise_std: float = 0.01,
) -> dict:
    """Build asymmetric observations for actor/critic.

    Actor: [p, R(9), v, omega, p_ball, v_ball, delta_p_ball, t_hit, delta_p_hit] -> 31 dims
    Critic: actor + [time_norm, ball_mass, drag_length] -> 34 dims
    """
    if torch is None or drone_pos is None:
        actor_obs = {
            "features": [0.0] * 31,
        }
        critic_obs = {"features": [0.0] * 34}
        return {"policy": actor_obs, "critic": critic_obs}

    rel_ball = ball_pos - drone_pos
    rotmat = _quat_to_rotmat_wxyz(drone_quat_w).reshape(drone_pos.shape[0], 9)
    if hit_time_s is None:
        hit_time_s = torch.zeros((drone_pos.shape[0],), dtype=drone_pos.dtype, device=drone_pos.device)
    if hit_point_rel is None:
        hit_point_rel = torch.zeros_like(drone_pos)
    if hit_time_s.dim() == 1:
        hit_time_s = hit_time_s.unsqueeze(-1)

    actor = torch.cat(
        [
            drone_pos,
            rotmat,
            drone_lin_vel,
            drone_ang_vel,
            ball_pos,
            ball_lin_vel,
            rel_ball,
            hit_time_s,
            hit_point_rel,
        ],
        dim=-1,
    )

    if add_actor_noise and noise_std > 0.0:
        actor = actor + noise_std * torch.randn_like(actor)

    if privileged is None:
        critic = actor
    else:
        priv_terms = []
        for key in ["time_norm", "ball_mass", "drag_length"]:
            if key in privileged:
                value = privileged[key]
                if value.dim() == 1:
                    value = value.unsqueeze(-1)
                priv_terms.append(value)
        critic = actor if len(priv_terms) == 0 else torch.cat([actor] + priv_terms, dim=-1)

    return {"policy": actor, "critic": critic}
