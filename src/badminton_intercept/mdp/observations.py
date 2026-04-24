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


def build_serve_hover_observations(
    drone_pos=None,
    drone_quat_w=None,
    drone_lin_vel=None,
    drone_ang_vel=None,
    ball_pos=None,
    ball_lin_vel=None,
    hit_time_s=None,
    hit_point_rel=None,
    hover_target_pos=None,
    post_hit_mask=None,
    serve_hover_mask=None,
    prev_action=None,
    privileged: dict | None = None,
    add_actor_noise: bool = False,
    noise_std: float = 0.01,
) -> dict:
    """Build observations for dual-policy serve-hover training.

    Returned groups:
    - task2_policy: frozen task2 actor input
    - task2_critic_obs: frozen task2 critic input
    - policy: hover actor input = drone_pos + rotmat + lin_vel + ang_vel + prev_action
    - critic: hover critic input
    - post_hit_mask: raw post-hit phase flag
    - serve_hover_mask: delayed phase selector used by the dual policy / PPO mask

    Policy obs structure (23 dims):
        0-2:   drone_pos (3)
        3-11:  rotmat (9)
        12-14: drone_lin_vel (3)
        15-17: drone_ang_vel (3)
        18-21: prev_action (4)
    """
    if torch is None or drone_pos is None:
        zeros_task2_policy = [0.0] * 31
        zeros_task2_critic = [0.0] * 34
        zeros_hover = [0.0] * 23
        return {
            "task2_policy": zeros_task2_policy,
            "task2_critic_obs": zeros_task2_critic,
            "policy": zeros_hover,
            "critic": zeros_hover,
            "post_hit_mask": [[0.0]],
            "serve_hover_mask": [[0.0]],
        }

    task2_obs = build_actor_critic_observations(
        drone_pos=drone_pos,
        drone_quat_w=drone_quat_w,
        drone_lin_vel=drone_lin_vel,
        drone_ang_vel=drone_ang_vel,
        ball_pos=ball_pos,
        ball_lin_vel=ball_lin_vel,
        hit_time_s=hit_time_s,
        hit_point_rel=hit_point_rel,
        privileged=privileged,
        add_actor_noise=add_actor_noise,
        noise_std=noise_std,
    )

    rotmat = _quat_to_rotmat_wxyz(drone_quat_w).reshape(drone_pos.shape[0], 9)
    if post_hit_mask is None:
        post_hit_mask = torch.zeros((drone_pos.shape[0], 1), dtype=drone_pos.dtype, device=drone_pos.device)
    elif post_hit_mask.dim() == 1:
        post_hit_mask = post_hit_mask.unsqueeze(-1)
    if serve_hover_mask is None:
        serve_hover_mask = post_hit_mask
    elif serve_hover_mask.dim() == 1:
        serve_hover_mask = serve_hover_mask.unsqueeze(-1)
    if prev_action is None:
        prev_action = torch.zeros((drone_pos.shape[0], 4), dtype=drone_pos.dtype, device=drone_pos.device)

    hover_actor = torch.cat(
        [
            drone_pos,      # 3
            rotmat,         # 9
            drone_lin_vel,  # 3
            drone_ang_vel,  # 3
            prev_action,    # 4
        ],
        dim=-1,
    )

    if add_actor_noise and noise_std > 0.0:
        hover_actor = hover_actor + noise_std * torch.randn_like(hover_actor)

    hover_critic = hover_actor.clone()
    return {
        "task2_policy": task2_obs["policy"],
        "task2_critic_obs": task2_obs["critic"],
        "policy": hover_actor,
        "critic": hover_critic,
        "post_hit_mask": post_hit_mask.to(dtype=hover_actor.dtype),
        "serve_hover_mask": serve_hover_mask.to(dtype=hover_actor.dtype),
    }
