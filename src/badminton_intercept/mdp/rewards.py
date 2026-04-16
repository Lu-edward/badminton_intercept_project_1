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


def compute_rewards(
    racket_pos_w=None,
    ball_pos_w=None,
    ball_vel_w=None,
    contact=None,
    action=None,
    prev_action=None,
    yaw=None,
    d_axis=None,
    bound_dist=None,
    drone_up_w=None,
    drone_ang_vel_w=None,
    drone_lin_vel_w=None,
    drone_pos_w=None,
    episode_length_buf=None,
    max_episode_length=None,
    c_hit: float = 20.0,
    c_center: float = 10.0,
    lambda_c: float = 5.0,
    c_pos: float = 5, #5.0
    c_smooth: float = 2.0,#2.0本来
    c_spin: float = 10.0,
    c_bound: float = 5.0,#2.5原来
    c_ang_vel: float = 0.05,#0.05本来
    c_vert_vel: float = 0.1,
    pos_floor: float = 0.1,#0.2本来
    c_low_shuttle: float = 2.0,#新增
    c_hit_velocity_bonus: float = 70.0,
) -> dict:
    """Compute sparse+dense reward terms for interception."""
    if torch is None or racket_pos_w is None:
        return {
            "total": 0.0,
            "dense_tracking": 0.0,
            "sparse_hit": 0.0,
            "sweet_spot_bonus": 0.0,
            "alignment_reward": 0.0,
            "smoothness": 0.0,
            "spin_penalty": 0.0,
            "boundary_penalty": 0.0,
            "tilt_reward": 0.0,
            "ang_vel_penalty": 0.0,
            "vert_vel_penalty": 0.0,
            "low_shuttle_penalty": 0.0,
        }

    contact_f = contact.float()

    # 击中球的奖励：基础hit奖励 + 速度方向奖励
    # 基础hit奖励：只要击中就给20
    r_hit_base = c_hit * contact_f
    
    # 速度方向奖励：无人机在x轴上的线速度分量指向负方向且大于4m/s，给予70奖励
    r_hit_velocity_bonus = torch.zeros_like(contact_f)
    if drone_lin_vel_w is not None:
        # 无人机在x轴上的线速度分量
        drone_vel_x = drone_lin_vel_w[:, 0]
        # 条件：指向负方向 (< 0) 且速度大小 > 4m/s
        velocity_condition = (drone_vel_x < -4.0).float()
        r_hit_velocity_bonus = c_hit_velocity_bonus * contact_f * velocity_condition
    
    r_hit = r_hit_base + r_hit_velocity_bonus

    # 位置追踪奖励：球拍中心与球的距离
        # 3.20修改：水平距离>50cm时用水平距离，<50cm时用3D距离
    d_3d = torch.linalg.norm(ball_pos_w - racket_pos_w, dim=-1)
    d_xy = torch.linalg.norm(ball_pos_w[:, :2] - racket_pos_w[:, :2], dim=-1)  # 水平距离
    
    # 根据水平距离选择使用哪种距离
    dist_for_reward = torch.where(d_xy > 0.5, d_xy, d_3d)  # >50cm用d_xy，<=50cm用d_3d
    
    r_margin = 0.05   
    r_pos = torch.where(
        dist_for_reward <= r_margin,
        torch.full_like(dist_for_reward, c_pos),  # 以内：最大奖励
        c_pos / (1.0 + dist_for_reward - r_margin)  # 超过5cm：距离越大奖励越低
    )

    #甜区奖励（已禁用）
    r_center = torch.zeros_like(r_pos)
    
    # 法向量对齐奖励（已禁用，3.20新增）
    # 当距离 < 70cm时，奖励球拍法向量（无人机Z轴）对齐指向球的方向
    r_alignment = torch.zeros_like(r_pos)
    
    # 原来的水平位置奖励（已禁用）
    # r_pos = c_pos / (1.0 + torch.maximum(d_xy, torch.full_like(d_xy, pos_floor)))

    #动作平滑奖励
    if action is None or prev_action is None:
        r_smooth = torch.zeros_like(r_pos)
    else:
        delta_a_sq = torch.sum((action - prev_action) ** 2, dim=-1)
        r_smooth = c_smooth * torch.exp(-delta_a_sq)

    #限制yaw旋转
    if yaw is None:
        r_spin = torch.zeros_like(r_pos)
    else:
        r_spin = -c_spin * torch.abs(yaw)

    #出界惩罚
    if bound_dist is None:
        r_bound = torch.zeros_like(r_pos)
    else:
        r_bound = -c_bound * bound_dist

    # 无人机倾斜奖励：奖励无人机保持正立（Z轴朝上）
    if drone_up_w is None:
        r_tilt = torch.zeros_like(r_pos)
    else:
        r_tilt = (drone_up_w[:, 2] > 0.0).float()

    # 角速度惩罚
    if drone_ang_vel_w is None:
        r_ang = torch.zeros_like(r_pos)
    else:
        r_ang = -c_ang_vel * torch.linalg.norm(drone_ang_vel_w, dim=-1)
    #r_ang = torch.zeros_like(r_pos)

    # 垂直速度惩罚（已禁用）
    # if drone_lin_vel_w is None:
    #     r_vz = torch.zeros_like(r_pos)
    # else:
    #     r_vz = -c_vert_vel * torch.abs(drone_lin_vel_w[:, 2])
    r_vz = torch.zeros_like(r_pos)

    # 低羽毛球惩罚（已禁用，改为在termination中处理）
    # if racket_pos_w is not None and ball_pos_w is not None:
    #     racket_z_threshold = racket_pos_w[:, 2] - 0.05
    #     in_drone_half = ball_pos_w[:, 0] > 0.0
    #     below_threshold = ball_pos_w[:, 2] < racket_z_threshold
    #     r_low_shuttle = -c_low_shuttle * in_drone_half.float() * below_threshold.float()
    # else:
    #     r_low_shuttle = torch.zeros_like(r_pos)
    r_low_shuttle = torch.zeros_like(r_pos)

    total = r_hit + r_center + r_pos + r_alignment + r_smooth + r_spin + r_bound + r_tilt + r_ang + r_vz + r_low_shuttle
    total = torch.nan_to_num(total, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    return {
        "total": total,
        "dense_tracking": r_pos,
        "sparse_hit": r_hit,
        "sweet_spot_bonus": r_center,
        "alignment_reward": r_alignment,
        "smoothness": r_smooth,
        "spin_penalty": r_spin,
        "boundary_penalty": r_bound,
        "tilt_reward": r_tilt,
        "ang_vel_penalty": r_ang,
        "vert_vel_penalty": r_vz,
        "low_shuttle_penalty": r_low_shuttle,
    }


def compute_racket_normal_x_component(drone_quat_w):
    """计算球拍平面法向量在 x 轴的分量。
    
    球拍法向量假设与无人机 Z 轴对齐。
    返回法向量在世界坐标系 x 轴的分量。
    
    Args:
        drone_quat_w: 无人机四元数 (w, x, y, z)，形状为 (N, 4)
    
    Returns:
        法向量在世界坐标系 x 轴的分量，形状为 (N,)
    """
    w = drone_quat_w[:, 0]
    x = drone_quat_w[:, 1]
    y = drone_quat_w[:, 2]
    z = drone_quat_w[:, 3]
    
    normal_x = 2.0 * (x * z + w * y)
    return normal_x


def compute_rewards_task2(
    racket_pos_w=None,
    ball_pos_w=None,
    ball_vel_w=None,
    contact=None,
    action=None,
    prev_action=None,
    yaw=None,
    d_axis=None,
    bound_dist=None,
    drone_up_w=None,
    drone_ang_vel_w=None,
    drone_lin_vel_w=None,
    drone_pos_w=None,
    drone_quat_w=None,
    episode_length_buf=None,
    max_episode_length=None,
    has_hit_ball=None,
    c_hit: float = 50.0,
    c_center: float = 5.0,
    lambda_c: float = 5.0,
    c_pos: float = 5.0,
    c_smooth: float = 0.5,
    c_spin: float = 10.0,
    c_bound: float = 5.0,
    c_ang_vel: float = 0.05,
    c_vert_vel: float = 0.1,
    pos_floor: float = 0.1,
    c_low_shuttle: float = 2.0,
    c_rpos_anchor: float = 100.0,
    anchor_pos=(-2, 0, 1.55),
    w_vel: float = 5.0,
    v_threshold: float = 1.0,
) -> dict:
    """Compute rewards for Task 2 (return shot), organized by phase.

    Phase 1 - 击球阶段 (pre-hit):
      r_hit:       姿态感知击中奖励（球拍法向量 x < 0 才给）
      r_center:   甜区奖励
      r_pos:       位置追踪奖励（球拍靠近球）

    Phase 2 - 击球后 (post-hit):
      r_pos_anchor: 球靠近目标落点的奖励 25.0 / (1 + 2 * ||ball - anchor||²)

    Phase 3 - 通用（与终止条件相关）:
      r_smooth, r_spin, r_bound, r_tilt, r_ang, r_vz, r_low_shuttle
    """
    if torch is None or racket_pos_w is None:
        return {
            "total": 0.0,
            "sparse_hit": 0.0,
            "hit_velocity": 0.0,
            "sweet_spot_bonus": 0.0,
            "dense_tracking": 0.0,
            "post_hit_pos_anchor": 0.0,
            "smoothness": 0.0,
            "spin_penalty": 0.0,
            "boundary_penalty": 0.0,
            "tilt_reward": 0.0,
            "ang_vel_penalty": 0.0,
            "vert_vel_penalty": 0.0,
            "low_shuttle_penalty": 0.0,
        }

    contact_f = contact.float()
    batch_size = racket_pos_w.shape[0]
    device = racket_pos_w.device

    if has_hit_ball is None:
        has_hit_ball = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # ── Phase 1: 击球阶段 ──────────────────────────────────────────────
    normal_x = compute_racket_normal_x_component(drone_quat_w)
    correct_posture = (normal_x < 0.0).float()
    r_hit = c_hit * contact_f * correct_posture

    # Disable hit-velocity bonus for task2 (ablation requested).
    r_hit_velocity = torch.zeros(batch_size, device=device)

    if d_axis is not None:
        r_center = c_center * torch.exp(-lambda_c * d_axis) * contact_f
    else:
        r_center = torch.zeros(batch_size, device=device)

    # ── Phase 2: 击球后 ────────────────────────────────────────────────
    anchor = torch.tensor(anchor_pos, dtype=racket_pos_w.dtype, device=device)
    dist_sq = torch.sum((ball_pos_w - anchor) ** 2, dim=-1)
    r_pos_anchor = c_rpos_anchor / (1.0 + 2 * dist_sq)
    # 仅在 has_hit_ball 后生效
    r_pos_anchor = r_pos_anchor * has_hit_ball.float()

    # ── Phase 3: 通用（仅 pre-hit 阶段生效，post-hit 阶段全部归零） ──
    pre_hit_mask = ~has_hit_ball

    r_x_boundary = -100.0 * (drone_pos_w[:, 0] < 0.1).float() * pre_hit_mask if drone_pos_w is not None else torch.zeros(batch_size, device=device)

    if action is None or prev_action is None:
        r_smooth = torch.zeros(batch_size, device=device)
    else:
        delta_a_sq = torch.sum((action - prev_action) ** 2, dim=-1)
        r_smooth = c_smooth * torch.exp(-delta_a_sq) * pre_hit_mask

    if yaw is None:
        r_spin = torch.zeros(batch_size, device=device)
    else:
        r_spin = -c_spin * torch.abs(yaw) * pre_hit_mask

    if bound_dist is None:
        r_bound = torch.zeros(batch_size, device=device)
    else:
        r_bound = -c_bound * bound_dist * pre_hit_mask

    if drone_up_w is None:
        r_tilt = torch.zeros(batch_size, device=device)
    else:
        r_tilt = (drone_up_w[:, 2] > 0.0).float() * pre_hit_mask

    if drone_ang_vel_w is None:
        r_ang = torch.zeros(batch_size, device=device)
    else:
        r_ang = -c_ang_vel * torch.linalg.norm(drone_ang_vel_w, dim=-1) * pre_hit_mask

    # 位置追踪奖励（仅 pre-hit 阶段生效）
    d_3d = torch.linalg.norm(ball_pos_w - racket_pos_w, dim=-1)
    d_xy = torch.linalg.norm(ball_pos_w[:, :2] - racket_pos_w[:, :2], dim=-1)
    dist_for_reward = torch.where(d_xy > 0.5, d_xy, d_3d)
    r_margin = 0.05
    r_pos = torch.where(
        dist_for_reward <= r_margin,
        torch.full_like(dist_for_reward, c_pos),
        c_pos / (1.0 + dist_for_reward - r_margin)
    )
    r_pos = r_pos * pre_hit_mask

    r_vz = torch.zeros(batch_size, device=device)
    r_low_shuttle = torch.zeros(batch_size, device=device)

    total = (
        r_hit + r_center
        + r_pos_anchor
        + r_smooth + r_spin + r_bound + r_tilt + r_ang + r_vz + r_low_shuttle + r_pos + r_x_boundary
    )
    total = torch.nan_to_num(total, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    return {
        "total": total,
        "sparse_hit": r_hit,
        "hit_velocity": r_hit_velocity,
        "sweet_spot_bonus": r_center,
        "dense_tracking": r_pos,
        "post_hit_pos_anchor": r_pos_anchor,
        "smoothness": r_smooth,
        "spin_penalty": r_spin,
        "boundary_penalty": r_bound,
        "tilt_reward": r_tilt,
        "ang_vel_penalty": r_ang,
        "vert_vel_penalty": r_vz,
        "low_shuttle_penalty": r_low_shuttle,
        "x_boundary_penalty": r_x_boundary,
    }


def compute_rewards_serve_hover(
    drone_pos=None,
    drone_up_w=None,
    drone_ang_vel_w=None,
    drone_quat_w=None,
    has_hit_ball=None,
    hover_target_pos=(-2.0, 0.0, 1.25),
) -> dict:
    """Compute post-hit hovering rewards.

    All positions are in LOCAL (court) frame.
    The hover policy is only trained after a valid hit. Before that point the
    reward is exactly zero so pre-hit task2 behavior stays frozen.

    Reward formula:
        heading = body X axis in world frame
        rheading = [-1, 0, 0] - heading  (期望朝向 -x)
        distance = L2(concat(rpos_hover, rheading))
        reward_pose = 1.0 / (1.0 + (1.2 * distance)^2)
        reward_up = ((up_z + 1) / 2)^2
        reward_spin = 1.0 / (1.0 + ang_z^2)
        reward_hover = 3.0 * (reward_pose + reward_pose * (reward_up + reward_spin))
    """
    if torch is None or drone_pos is None:
        return {
            "total": 0.0,
            "reward_hover": 0.0,
            "reward_pose": 0.0,
            "reward_up": 0.0,
            "reward_spin": 0.0,
        }

    batch_size = drone_pos.shape[0]
    device = drone_pos.device
    dtype = drone_pos.dtype
    if has_hit_ball is None:
        has_hit_ball = torch.zeros(batch_size, dtype=torch.bool, device=device)

    active = has_hit_ball.float()

    # heading = body X axis in world frame (first column of rotmat)
    if drone_quat_w is not None:
        rotmat = _quat_to_rotmat_wxyz(drone_quat_w).reshape(batch_size, 9)
        heading = torch.stack([rotmat[:, 0], rotmat[:, 3], rotmat[:, 6]], dim=-1)
    else:
        heading = torch.zeros(batch_size, 3, device=device, dtype=dtype)

    target_heading = torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, device=device)
    rheading = target_heading - heading

    # rpos_hover
    target_pos = torch.tensor(hover_target_pos, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1)
    rpos_hover = drone_pos - target_pos

    # 6D L2 distance: concat position error + heading error
    combined = torch.cat([rpos_hover, rheading], dim=-1)
    distance = torch.linalg.norm(combined, dim=-1)

    # reward_pose
    reward_pose = 1.0 / (1.0 + (1.2 * distance) ** 2)

    # reward_up
    if drone_up_w is not None:
        up_z = drone_up_w[:, 2]
        reward_up = ((up_z + 1.0) / 2.0) ** 2
    else:
        reward_up = torch.zeros(batch_size, device=device, dtype=dtype)

    # reward_spin (only z-axis angular velocity)
    if drone_ang_vel_w is not None:
        ang_z = drone_ang_vel_w[:, 2]
        reward_spin = 1.0 / (1.0 + ang_z ** 2)
    else:
        reward_spin = torch.zeros(batch_size, device=device, dtype=dtype)

    # 综合 hover 奖励
    reward_hover = 3.0 * (reward_pose + reward_pose * (reward_up + reward_spin))
    reward_hover = reward_hover * active

    total = torch.nan_to_num(reward_hover, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
    return {
        "total": total,
        "reward_hover": reward_hover,
        "reward_pose": reward_pose * active,
        "reward_up": reward_up * active,
        "reward_spin": reward_spin * active,
    }
