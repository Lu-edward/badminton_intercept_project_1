from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


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
    c_hit: float = 70.0,
    c_center: float = 10.0,
    lambda_c: float = 5.0,
    c_pos: float = 5.0,
    c_smooth: float = 2.0,
    c_spin: float = 10.0,
    c_bound: float = 5.0,
    c_ang_vel: float = 0.05,
    c_vert_vel: float = 0.1,
    pos_floor: float = 0.1,
    c_low_shuttle: float = 2.0,
    c_post_hit_tracking: float = 3.0,
) -> dict:
    """Compute sparse+dense reward terms for Task 2 (return shot).
    
    Task 2 特点：
    1. 姿态感知 hit 奖励：只有当球拍法向量 x 分量 < 0 时才给予击球奖励
    2. 甜区击球奖励：根据击球点偏离球拍中心的距离给予奖励
    3. post-hit 阶段的球追踪奖励
    """
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
            "post_hit_tracking": 0.0,
            "posture_aware_hit": 0.0,
        }

    contact_f = contact.float()
    batch_size = racket_pos_w.shape[0]
    
    if has_hit_ball is None:
        has_hit_ball = torch.zeros(batch_size, dtype=torch.bool, device=racket_pos_w.device)

    normal_x = compute_racket_normal_x_component(drone_quat_w)
    correct_posture = (normal_x < 0.0).float()
    r_hit = c_hit * contact_f * correct_posture
    r_posture_aware_hit = r_hit.clone()

    if d_axis is not None:
        r_center = c_center * torch.exp(-lambda_c * d_axis) * contact_f
    else:
        r_center = torch.zeros(batch_size, device=racket_pos_w.device)

    d_3d = torch.linalg.norm(ball_pos_w - racket_pos_w, dim=-1)
    d_xy = torch.linalg.norm(ball_pos_w[:, :2] - racket_pos_w[:, :2], dim=-1)
    
    dist_for_reward = torch.where(d_xy > 0.5, d_xy, d_3d)
    
    r_margin = 0.05   
    r_pos = torch.where(
        dist_for_reward <= r_margin,
        torch.full_like(dist_for_reward, c_pos),
        c_pos / (1.0 + dist_for_reward - r_margin)
    )

    r_alignment = torch.zeros_like(r_pos)

    if action is None or prev_action is None:
        r_smooth = torch.zeros_like(r_pos)
    else:
        delta_a_sq = torch.sum((action - prev_action) ** 2, dim=-1)
        r_smooth = c_smooth * torch.exp(-delta_a_sq)

    if yaw is None:
        r_spin = torch.zeros_like(r_pos)
    else:
        r_spin = -c_spin * torch.abs(yaw)

    if bound_dist is None:
        r_bound = torch.zeros_like(r_pos)
    else:
        r_bound = -c_bound * bound_dist

    if drone_up_w is None:
        r_tilt = torch.zeros_like(r_pos)
    else:
        r_tilt = (drone_up_w[:, 2] > 0.0).float()

    if drone_ang_vel_w is None:
        r_ang = torch.zeros_like(r_pos)
    else:
        r_ang = -c_ang_vel * torch.linalg.norm(drone_ang_vel_w, dim=-1)

    r_vz = torch.zeros_like(r_pos)

    r_low_shuttle = torch.zeros_like(r_pos)

    r_post_hit = torch.zeros_like(r_pos)
    if ball_vel_w is not None:
        ball_speed = torch.linalg.norm(ball_vel_w, dim=-1)
        ball_moving_towards_opponent = ball_vel_w[:, 0] > 0.0
        post_hit_mask = has_hit_ball.float()
        r_post_hit = c_post_hit_tracking * post_hit_mask * ball_moving_towards_opponent.float() * torch.tanh(ball_speed)

    total = (r_hit + r_center + r_pos + r_alignment + r_smooth + 
             r_spin + r_bound + r_tilt + r_ang + r_vz + r_low_shuttle + r_post_hit)
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
        "post_hit_tracking": r_post_hit,
        "posture_aware_hit": r_posture_aware_hit,
    }
