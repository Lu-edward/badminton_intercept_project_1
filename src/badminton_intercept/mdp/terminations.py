from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def compute_dones_task2(
    ball_pos_w=None,
    net_contact=None,
    drone_net_collision=None,
    z_threshold: float = 0.1,
    max_ball_height: float = 7.0,
    court_bounds: dict | None = None,
    episode_length_buf=None,
    max_episode_length: int | None = None,
    return_reason_masks: bool = False,
):
    """Compute termination signals and rewards for Task 2 (post-hit phase).

    This function handles ONLY the post-hit phase. It assumes all environments
    have already transitioned past the hit moment.

    Termination conditions (post-hit, no timeout):
    1. Ball touches the net
    2. Ball touches the ground (drone half x>0 or server half x<0)
    3. Ball goes out of bounds (both halves)
    4. Ball height exceeds 7m
    5. Drone/racket touches the net

    NOTE: There is NO episode timeout for post-hit. The episode runs until the
    ball actually lands/goes out, regardless of simulated time elapsed.

    Reward rules:
    - Server half ground: +70
    - Server half out of bounds (no ground): +10
    - All other conditions: 0

    Args:
        ball_pos_w: Ball position in local court frame, shape (N, 3)
        net_contact: Boolean tensor for net contact, shape (N,)
        drone_net_collision: Boolean tensor for drone/racket-net contact, shape (N,)
        z_threshold: Ground threshold for ball grounded detection
        max_ball_height: Maximum allowed ball height (default 7.0m)
        court_bounds: Dict with 'x' and 'y' bounds, e.g., {'x': (-6.7, 6.7), 'y': (-3.05, 3.05)}
        episode_length_buf: Unused, kept for API compatibility
        max_episode_length: Unused, kept for API compatibility
        return_reason_masks: If True, return reason masks dict

    Returns:
        If return_reason_masks is False:
            terminated: Boolean tensor of termination flags
            rewards: Float tensor of rewards for each environment
            truncated: Boolean tensor (always False, no timeout in post-hit)
        If return_reason_masks is True:
            terminated: Boolean tensor of termination flags
            rewards: Float tensor of rewards for each environment
            truncated: Boolean tensor (always False, no timeout in post-hit)
            reason_masks: Dict of reason masks and reward masks
    """
    if torch is None or ball_pos_w is None:
        if return_reason_masks:
            return False, 0.0, False, {}
        return False, 0.0, False

    device = ball_pos_w.device
    num_envs = ball_pos_w.shape[0]

    if net_contact is None:
        net_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if drone_net_collision is None:
        drone_net_collision = torch.zeros(num_envs, dtype=torch.bool, device=device)

    if court_bounds is None:
        court_bounds = {
            "x": (-6.7, 6.7),
            "y": (-3.05, 3.05),
        }

    ball_x = ball_pos_w[:, 0]
    ball_y = ball_pos_w[:, 1]
    ball_z = ball_pos_w[:, 2]

    x_min, x_max = court_bounds["x"]
    y_min, y_max = court_bounds["y"]

    in_drone_half = ball_x > 0.0
    in_server_half = ~in_drone_half

    net_contact_flag = net_contact
    drone_net_collision_flag = drone_net_collision

    ball_grounded = ball_z <= z_threshold
    drone_half_grounded = in_drone_half & ball_grounded
    server_half_grounded = in_server_half & ball_grounded

    ball_out_x = (ball_x < x_min) | (ball_x > x_max)
    ball_out_y = (ball_y < y_min) | (ball_y > y_max)
    ball_out_of_bounds = ball_out_x | ball_out_y
    drone_half_out = in_drone_half & ball_out_of_bounds
    server_half_out = in_server_half & ball_out_of_bounds

    ball_too_high = ball_z > max_ball_height

    terminated = (
        net_contact_flag
        | drone_half_grounded
        | server_half_grounded
        | drone_half_out
        | server_half_out
        | ball_too_high
        | drone_net_collision_flag
    )

    rewards = torch.zeros(num_envs, dtype=torch.float32, device=device)
    rewards = torch.where(server_half_grounded, torch.tensor(70.0, device=device), rewards)
    rewards = torch.where(server_half_out & (~server_half_grounded), torch.tensor(10.0, device=device), rewards)

    # No episode timeout in post-hit: ball runs to completion
    truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)

    if not return_reason_masks:
        return terminated, rewards, truncated

    reason_masks = {
        "net_contact": net_contact_flag,
        "drone_half_grounded": drone_half_grounded,
        "server_half_grounded": server_half_grounded,
        "drone_half_out": drone_half_out,
        "server_half_out": server_half_out,
        "ball_too_high": ball_too_high,
        "drone_net_collision": drone_net_collision_flag,
        "reward_server_half_grounded": server_half_grounded,
        "reward_server_half_out": server_half_out & (~server_half_grounded),
    }
    return terminated, rewards, truncated, reason_masks


def compute_dones_serve_hover(
    drone_pos_w=None,
    drone_bottom_z=None,
    contact=None,
    has_hit_ball=None,
    correct_posture=None,
    drone_net_collision=None,
    episode_length_buf=None,
    max_episode_length: int | None = None,
    min_height: float = 0.1,
    max_height: float = 4.0,
    return_reason_masks: bool = False,
):
    """Termination logic for serve-hover training.

    Active termination conditions:
    - drone height out of range
    - drone x out of range (x <= 0.1)
    - wrong_hit (pre-hit invalid contact or post-hit contact)
    - drone_net_collision (drone body hits net after ball is hit)
    - timeout
    """
    if torch is None or drone_pos_w is None:
        if return_reason_masks:
            return False, False, {}, 0.0
        return False, False

    device = drone_pos_w.device
    num_envs = drone_pos_w.shape[0]

    if contact is None:
        contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if has_hit_ball is None:
        has_hit_ball = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if correct_posture is None:
        correct_posture = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if drone_net_collision is None:
        drone_net_collision = torch.zeros(num_envs, dtype=torch.bool, device=device)

    low_height = (
        drone_bottom_z < min_height
        if drone_bottom_z is not None
        else (drone_pos_w[:, 2] < min_height)
    )
    high_height = drone_pos_w[:, 2] > max_height
    height_out_of_range = low_height | high_height

    # x方向边界：无人机越过底线进入对方半场（x <= -3视为出界）
    x_out_of_range = drone_pos_w[:, 0] <= 0.1

    post_hit_contact = contact & has_hit_ball
    wrong_hit = post_hit_contact

    # Drone/racket-net collision only terminates after a successful hit has entered
    # the hover phase. The env masks out the first hit-transition frame before calling.
    net_collision = drone_net_collision & has_hit_ball

    terminated = height_out_of_range | wrong_hit | net_collision | x_out_of_range
    if episode_length_buf is None or max_episode_length is None:
        truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)
    else:
        truncated = (episode_length_buf >= (max_episode_length - 1)) & (~terminated)

    timeout_without_hit = truncated & (~has_hit_ball)
    wrong_hit = wrong_hit | timeout_without_hit

    if not return_reason_masks:
        return terminated, truncated, torch.zeros(num_envs, device=device)

    reason_masks = {
        "height_out_of_range": height_out_of_range,
        "x_out_of_range": x_out_of_range,
        "wrong_hit": wrong_hit,
        "wrong_hit_post_contact": post_hit_contact,
        "hover_phase_reached": has_hit_ball,
        "timeout": truncated,
        "timeout_without_hit": timeout_without_hit,
        "drone_net_collision": net_collision,
    }

    # reward_end: -10 when terminated and drone crashed (z < 0.3)
    crash = drone_pos_w[:, 2] < 0.3
    extra_termination_rewards = torch.where(
        terminated & crash,
        torch.full((num_envs,), -10.0, dtype=drone_pos_w.dtype, device=device),
        torch.zeros((num_envs,), dtype=drone_pos_w.dtype, device=device),
    )

    return terminated, truncated, reason_masks, extra_termination_rewards


def compute_dones(
    ball_pos_w=None,
    racket_pos_w=None,
    drone_pos_w=None,
    contact=None,
    net_contact=None,
    weak_hit_failure=None,
    drone_up_w=None,
    drone_bottom_z=None,
    episode_length_buf=None,
    max_episode_length=None,
    safe_bounds: dict | None = None,
    z_threshold: float = 0.1,
    return_reason_masks: bool = False,
):
    """Compute success/failure/timeout termination signals."""
    if torch is None or ball_pos_w is None:
        if return_reason_masks:
            return False, False, {}
        return {
            "terminated": False,
            "truncated": False,
            "reason": "running",
        }

    success = contact
    if net_contact is None:
        net_contact = torch.zeros_like(success)
    if weak_hit_failure is None:
        weak_hit_failure = torch.zeros_like(success)
    failure_net = (~success) & (~weak_hit_failure) & net_contact

    ball_z = ball_pos_w[:, 2]
    racket_z = racket_pos_w[:, 2]
    # 球低于球拍阈值（球拍z - 5cm），且球在无人机所在半场（x > 0）时终止
    racket_z_threshold = racket_z - 0.05  # 球拍z - 5cm
    in_drone_half = ball_pos_w[:, 0] > 0.0  # 球在无人机半场
    ball_below_racket = ball_z < racket_z_threshold
    ball_grounded = ball_z <= z_threshold
    failure_server_side_grounded = (~success) & (~weak_hit_failure) & (~failure_net) & (~in_drone_half) & ball_grounded
    failure_ball = (
        (~success)
        & (~weak_hit_failure)
        & (~failure_net)
        & (~failure_server_side_grounded)
        & in_drone_half
        & ball_below_racket
    )

    if safe_bounds is None:
        failure_bounds = torch.zeros_like(failure_ball)
    else:
        x_min, x_max = safe_bounds["x"]
        y_min, y_max = safe_bounds["y"]
        z_min, z_max = safe_bounds["z"]
        # 如果提供了 drone_bottom_z，用它来判断z边界；否则用中心点
        z_to_check = drone_bottom_z if drone_bottom_z is not None else drone_pos_w[:, 2]
        failure_bounds = (
            (drone_pos_w[:, 0] < x_min)
            | (drone_pos_w[:, 0] > x_max)
            | (drone_pos_w[:, 1] < y_min)
            | (drone_pos_w[:, 1] > y_max)
            | (z_to_check < z_min)
            | (z_to_check > z_max)
        )
        failure_bounds = (
            (~success)
            & (~weak_hit_failure)
            & (~failure_net)
            & (~failure_server_side_grounded)
            & (~failure_ball)
            & failure_bounds
        )

    if drone_up_w is None:
        failure_tilt = torch.zeros_like(failure_ball)
    else:
        # z-axis facing down means unstable/flip-like posture
        failure_tilt = (
            (~success)
            & (~weak_hit_failure)
            & (~failure_net)
            & (~failure_server_side_grounded)
            & (~failure_ball)
            & (~failure_bounds)
            & (drone_up_w[:, 2] < 0.0)
        )

    terminated = (
        success
        | weak_hit_failure
        | failure_net
        | failure_server_side_grounded
        | failure_ball
        | failure_bounds
        | failure_tilt
    )

    if episode_length_buf is None or max_episode_length is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = (episode_length_buf >= (max_episode_length - 1)) & (~terminated)

    if not return_reason_masks:
        return terminated, truncated

    reason_masks = {
        "success_contact": success,
        "weak_hit_failure": weak_hit_failure,
        "failure_net_contact": failure_net,
        "failure_server_side_grounded": failure_server_side_grounded,
        "failure_ball_drop": failure_ball,
        "failure_out_of_bounds": failure_bounds,
        "failure_tilt": failure_tilt,
        "timeout": truncated,
    }
    return terminated, truncated, reason_masks
