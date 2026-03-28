from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def compute_dones(
    ball_pos_w=None,
    racket_pos_w=None,
    drone_pos_w=None,
    contact=None,
    net_contact=None,
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
    failure_net = (~success) & net_contact

    ball_z = ball_pos_w[:, 2]
    racket_z = racket_pos_w[:, 2]
    # 球低于球拍阈值（球拍z - 5cm），且球在无人机所在半场（x > 0）时终止
    racket_z_threshold = racket_z - 0.05  # 球拍z - 5cm
    in_drone_half = ball_pos_w[:, 0] > 0.0  # 球在无人机半场
    ball_below_racket = ball_z < racket_z_threshold
    ball_grounded = ball_z <= z_threshold
    failure_server_side_grounded = (~success) & (~failure_net) & (~in_drone_half) & ball_grounded
    failure_ball = (~success) & (~failure_net) & (~failure_server_side_grounded) & in_drone_half & ball_below_racket

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
        failure_bounds = (~success) & (~failure_net) & (~failure_server_side_grounded) & (~failure_ball) & failure_bounds

    if drone_up_w is None:
        failure_tilt = torch.zeros_like(failure_ball)
    else:
        # z-axis facing down means unstable/flip-like posture
        failure_tilt = (
            (~success)
            & (~failure_net)
            & (~failure_server_side_grounded)
            & (~failure_ball)
            & (~failure_bounds)
            & (drone_up_w[:, 2] < 0.0)
        )

    terminated = success | failure_net | failure_server_side_grounded | failure_ball | failure_bounds | failure_tilt

    if episode_length_buf is None or max_episode_length is None:
        truncated = torch.zeros_like(terminated)
    else:
        truncated = (episode_length_buf >= (max_episode_length - 1)) & (~terminated)

    if not return_reason_masks:
        return terminated, truncated

    reason_masks = {
        "success_contact": success,
        "failure_net_contact": failure_net,
        "failure_server_side_grounded": failure_server_side_grounded,
        "failure_ball_drop": failure_ball,
        "failure_out_of_bounds": failure_bounds,
        "failure_tilt": failure_tilt,
        "timeout": truncated,
    }
    return terminated, truncated, reason_masks
