from __future__ import annotations

from badminton_intercept.envs.intercept_env_common import ISAACLAB_RUNTIME_AVAILABLE, torch
from badminton_intercept.mdp.observations import (
    build_actor_critic_observations,
    build_serve_hover_observations,
)
from badminton_intercept.mdp.rewards import compute_rewards, compute_rewards_serve_hover
from badminton_intercept.mdp.terminations import compute_dones, compute_dones_serve_hover


class InterceptEnvMDPMixin:
    def _get_observations(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            self._ensure_runtime_buffers()
            # Override ball with analytical trajectory for post-hit environments
            self._apply_post_hit_analytical_trajectory()
            # Enforce a hard post-step speed cap so logged/observed states remain bounded.
            self._enforce_linear_speed_limit()
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            
            drone_pos_local = drone_pos_w
            ball_pos_local = ball_pos_w
            if hasattr(self.scene, "env_origins"):
                drone_pos_local = drone_pos_w - self.scene.env_origins
                ball_pos_local = ball_pos_w - self.scene.env_origins
            task2_ball_pos_local, task2_ball_lin_vel_w = self._get_task2_ball_state_for_policy(
                ball_pos_local,
                ball_lin_vel_w,
            )
            hit_point_w = self._launch_hit_point_local + self.scene.env_origins
            hit_point_rel_w = hit_point_w - drone_pos_w

            time_norm = (self.episode_length_buf.float() / float(self.max_episode_length)).clamp(0.0, 1.0)
            privileged = {
                "time_norm": time_norm,
                "drag_length": self._drag_length_m,
                "ball_mass": self._shuttle_mass_kg,
            }
            if self.enable_serve_hover:
                post_hit_mask = (
                    self.post_hit.unsqueeze(-1)
                    if self.post_hit is not None
                    else torch.zeros((self.num_envs, 1), device=drone_pos_local.device, dtype=torch.bool)
                )
                serve_hover_mask = self._get_serve_hover_policy_mask(device=drone_pos_local.device)
                if serve_hover_mask.dim() == 1:
                    serve_hover_mask = serve_hover_mask.unsqueeze(-1)
                obs = build_serve_hover_observations(
                    drone_pos=drone_pos_local,
                    drone_quat_w=drone_quat_w,
                    drone_lin_vel=drone_lin_vel_w,
                    drone_ang_vel=drone_ang_vel_w,
                    ball_pos=ball_pos_local,
                    ball_lin_vel=ball_lin_vel_w,
                    hit_time_s=self._launch_hit_time_s,
                    hit_point_rel=hit_point_rel_w,
                    post_hit_mask=post_hit_mask,
                    serve_hover_mask=serve_hover_mask,
                    prev_action=self._prev_actions,
                    privileged=privileged,
                    add_actor_noise=True,
                    noise_std=0.01,
                )
            else:
                obs = build_actor_critic_observations(
                    drone_pos=drone_pos_local,
                    drone_quat_w=drone_quat_w,
                    drone_lin_vel=drone_lin_vel_w,
                    drone_ang_vel=drone_ang_vel_w,
                    ball_pos=task2_ball_pos_local,
                    ball_lin_vel=task2_ball_lin_vel_w,
                    hit_time_s=self._launch_hit_time_s,
                    hit_point_rel=hit_point_rel_w,
                    privileged=privileged,
                    add_actor_noise=True,
                    noise_std=0.01,
                )
            if isinstance(obs, dict):
                for k, v in obs.items():
                    obs[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0e3, neginf=-1.0e3).clamp(-1.0e3, 1.0e3)
            return obs
        if self.enable_serve_hover:
            return build_serve_hover_observations()
        return build_actor_critic_observations()

    def _get_rewards(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            # Override ball with analytical trajectory for post-hit environments
            self._apply_post_hit_analytical_trajectory()
            drone_pos_w, drone_quat_w, drone_lin_vel_w, drone_ang_vel_w = self._get_drone_kinematics()
            racket_pos_w = self._get_racket_center_pos_w(drone_pos_w)
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            racket_normal_w = self._get_racket_normal_w(drone_quat_w)
            contact, d_axis = self._compute_contact_signal(
                racket_pos_w,
                ball_pos_w,
                ball_lin_vel_w=ball_lin_vel_w,
                racket_normal_w=racket_normal_w,
            )
            yaw = self._quat_to_yaw(drone_quat_w)
            drone_up_w = self._quat_to_up_axis_w(drone_quat_w)
            bound_dist = self._compute_bound_distance(drone_pos_w)

            # Convert to local coordinates (relative to each env's origin), consistent with _get_dones
            drone_pos_local = drone_pos_w
            racket_pos_local = racket_pos_w
            ball_pos_local = ball_pos_w
            if hasattr(self.scene, "env_origins"):
                drone_pos_local = drone_pos_w - self.scene.env_origins
                racket_pos_local = racket_pos_w - self.scene.env_origins
                ball_pos_local = ball_pos_w - self.scene.env_origins
            task2_ball_pos_local, task2_ball_lin_vel_w = self._get_task2_ball_state_for_policy(
                ball_pos_local,
                ball_lin_vel_w,
            )
            if self.enable_serve_hover:
                from badminton_intercept.mdp.rewards import compute_rewards_serve_hover
                rewards = compute_rewards_serve_hover(
                    drone_pos=drone_pos_local,
                    drone_up_w=drone_up_w,
                    drone_ang_vel_w=drone_ang_vel_w,
                    ball_pos_w=task2_ball_pos_local,
                    has_hit_ball=self.post_hit,
                    action=self._actions,
                    prev_action=self._prev_actions,
                    yaw=yaw,
                    bound_dist=bound_dist,
                )
                total_reward = rewards["total"]
                if self._serve_hover_termination_rewards is not None:
                    total_reward = total_reward + self._serve_hover_termination_rewards
                if self._weak_hit_termination_rewards is not None:
                    total_reward = total_reward + self._weak_hit_termination_rewards
                return total_reward

            # 任务二模式：使用 compute_rewards_task2
            if self.enable_post_hit_tracking:
                from badminton_intercept.mdp.rewards import compute_rewards_task2, compute_racket_normal_x_component
                normal_x = compute_racket_normal_x_component(drone_quat_w)

                rewards = compute_rewards_task2(
                    racket_pos_w=racket_pos_local,
                    ball_pos_w=task2_ball_pos_local,
                    ball_vel_w=task2_ball_lin_vel_w,
                    contact=contact,
                    action=self._actions,
                    prev_action=self._prev_actions,
                    yaw=yaw,
                    d_axis=d_axis,
                    bound_dist=bound_dist,
                    drone_up_w=drone_up_w,
                    drone_ang_vel_w=drone_ang_vel_w,
                    drone_lin_vel_w=drone_lin_vel_w,
                    drone_pos_w=drone_pos_local,
                    drone_quat_w=drone_quat_w,
                    episode_length_buf=self.episode_length_buf,
                    max_episode_length=self.max_episode_length,
                    has_hit_ball=self.post_hit,
                    c_ang_vel=float(getattr(self.cfg, "reward_c_ang_vel", 0.05)),
                )
                total_reward = rewards["total"]
                if self._weak_hit_termination_rewards is not None:
                    total_reward = total_reward + self._weak_hit_termination_rewards
                return total_reward

            # 任务一模式：使用原有奖励函数
            rewards = compute_rewards(
                racket_pos_w=racket_pos_local,
                ball_pos_w=ball_pos_local,
                contact=contact,
                action=self._actions,
                prev_action=self._prev_actions,
                yaw=yaw,
                d_axis=d_axis,
                bound_dist=bound_dist,
                drone_up_w=drone_up_w,
                drone_ang_vel_w=drone_ang_vel_w,
                drone_lin_vel_w=drone_lin_vel_w,
                drone_pos_w=drone_pos_local,
                episode_length_buf=self.episode_length_buf,
                max_episode_length=self.max_episode_length,
                c_ang_vel=float(getattr(self.cfg, "reward_c_ang_vel", 0.05)),
            )
            return rewards["total"]
        return compute_rewards()

    def _get_dones(self):
        if ISAACLAB_RUNTIME_AVAILABLE and torch is not None:
            # Override ball with analytical trajectory for post-hit environments
            self._apply_post_hit_analytical_trajectory()
            drone_pos_w, drone_quat_w, _, _ = self._get_drone_kinematics()
            racket_pos_w = self._get_racket_center_pos_w(drone_pos_w)
            ball_pos_w = self._shuttlecock.data.root_pos_w
            ball_lin_vel_w = self._shuttlecock.data.root_lin_vel_w
            racket_normal_w = self._get_racket_normal_w(drone_quat_w)
            contact, _ = self._compute_contact_signal(
                racket_pos_w,
                ball_pos_w,
                ball_lin_vel_w=ball_lin_vel_w,
                racket_normal_w=racket_normal_w,
            )
            net_contact = self._compute_net_contact_signal()
            drone_net_contact = self._compute_drone_net_contact_signal()
            self._last_contact = contact.clone()
            weak_hit_failure = (
                self._last_weak_hit_failure.clone()
                if self._last_weak_hit_failure is not None
                else torch.zeros_like(contact)
            )
            pending_weak_hit = (
                self._pending_weak_hit.clone()
                if self._pending_weak_hit is not None
                else torch.zeros_like(contact)
            )
            weak_hit_reward = float(getattr(self.cfg, "weak_hit_reward", 5))
            self._weak_hit_termination_rewards = torch.where(
                weak_hit_failure,
                torch.full((self.num_envs,), weak_hit_reward, dtype=ball_pos_w.dtype, device=self.device),
                torch.zeros((self.num_envs,), dtype=ball_pos_w.dtype, device=self.device),
            )
            self._serve_hover_termination_rewards = torch.zeros((self.num_envs,), dtype=ball_pos_w.dtype, device=self.device)
            drone_up_w = self._quat_to_up_axis_w(drone_quat_w)
            drone_pos_local = drone_pos_w
            racket_pos_local = racket_pos_w
            ball_pos_local = ball_pos_w
            if hasattr(self.scene, "env_origins"):
                drone_pos_local = drone_pos_w - self.scene.env_origins
                racket_pos_local = racket_pos_w - self.scene.env_origins
                ball_pos_local = ball_pos_w - self.scene.env_origins
            task2_ball_pos_local, _ = self._get_task2_ball_state_for_policy(ball_pos_local, None)
            # 使用无人机的最下面点来判断（中心高度 - 0.13m）
            drone_bottom_z = drone_pos_local[:, 2] - 0.13
            safe_bounds = {"x": (0.0, 6.7), "y": (-3.05, 3.05), "z": (0.2, 3.0)}  # 最下面不低于20cm

            if self.enable_serve_hover:
                from badminton_intercept.mdp.rewards import compute_racket_normal_x_component

                normal_x = compute_racket_normal_x_component(drone_quat_w)
                correct_posture = normal_x < 0.0
                just_entered_post_hit = self._just_entered_post_hit if self._just_entered_post_hit is not None else torch.zeros_like(contact)
                post_hit_active = self.post_hit & (~just_entered_post_hit)
                pre_hit_envs = ~post_hit_active
                terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                truncated = torch.zeros_like(terminated)
                reason_masks = {}

                # Debug: log drone_net_contact and has_hit_ball status
                drone_net_contact_for_hover = drone_net_contact & post_hit_active
                if bool(getattr(self.cfg, "drone_net_contact_debug_print", False)):
                    if int(drone_net_contact_for_hover.sum()) > 0:
                        print(
                            "[DEBUG] drone_net_contact detected: "
                            f"{int(drone_net_contact_for_hover.sum())}, post_hit_active={int(post_hit_active.sum())}"
                        )

                if post_hit_active.any():
                    terminated_post, truncated_post, reason_masks_post, extra_term_rewards_post = compute_dones_serve_hover(
                        drone_pos_w=drone_pos_local[post_hit_active],
                        drone_bottom_z=drone_bottom_z[post_hit_active],
                        contact=contact[post_hit_active],
                        has_hit_ball=torch.ones(
                            int(post_hit_active.sum().item()),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                        correct_posture=correct_posture[post_hit_active],
                        drone_net_collision=drone_net_contact[post_hit_active],
                        episode_length_buf=self.episode_length_buf[post_hit_active],
                        max_episode_length=self.max_episode_length,
                        min_height=float(getattr(self.cfg, "serve_hover_min_height", 0.1)),
                        max_height=float(getattr(self.cfg, "serve_hover_max_height", 4.0)),
                        return_reason_masks=True,
                    )
                    terminated[post_hit_active] = terminated_post
                    truncated[post_hit_active] = truncated_post
                    self._serve_hover_termination_rewards[post_hit_active] = extra_term_rewards_post
                    for key, mask in reason_masks_post.items():
                        full_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        full_mask[post_hit_active] = mask
                        reason_masks[key] = full_mask

                if pre_hit_envs.any():
                    contact_for_prehit = contact[pre_hit_envs] & (~just_entered_post_hit[pre_hit_envs])
                    terminated_pre, truncated_pre, reason_masks_pre = compute_dones(
                        ball_pos_w=ball_pos_local[pre_hit_envs],
                        racket_pos_w=racket_pos_local[pre_hit_envs],
                        drone_pos_w=drone_pos_local[pre_hit_envs],
                        drone_bottom_z=drone_bottom_z[pre_hit_envs],
                        contact=contact_for_prehit,
                        net_contact=net_contact[pre_hit_envs],
                        weak_hit_failure=weak_hit_failure[pre_hit_envs],
                        drone_up_w=drone_up_w[pre_hit_envs],
                        episode_length_buf=self.episode_length_buf[pre_hit_envs],
                        max_episode_length=self.max_episode_length,
                        safe_bounds=safe_bounds,
                        z_threshold=0.1,
                        return_reason_masks=True,
                    )
                    truncated_pre = truncated_pre & (~just_entered_post_hit[pre_hit_envs])
                    pending_pre = pending_weak_hit[pre_hit_envs] & (~weak_hit_failure[pre_hit_envs])
                    terminated_pre = terminated_pre & (~pending_pre)
                    truncated_pre = truncated_pre & (~pending_pre)
                    for key, mask in reason_masks_pre.items():
                        reason_masks_pre[key] = mask & (~pending_pre)
                    terminated[pre_hit_envs] = terminated_pre
                    truncated[pre_hit_envs] = truncated_pre
                    for key, mask in reason_masks_pre.items():
                        if key not in reason_masks:
                            reason_masks[key] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        reason_masks[key][pre_hit_envs] = mask
                self._last_done_reasons = reason_masks
            # 任务二模式：post-hit 阶段使用新的 termination 逻辑
            elif self.enable_post_hit_tracking and self.post_hit is not None and self.post_hit.any():
                from badminton_intercept.mdp.terminations import compute_dones_task2

                post_hit_envs = self.post_hit
                terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                truncated = torch.zeros_like(terminated)
                reason_masks = {}

                # post-hit 环境：使用 compute_dones_task2
                if post_hit_envs.any():
                    task2_court_bounds = {"x": (-6.7, 6.7), "y": (-3.05, 3.05)}
                    terminated_task2, rewards_task2, truncated_task2, reason_masks_task2 = compute_dones_task2(
                        ball_pos_w=task2_ball_pos_local[post_hit_envs],
                        net_contact=net_contact[post_hit_envs],
                        drone_net_collision=drone_net_contact[post_hit_envs],
                        z_threshold=0.1,
                        max_ball_height=7.0,
                        court_bounds=task2_court_bounds,
                        return_reason_masks=True,
                    )
                    hold_active = (
                        self._task2_server_ground_hold_active[post_hit_envs]
                        if self._task2_server_ground_hold_active is not None
                        else torch.zeros_like(terminated_task2)
                    )
                    hold_expired = (
                        hold_active
                        & (
                            self._task2_server_ground_hold_elapsed_s[post_hit_envs]
                            >= self._get_task2_server_ground_hold_duration_s()
                        )
                    )
                    other_termination = (
                        reason_masks_task2["net_contact"]
                        | reason_masks_task2["drone_half_grounded"]
                        | reason_masks_task2["drone_half_out"]
                        | reason_masks_task2["server_half_out"]
                        | reason_masks_task2["ball_too_high"]
                        | reason_masks_task2["drone_net_collision"]
                    )
                    terminated_task2 = other_termination | hold_expired
                    truncated_task2 = truncated_task2 & (~hold_active)
                    reason_masks_task2["server_half_ground_hold_active"] = hold_active
                    reason_masks_task2["server_half_ground_hold_complete"] = hold_expired
                    terminated[post_hit_envs] = terminated_task2
                    truncated[post_hit_envs] = truncated_task2

                    # 转换 reason_masks 到完整形状
                    for key, mask in reason_masks_task2.items():
                        full_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        full_mask[post_hit_envs] = mask
                        reason_masks[key] = full_mask

                # 非 post-hit 环境：使用原有 compute_dones 逻辑（但遮蔽 contact 终止）
                # 原因：这些 envs 正在被击球，contact 会立即终止它们，导致 post_hit 来不及设置
                non_post_hit = ~post_hit_envs
                if non_post_hit.any():
                    contact_masked = contact[non_post_hit].clone()
                    # 遮蔽掉那些刚被击中（且姿态正确）的 envs，让它们转入 post-hit 而非立即终止
                    from badminton_intercept.mdp.rewards import compute_racket_normal_x_component
                    normal_x = compute_racket_normal_x_component(drone_quat_w[non_post_hit])
                    correct_posture = normal_x < 0.0
                    transitioning_to_post_hit = contact[non_post_hit] & correct_posture
                    contact_for_dones = contact_masked & ~transitioning_to_post_hit

                    terminated_non, truncated_non, reason_masks_non = compute_dones(
                        ball_pos_w=ball_pos_local[non_post_hit],
                        racket_pos_w=racket_pos_local[non_post_hit],
                        drone_pos_w=drone_pos_local[non_post_hit],
                        drone_bottom_z=drone_bottom_z[non_post_hit],
                        contact=contact_for_dones,
                        net_contact=net_contact[non_post_hit],
                        weak_hit_failure=weak_hit_failure[non_post_hit],
                        drone_up_w=drone_up_w[non_post_hit],
                        episode_length_buf=self.episode_length_buf[non_post_hit],
                        max_episode_length=self.max_episode_length,
                        safe_bounds=safe_bounds,
                        z_threshold=0.1,
                        return_reason_masks=True,
                    )
                    # 遮蔽 transition 时刻的 timeout，避免 episode 时间到了就打断转入 post-hit
                    truncated_non = truncated_non & ~transitioning_to_post_hit
                    pending_non = pending_weak_hit[non_post_hit] & (~weak_hit_failure[non_post_hit])
                    terminated_non = terminated_non & (~pending_non)
                    truncated_non = truncated_non & (~pending_non)
                    for key, mask in reason_masks_non.items():
                        reason_masks_non[key] = mask & (~pending_non)
                    terminated[non_post_hit] = terminated_non
                    truncated[non_post_hit] = truncated_non

                    # 合并 reason_masks
                    for key, mask in reason_masks_non.items():
                        if key not in reason_masks:
                            reason_masks[key] = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                        reason_masks[key][non_post_hit] = mask

                self._last_done_reasons = reason_masks
            else:
                # 任务一模式或任务二 pre-hit 阶段：使用原有逻辑
                terminated, truncated, reason_masks = compute_dones(
                    ball_pos_w=ball_pos_local,
                    racket_pos_w=racket_pos_local,
                    drone_pos_w=drone_pos_local,
                    drone_bottom_z=drone_bottom_z,  # 传递最下面高度用于z边界判定
                    contact=contact,
                    net_contact=net_contact,
                    weak_hit_failure=weak_hit_failure,
                    drone_up_w=drone_up_w,
                    episode_length_buf=self.episode_length_buf,
                    max_episode_length=self.max_episode_length,
                    safe_bounds=safe_bounds,
                    z_threshold=0.1,
                    return_reason_masks=True,
                )
                pending_running = pending_weak_hit & (~weak_hit_failure)
                terminated = terminated & (~pending_running)
                truncated = truncated & (~pending_running)
                for key, mask in reason_masks.items():
                    reason_masks[key] = mask & (~pending_running)
                self._last_done_reasons = reason_masks
            
            if hasattr(self, "extras"):
                # 只有当有 env 真正结束时才上报 episode 统计
                num_resets = int((terminated | truncated).sum().item())
                if num_resets > 0:
                    done_mask = terminated | truncated

                    if self.enable_serve_hover:
                        # ============================================================
                        # Serve Hover 日志（与 task2 日志完全独立）
                        # reason_masks keys from compute_dones_serve_hover:
                        #   height_out_of_range, x_out_of_range, wrong_hit, wrong_hit_post_contact,
                        #   hover_phase_reached, timeout, timeout_without_hit,
                        #   drone_net_collision
                        # ============================================================
                        summary = {}
                        zero_mask = torch.zeros_like(done_mask)

                        # Post-hit 阶段结束且有 hover 记录的 env 数量
                        hover_reached_mask = reason_masks.get("hover_phase_reached", zero_mask)
                        post_hit_done_mask = done_mask & hover_reached_mask
                        post_hit_done_count = int(post_hit_done_mask.sum().item())
                        hover_reached_count = int((done_mask & hover_reached_mask).sum().item())

                        # Post-hit 阶段内的终止原因
                        height_out = reason_masks.get("height_out_of_range", zero_mask)
                        wrong_hit_mask = reason_masks.get("wrong_hit", zero_mask)
                        timeout_mask = reason_masks.get("timeout", zero_mask)
                        drone_net_collision_mask = reason_masks.get("drone_net_collision", zero_mask)
                        weak_hit_failure_mask = reason_masks.get("weak_hit_failure", zero_mask)
                        x_boundary_mask = reason_masks.get("x_out_of_range", zero_mask)

                        height_out_count = int((post_hit_done_mask & height_out).sum().item())
                        wrong_hit_count = int((post_hit_done_mask & wrong_hit_mask).sum().item())
                        timeout_count = int((post_hit_done_mask & timeout_mask).sum().item())
                        drone_net_collision_count = int((post_hit_done_mask & drone_net_collision_mask).sum().item())
                        weak_hit_failure_count = int((done_mask & weak_hit_failure_mask).sum().item())
                        x_boundary_count = int((post_hit_done_mask & x_boundary_mask).sum().item())

                        wrong_post_count = int((post_hit_done_mask & reason_masks.get("wrong_hit_post_contact", zero_mask)).sum().item())

                        # 分母为 post-hit 阶段结束的 env 数量
                        if post_hit_done_count > 0:
                            denom = float(post_hit_done_count)
                        else:
                            denom = 1.0  # 避免除零

                        summary["serve_hover_height_out_rate"] = float(height_out_count / denom)
                        summary["serve_hover_wrong_hit_rate"] = float(wrong_hit_count / denom)
                        summary["serve_hover_timeout_rate"] = float(timeout_count / denom)
                        summary["serve_hover_drone_net_collision_rate"] = float(drone_net_collision_count / denom)
                        summary["serve_hover_x_boundary_rate"] = float(x_boundary_count / denom)
                        summary["pre_hit_weak_hit_failure_rate"] = float(weak_hit_failure_count / float(max(num_resets, 1)))

                        # 错误击球细分
                        summary["serve_hover_wrong_hit_post_rate"] = float(wrong_post_count / denom)

                        # pre_hit_success_hit_rate（分母为所有结束 episodes）
                        summary["pre_hit_success_hit_rate"] = float(hover_reached_count / num_resets)

                        current_stage = self._curriculum.current_stage
                        summary["curriculum_stage_id"] = current_stage.stage_id
                        self.extras["episode"] = summary
                    else:
                        # ============================================================
                        # Task2 / Task1 日志（原有逻辑，不受影响）
                        # ============================================================
                        summary = {}

                        zero_mask = torch.zeros_like(done_mask)
                        if self.post_hit is not None:
                            post_hit_done_mask = done_mask & self.post_hit
                        else:
                            post_hit_done_mask = zero_mask
                        pre_hit_done_mask = done_mask & (~post_hit_done_mask)
                        post_hit_done_count = int(post_hit_done_mask.sum().item())
                        pre_hit_done_count = int(pre_hit_done_mask.sum().item())
                        # Build exclusive post-hit categories with precedence + unknown fallback.
                        if post_hit_done_count > 0:
                            post_hit_remaining = post_hit_done_mask.clone()
                            post_hit_category_masks = (
                                ("post_hit_net_contact_rate", reason_masks.get("net_contact", zero_mask)),
                                ("post_hit_drone_net_collision_rate", reason_masks.get("drone_net_collision", zero_mask)),
                                ("post_hit_ball_too_high_rate", reason_masks.get("ball_too_high", zero_mask)),
                                ("post_hit_drone_half_grounded_rate", reason_masks.get("drone_half_grounded", zero_mask)),
                                ("post_hit_server_half_grounded_rate", reason_masks.get("server_half_grounded", zero_mask)),
                                ("post_hit_drone_half_out_rate", reason_masks.get("drone_half_out", zero_mask)),
                                ("post_hit_server_half_out_rate", reason_masks.get("server_half_out", zero_mask)),
                            )
                            for metric_name, reason_mask in post_hit_category_masks:
                                category_mask = post_hit_remaining & reason_mask
                                category_count = int(category_mask.sum().item())
                                summary[metric_name] = float(category_count / post_hit_done_count)
                                post_hit_remaining = post_hit_remaining & (~reason_mask)

                        # Pre-hit success means the episode made it into post-hit.
                        # Therefore its count should match the total number of post-hit-completed episodes.
                        summary["pre_hit_success_hit_rate"] = float(post_hit_done_count / num_resets)

                        # Build exclusive pre-hit failure categories with precedence.
                        # Normalize them by all completed episodes so the pre-hit success/failure
                        # rates live on the same denominator and sum to ~1 together.
                        pre_hit_remaining = pre_hit_done_mask.clone()
                        pre_hit_category_masks = (
                            ("pre_hit_weak_hit_failure_rate", reason_masks.get("weak_hit_failure", zero_mask)),
                            ("pre_hit_failure_net_contact_rate", reason_masks.get("failure_net_contact", zero_mask)),
                            ("pre_hit_failure_server_side_grounded_rate", reason_masks.get("failure_server_side_grounded", zero_mask)),
                            ("pre_hit_failure_ball_drop_rate", reason_masks.get("failure_ball_drop", zero_mask)),
                            ("pre_hit_failure_out_of_bounds_rate", reason_masks.get("failure_out_of_bounds", zero_mask)),
                            ("pre_hit_failure_tilt_rate", reason_masks.get("failure_tilt", zero_mask)),
                            ("pre_hit_timeout_rate", reason_masks.get("timeout", zero_mask)),
                        )
                        for metric_name, reason_mask in pre_hit_category_masks:
                            category_mask = pre_hit_remaining & reason_mask
                            category_count = int(category_mask.sum().item())
                            summary[metric_name] = float(category_count / num_resets)
                            pre_hit_remaining = pre_hit_remaining & (~reason_mask)
                        # Backward-compatible key used by some log/curriculum pipelines.
                        summary["success_rate"] = summary["pre_hit_success_hit_rate"]
                        # 添加课程学习阶段信息
                        current_stage = self._curriculum.current_stage
                        summary["curriculum_stage_id"] = current_stage.stage_id
                        # RSL-RL logger 只读取 extras["episode"] 或 extras["log"]
                        self.extras["episode"] = summary
                else:
                    # 没有 env 结束时，清除 episode 统计避免被错误平均
                    if "episode" in self.extras:
                        del self.extras["episode"]
            return terminated, truncated
        return compute_dones()

