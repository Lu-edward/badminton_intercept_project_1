from __future__ import annotations

from badminton_intercept.envs.intercept_env_common import ISAACLAB_RUNTIME_AVAILABLE, StepOutput, torch
from badminton_intercept.envs.launch_sampler import build_launch_sampling_spec, sample_launch_candidates
from badminton_intercept.randomization.reset_manager import reset_subenvs


class InterceptEnvResetMixin:
    def _resolve_env_ids(self, env_ids):
        if env_ids is None:
            if self._drone is not None and hasattr(self._drone, "_ALL_INDICES"):
                return self._drone._ALL_INDICES
            if torch is not None:
                return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            return None
        return env_ids

    def _apply_domain_randomization(self, env_ids):
        if not (ISAACLAB_RUNTIME_AVAILABLE and torch is not None):
            return

        self._ensure_runtime_buffers()

        if not bool(getattr(self.cfg, "randomization_enabled", True)):
            return

        num_ids = int(len(env_ids))
        env_ids_cpu = env_ids.detach().to(device="cpu")

        # Sample episode-wise random physics parameters.
        racket_restitution = self._sample_uniform(getattr(self.cfg, "dr_racket_restitution_range", (0.75, 0.90)), num_ids, self.device)
        shuttle_mass = self._sample_uniform(getattr(self.cfg, "dr_ball_mass_range", (0.0045, 0.0055)), num_ids, self.device)
        drone_mass_scale = self._sample_uniform(getattr(self.cfg, "dr_quad_mass_scale_range", (0.95, 1.05)), num_ids, self.device)
        drone_inertia_scale = self._sample_uniform(getattr(self.cfg, "dr_quad_inertia_scale_range", (0.95, 1.05)), num_ids, self.device)
        drone_thrust_scale = self._sample_uniform(getattr(self.cfg, "dr_thrust_coeff_scale_range", (0.95, 1.05)), num_ids, self.device)

        self._racket_restitution[env_ids] = racket_restitution
        self._shuttle_mass_kg[env_ids] = shuttle_mass
        if self._launch_sampler_mode == "online":
            drag_length = self._sample_uniform(getattr(self.cfg, "dr_drag_length_range", (3.8, 4.4)), num_ids, self.device)
            self._drag_length_m[env_ids] = drag_length
        self._drone_mass_scale[env_ids] = drone_mass_scale
        self._drone_inertia_scale[env_ids] = drone_inertia_scale
        self._drone_thrust_scale[env_ids] = drone_thrust_scale

        if self._shuttlecock is not None:
            masses = self._shuttlecock.root_physx_view.get_masses()
            default_mass = self._shuttlecock.data.default_mass
            shuttle_mass_cpu = shuttle_mass.detach().cpu()

            if masses.ndim == 1:
                masses[env_ids_cpu] = shuttle_mass_cpu
                ratio = shuttle_mass_cpu / default_mass[env_ids_cpu].clamp_min(1.0e-6)
            else:
                masses[env_ids_cpu] = shuttle_mass_cpu.unsqueeze(-1)
                ratio = shuttle_mass_cpu.unsqueeze(-1) / default_mass[env_ids_cpu].clamp_min(1.0e-6)

            self._shuttlecock.root_physx_view.set_masses(masses, env_ids_cpu)

            inertias = self._shuttlecock.root_physx_view.get_inertias()
            default_inertia = self._shuttlecock.data.default_inertia
            if inertias.ndim == 2:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratio
            elif inertias.ndim == 3:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * ratio.unsqueeze(-1)
            self._shuttlecock.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if self._drone is not None and hasattr(self._drone, "root_physx_view"):
            masses = self._drone.root_physx_view.get_masses()
            default_mass = self._drone.data.default_mass
            mass_scale_cpu = drone_mass_scale.detach().cpu()

            if masses.ndim == 1:
                masses[env_ids_cpu] = default_mass[env_ids_cpu] * mass_scale_cpu
            else:
                masses[env_ids_cpu] = default_mass[env_ids_cpu] * mass_scale_cpu.unsqueeze(-1)
            self._drone.root_physx_view.set_masses(masses, env_ids_cpu)

            inertias = self._drone.root_physx_view.get_inertias()
            default_inertia = self._drone.data.default_inertia
            inertia_scale_cpu = drone_inertia_scale.detach().cpu()

            if inertias.ndim == 2:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * inertia_scale_cpu.unsqueeze(-1)
            elif inertias.ndim == 3:
                inertias[env_ids_cpu] = default_inertia[env_ids_cpu] * inertia_scale_cpu.unsqueeze(-1).unsqueeze(-1)
            self._drone.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _reset_drone_state(self, env_ids):
        if self._drone is None or not hasattr(self._drone, "data"):
            return

        default_root_state = self._drone.data.default_root_state[env_ids].clone()
        origins = self.scene.env_origins[env_ids]
        num_ids = int(len(env_ids))

        # 根据课程阶段获取无人机初始位置范围
        current_stage = self._curriculum.current_stage
        x_range = current_stage.uav_init_x_range
        y_range = current_stage.uav_init_y_range
        z_range = current_stage.uav_init_z_range
        x = self._sample_uniform(x_range, num_ids, self.device)
        y = self._sample_uniform(y_range, num_ids, self.device)
        z = self._sample_uniform(z_range, num_ids, self.device)
        vel_noise = self._sample_uniform(getattr(self.cfg, "uav_init_lin_vel_range", (-0.1, 0.1)), num_ids * 3, self.device).view(num_ids, 3)

        default_root_state[:, 0] = origins[:, 0] + x
        default_root_state[:, 1] = origins[:, 1] + y
        default_root_state[:, 2] = origins[:, 2] + z
        default_root_state[:, 3] = 1.0
        default_root_state[:, 4:7] = 0.0
        default_root_state[:, 7:10] = vel_noise
        default_root_state[:, 10:13] = 0.0

        self._drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        if hasattr(self._drone.data, "default_joint_pos") and hasattr(self._drone, "write_joint_state_to_sim"):
            joint_pos = self._drone.data.default_joint_pos[env_ids]
            joint_vel = self._drone.data.default_joint_vel[env_ids]
            self._drone.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_shuttlecock_state(self, env_ids):
        if self._shuttlecock is None or not hasattr(self._shuttlecock, "data"):
            return {}

        self._ensure_runtime_buffers()

        origins = self.scene.env_origins[env_ids]
        num_ids = int(len(env_ids))
        current_stage = self._curriculum.current_stage
        launch_spec = build_launch_sampling_spec(self.cfg, current_stage)

        if self._launch_sampler_mode == "trajectory_library":
            self._ensure_launch_libraries_loaded()
            stage_id = int(current_stage.stage_id)
            library = None if self._launch_libraries is None else self._launch_libraries.get(stage_id)
            if library is None:
                raise RuntimeError(f"Trajectory library for stage {stage_id} is not loaded.")

            num_entries = int(library["pos_local"].shape[0])
            sample_idx = torch.randint(num_entries, (num_ids,), device="cpu")
            pos_local = library["pos_local"].index_select(0, sample_idx).to(device=self.device)
            vel_local = library["vel_local"].index_select(0, sample_idx).to(device=self.device)
            drag_length_m = library["drag_length_m"].index_select(0, sample_idx).to(device=self.device)
            hit_plane_z = library["hit_plane_z"].index_select(0, sample_idx).to(device=self.device)
            hit_time_s = library["hit_time_s"].index_select(0, sample_idx).to(device=self.device)
            hit_point_local = library["hit_point_local"].index_select(0, sample_idx).to(device=self.device)
            self._drag_length_m[env_ids] = drag_length_m
            stats = {
                "launch_source": "library",
                "launch_valid_rate": 1.0,
                "launch_resample_rounds_mean": 0.0,
                "launch_fallback_count": 0,
                "launch_no_cross_count": 0,
                "launch_oob_count": 0,
                "launch_no_hit_plane_count": 0,
            }
        else:
            max_rounds = int(getattr(self.cfg, "launch_max_resample_rounds", 2))
            vx_lo, vx_hi = launch_spec.v_forward_range
            vy_lo, vy_hi = launch_spec.v_side_range
            vz_lo, vz_hi = launch_spec.v_up_range
            vx_cross_lo, vx_cross_hi = launch_spec.vx_cross_range
            vx_mid = 0.5 * (vx_lo + vx_hi)
            vy_mid = 0.5 * (vy_lo + vy_hi)
            vz_mid = 0.5 * (vz_lo + vz_hi)

            sampled = sample_launch_candidates(num_ids, launch_spec, self.device, include_drag_length=False)
            pos_local = sampled["pos_local"]
            vel_local = sampled["vel_local"]
            hit_plane_z = sampled["hit_plane_z"]
            drag_length_m = self._drag_length_m[env_ids]

            eval_data = self._evaluate_launch_candidates(
                pos_local=pos_local,
                vel_local=vel_local,
                drag_length_m=drag_length_m,
                launch_spec=launch_spec,
                hit_plane_z=hit_plane_z,
            )
            valid = eval_data["valid"].clone()
            resample_counts = torch.zeros((num_ids,), dtype=torch.int32, device=self.device)

            for _round in range(max_rounds):
                invalid_mask = ~valid
                if not bool(torch.any(invalid_mask)):
                    break

                invalid_idx = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)
                k = int(invalid_idx.numel())
                resample_counts[invalid_idx] += 1

                vx_new = self._sample_uniform(launch_spec.v_forward_range, k, self.device)
                vy_new = self._sample_uniform(launch_spec.v_side_range, k, self.device)
                vz_new = self._sample_uniform(launch_spec.v_up_range, k, self.device)

                local_net_fail = (~eval_data["net_ok"][invalid_idx]) | eval_data["no_cross"][invalid_idx]
                local_x_short = eval_data["landing_x_short"][invalid_idx]
                local_x_long = eval_data["landing_x_long"][invalid_idx]
                local_y_out = eval_data["landing_y_out"][invalid_idx]
                local_no_landing = eval_data["no_landing"][invalid_idx]

                if bool(torch.any(local_net_fail)):
                    c = int(local_net_fail.sum().item())
                    vx_new[local_net_fail] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)
                    vz_new[local_net_fail] = self._sample_uniform((max(vz_mid, vz_lo), vz_hi), c, self.device)
                    vy_new[local_net_fail] = self._sample_uniform((0.45 * vy_lo, 0.45 * vy_hi), c, self.device)

                if bool(torch.any(local_x_short)):
                    c = int(local_x_short.sum().item())
                    vx_new[local_x_short] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)
                if bool(torch.any(local_x_long)):
                    c = int(local_x_long.sum().item())
                    vx_new[local_x_long] = self._sample_uniform((vx_lo, min(vx_mid, vx_hi)), c, self.device)
                if bool(torch.any(local_y_out)):
                    c = int(local_y_out.sum().item())
                    vy_new[local_y_out] = self._sample_uniform((0.35 * vy_lo, 0.35 * vy_hi), c, self.device)
                if bool(torch.any(local_no_landing)):
                    c = int(local_no_landing.sum().item())
                    vz_new[local_no_landing] = self._sample_uniform((max(vz_mid, vz_lo), vz_hi), c, self.device)
                    vx_new[local_no_landing] = self._sample_uniform((vx_cross_lo, vx_cross_hi), c, self.device)

                vel_local[invalid_idx, 0] = vx_new
                vel_local[invalid_idx, 1] = vy_new
                vel_local[invalid_idx, 2] = vz_new

                eval_sub = self._evaluate_launch_candidates(
                    pos_local=pos_local[invalid_idx],
                    vel_local=vel_local[invalid_idx],
                    drag_length_m=drag_length_m[invalid_idx],
                    launch_spec=launch_spec,
                    hit_plane_z=hit_plane_z[invalid_idx],
                )
                for key, value in eval_sub.items():
                    eval_data[key][invalid_idx] = value
                valid = eval_data["valid"]

            invalid_final = ~valid
            fallback_env_count = int(invalid_final.sum().item())
            if fallback_env_count > 0:
                fail_idx = torch.nonzero(invalid_final, as_tuple=False).squeeze(-1)
                m = int(fail_idx.numel())
                unresolved = torch.ones((m,), dtype=torch.bool, device=self.device)
                best_score = torch.full((m,), float("inf"), device=self.device)
                best_vel = vel_local[fail_idx].clone()

                candidate_bank = torch.tensor(
                    [
                        [vx_cross_lo, 0.0, vz_mid],
                        [0.95 * vx_cross_hi, 0.0, vz_mid],
                        [vx_cross_lo, 0.0, 0.95 * vz_hi],
                        [0.95 * vx_cross_hi, 0.0, 0.9 * vz_hi],
                        [0.9 * vx_cross_hi, 0.20 * vy_hi, 0.9 * vz_hi],
                        [0.9 * vx_cross_hi, 0.20 * vy_lo, 0.9 * vz_hi],
                        [vx_cross_lo, 0.12 * vy_hi, max(vz_mid, vz_lo)],
                        [vx_cross_lo, 0.12 * vy_lo, max(vz_mid, vz_lo)],
                        [vx_mid, 0.0, vz_mid],
                        [0.75 * vx_cross_hi, 0.0, max(vz_mid, vz_lo)],
                    ],
                    device=self.device,
                    dtype=vel_local.dtype,
                )
                candidate_bank[:, 0] = candidate_bank[:, 0].clamp(vx_lo, vx_hi)
                candidate_bank[:, 1] = candidate_bank[:, 1].clamp(vy_lo, vy_hi)
                candidate_bank[:, 2] = candidate_bank[:, 2].clamp(vz_lo, vz_hi)

                for candidate in candidate_bank:
                    unresolved_idx_local = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    if unresolved_idx_local.numel() == 0:
                        break
                    env_sel = fail_idx[unresolved_idx_local]
                    cand_vel = candidate.unsqueeze(0).repeat(int(unresolved_idx_local.numel()), 1)

                    eval_cand = self._evaluate_launch_candidates(
                        pos_local=pos_local[env_sel],
                        vel_local=cand_vel,
                        drag_length_m=drag_length_m[env_sel],
                        launch_spec=launch_spec,
                        hit_plane_z=hit_plane_z[env_sel],
                    )
                    score = self._launch_violation_score(
                        eval_data=eval_cand,
                        launch_spec=launch_spec,
                    )
                    better = score < best_score[unresolved_idx_local]
                    if bool(torch.any(better)):
                        best_score[unresolved_idx_local[better]] = score[better]
                        best_vel[unresolved_idx_local[better]] = cand_vel[better]

                    found_valid = eval_cand["valid"]
                    if bool(torch.any(found_valid)):
                        resolved_local = unresolved_idx_local[found_valid]
                        vel_local[fail_idx[resolved_local]] = cand_vel[found_valid]
                        unresolved[resolved_local] = False

                if bool(torch.any(unresolved)):
                    unresolved_idx_local = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    vel_local[fail_idx[unresolved_idx_local]] = best_vel[unresolved_idx_local]

                eval_fallback = self._evaluate_launch_candidates(
                    pos_local=pos_local[fail_idx],
                    vel_local=vel_local[fail_idx],
                    drag_length_m=drag_length_m[fail_idx],
                    launch_spec=launch_spec,
                    hit_plane_z=hit_plane_z[fail_idx],
                )
                for key, value in eval_fallback.items():
                    eval_data[key][fail_idx] = value

            final_valid = eval_data["valid"]
            stats = {
                "launch_source": "online",
                "launch_valid_rate": float(final_valid.float().mean().item()),
                "launch_resample_rounds_mean": float(resample_counts.float().mean().item()),
                "launch_fallback_count": int(fallback_env_count),
                "launch_no_cross_count": int(eval_data["no_cross"].sum().item()),
                "launch_oob_count": int((~eval_data["landing_ok"]).sum().item()),
                "launch_no_hit_plane_count": int(eval_data["no_hit_plane"].sum().item()),
            }
            hit_time_s = eval_data["hit_time_s"]
            hit_point_local = eval_data["hit_point_local"]

        invalid_hit = (~torch.isfinite(hit_time_s)) | (~torch.isfinite(hit_point_local).all(dim=-1))
        if bool(torch.any(invalid_hit)):
            hit_time_s = hit_time_s.clone()
            hit_point_local = hit_point_local.clone()
            hit_time_s[invalid_hit] = 0.0
            hit_point_local[invalid_hit] = pos_local[invalid_hit]

        self._launch_hit_plane_z[env_ids] = hit_plane_z
        self._launch_hit_time_s[env_ids] = hit_time_s
        self._launch_hit_point_local[env_ids] = hit_point_local

        pos_start_world = pos_local + origins

        shuttle_state = self._shuttlecock.data.default_root_state[env_ids].clone()
        shuttle_state[:, 0] = pos_start_world[:, 0]
        shuttle_state[:, 1] = pos_start_world[:, 1]
        shuttle_state[:, 2] = pos_start_world[:, 2]
        shuttle_state[:, 3] = 1.0
        shuttle_state[:, 4:7] = 0.0

        shuttle_state[:, 7] = vel_local[:, 0]
        shuttle_state[:, 8] = vel_local[:, 1]
        shuttle_state[:, 9] = vel_local[:, 2]
        shuttle_state[:, 10:13] = 0.0

        self._shuttlecock.write_root_pose_to_sim(shuttle_state[:, :7], env_ids=env_ids)
        self._shuttlecock.write_root_velocity_to_sim(shuttle_state[:, 7:], env_ids=env_ids)

        self._last_launch_stats = stats
        return stats

    def _update_curriculum_from_resets(self, env_ids):
        if self._last_contact is None:
            return
        if self._curriculum_fixed_stage:
            return
        success_flags = self._last_contact[env_ids].detach().cpu().tolist()
        self._curriculum.record_episode_outcomes([bool(s) for s in success_flags])

    def _reset_idx(self, env_ids):
        if ISAACLAB_RUNTIME_AVAILABLE:
            env_ids = self._resolve_env_ids(env_ids)
            if env_ids is None:
                return
            if bool(getattr(self.cfg, "termination_debug_print", False)) and self._last_done_reasons is not None:
                counts = {}
                for reason, mask in self._last_done_reasons.items():
                    counts[reason] = int(mask[env_ids].sum().item())
                counts_str = ", ".join([f"{k}={v}" for k, v in counts.items() if v > 0]) or "none"
                print(f"[done] reset_envs={int(len(env_ids))}, reasons: {counts_str}")
            self._update_curriculum_from_resets(env_ids)
            super()._reset_idx(env_ids)

            if self._drone is not None and hasattr(self._drone, "reset"):
                self._drone.reset(env_ids)
            if self._shuttlecock is not None and hasattr(self._shuttlecock, "reset"):
                self._shuttlecock.reset(env_ids)
            if self._racket_contact_sensor is not None and hasattr(self._racket_contact_sensor, "reset"):
                self._racket_contact_sensor.reset(env_ids)
            if self._net_contact_sensor is not None and hasattr(self._net_contact_sensor, "reset"):
                self._net_contact_sensor.reset(env_ids)
            if self._drone_net_contact_sensor is not None and hasattr(self._drone_net_contact_sensor, "reset"):
                self._drone_net_contact_sensor.reset(env_ids)
            if self._racket_net_contact_sensor is not None and hasattr(self._racket_net_contact_sensor, "reset"):
                self._racket_net_contact_sensor.reset(env_ids)
            if self._net_drone_contact_sensor is not None and hasattr(self._net_drone_contact_sensor, "reset"):
                self._net_drone_contact_sensor.reset(env_ids)

            # Reset debug counters
            self._debug_drone_net_count = 0
            self._debug_drone_net_error_printed = False

            self._apply_domain_randomization(env_ids)
            self._reset_drone_state(env_ids)
            launch_stats = self._reset_shuttlecock_state(env_ids)
            if hasattr(self, "extras") and isinstance(self.extras, dict):
                self.extras["launch"] = launch_stats
            if self._actions is not None:
                self._actions[env_ids] = 0.0
            if self._prev_actions is not None:
                self._prev_actions[env_ids] = 0.0
            if self._rotor_thrust_cmd_n is not None:
                self._rotor_thrust_cmd_n[env_ids] = 0.0
            if self._filtered_body_rate_rad_s is not None:
                self._filtered_body_rate_rad_s[env_ids] = 0.0
            if self._rate_integral is not None:
                self._rate_integral[env_ids] = 0.0
            if self._rotor_throttle is not None:
                # Seed motors near hover throttle on reset to avoid immediate drop/tilt transients.
                num_rotors = int(self._max_rotor_thrust_n.shape[0]) if self._max_rotor_thrust_n is not None else 4
                if self._max_rotor_thrust_n is not None and self._mass_kg is not None and self._drone_mass_scale is not None:
                    eff_mass = self._mass_kg[env_ids] * self._drone_mass_scale[env_ids] + float(
                        getattr(self.cfg, "fm_extra_mass_kg", 0.147)
                    )
                    hover_total_thrust = 9.81 * eff_mass
                    hover_per_rotor = (hover_total_thrust / float(max(num_rotors, 1))).unsqueeze(-1)
                    max_rotor = self._max_rotor_thrust_n.unsqueeze(0).expand(int(len(env_ids)), -1).clamp_min(1.0e-6)
                    hover_throttle = torch.sqrt((hover_per_rotor / max_rotor).clamp(0.0, 1.0))
                    self._rotor_throttle[env_ids] = hover_throttle
                    if self._rotor_thrust_cmd_n is not None:
                        self._rotor_thrust_cmd_n[env_ids] = hover_throttle.square() * self._max_rotor_thrust_n.unsqueeze(0)
                else:
                    self._rotor_throttle[env_ids] = 0.0
            # 重置扫掠接触检测的参考状态
            if self._contact_reference_valid is not None:
                self._contact_reference_valid[env_ids] = False
            if self._last_sensor_hit is not None:
                self._last_sensor_hit[env_ids] = False
            if self._last_geometric_hit is not None:
                self._last_geometric_hit[env_ids] = False
            if self._serve_hover_termination_rewards is not None:
                self._serve_hover_termination_rewards[env_ids] = 0.0
            if self.post_hit is not None:
                self.post_hit[env_ids] = False
            if self._just_entered_post_hit is not None:
                self._just_entered_post_hit[env_ids] = False
            # 重置 hit 时刻保存的无人机状态
            if self._hit_drone_pos_w is not None:
                self._hit_drone_pos_w[env_ids] = 0.0
            if self._hit_drone_quat_w is not None:
                self._hit_drone_quat_w[env_ids] = 0.0
                self._hit_drone_quat_w[env_ids, 0] = 1.0
            if self._hit_drone_lin_vel_w is not None:
                self._hit_drone_lin_vel_w[env_ids] = 0.0
            if self._hit_drone_ang_vel_w is not None:
                self._hit_drone_ang_vel_w[env_ids] = 0.0
            if self._hit_ball_pos_w is not None:
                self._hit_ball_pos_w[env_ids] = 0.0
            if self._hit_ball_vel_w is not None:
                self._hit_ball_vel_w[env_ids] = 0.0
            if self._hit_time_elapsed is not None:
                self._hit_time_elapsed[env_ids] = 0.0
            if self._task2_server_ground_hold_active is not None:
                self._task2_server_ground_hold_active[env_ids] = False
            if self._task2_server_ground_landing_pos_local is not None:
                self._task2_server_ground_landing_pos_local[env_ids] = 0.0
            if self._task2_server_ground_hold_elapsed_s is not None:
                self._task2_server_ground_hold_elapsed_s[env_ids] = 0.0
            return

        return reset_subenvs(env_ids)

    def reset(self, *args, **kwargs):
        if ISAACLAB_RUNTIME_AVAILABLE:
            return super().reset(*args, **kwargs)
        return self._get_observations()

    def step(self, actions):
        if ISAACLAB_RUNTIME_AVAILABLE:
            return super().step(actions)

        self._pre_physics_step(actions)
        self._apply_action()
        self._apply_external_forces()
        obs = self._get_observations()
        rew = self._get_rewards()
        done = self._get_dones()
        info = {"status": "skeleton_step"}
        return StepOutput(observations=obs, rewards=rew, dones=done, infos=info)
