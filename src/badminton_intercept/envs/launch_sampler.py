from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:
    torch = None

from badminton_intercept.mdp.curriculum import CurriculumManager, CurriculumStage
from badminton_intercept.physics.shuttle_aero import (
    evaluate_shuttle_trajectory_constraints,
    predict_shuttle_trajectory,
)

LAUNCH_LIBRARY_SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRARY_STAGE_FILES = ("serve_stage_1.pt", "serve_stage_2.pt", "serve_stage_3.pt")


@dataclass(frozen=True)
class LaunchSamplingSpec:
    stage_id: int
    stage_name: str
    net_height: float
    court_length: float
    court_width: float
    dt_pred: float
    max_time_pred: float
    net_margin: float
    x_margin: float
    y_margin: float
    init_x_range: tuple[float, float]
    init_y_range: tuple[float, float]
    init_z_range: tuple[float, float]
    v_forward_range: tuple[float, float]
    v_side_range: tuple[float, float]
    v_up_range: tuple[float, float]
    vx_cross_range: tuple[float, float]
    drag_length_range: tuple[float, float]
    hit_plane_z_range: tuple[float, float]
    use_near_net_init_x: bool
    center_ball_init_z_on_net: bool
    z_half_span_m: float
    landing_x_range: tuple[float, float] | None = None
    landing_y_range: tuple[float, float] | None = None


def get_curriculum_stages() -> list[CurriculumStage]:
    return list(CurriculumManager().stages)


def _build_cumulative_stage_velocity_ranges(stage: CurriculumStage) -> tuple[tuple[float, float], tuple[float, float]]:
    stages = get_curriculum_stages()
    selected = [candidate for candidate in stages if int(candidate.stage_id) <= int(stage.stage_id)]
    if len(selected) == 0:
        selected = [stage]

    vx_lo = min(float(candidate.ball_vel_x_range[0]) for candidate in selected)
    vx_hi = max(float(candidate.ball_vel_x_range[1]) for candidate in selected)
    vz_lo = min(float(candidate.ball_vel_z_range[0]) for candidate in selected)
    vz_hi = max(float(candidate.ball_vel_z_range[1]) for candidate in selected)
    return (vx_lo, vx_hi), (vz_lo, vz_hi)


def get_launch_sampler_mode(cfg: Any) -> str:
    mode = str(getattr(cfg, "launch_sampler_mode", "online")).strip().lower() or "online"
    if mode not in {"online", "trajectory_library"}:
        raise ValueError(f"Unsupported launch_sampler_mode: {mode}")
    return mode


def resolve_launch_library_dir(cfg: Any) -> Path:
    dir_value = str(getattr(cfg, "launch_library_dir", "assets/trajectory_library")).strip() or "assets/trajectory_library"
    path = Path(dir_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_launch_library_stage_paths(cfg: Any) -> tuple[Path, Path, Path]:
    stage_files = tuple(str(v) for v in getattr(cfg, "launch_library_stage_files", DEFAULT_LIBRARY_STAGE_FILES))
    if len(stage_files) != 3:
        raise ValueError("launch_library_stage_files must contain exactly three filenames.")
    library_dir = resolve_launch_library_dir(cfg)
    return tuple(library_dir / stage_file for stage_file in stage_files)


def build_launch_sampling_spec(cfg: Any, stage: CurriculumStage) -> LaunchSamplingSpec:
    arena_cfg = getattr(cfg, "arena", None)
    net_height = float(getattr(arena_cfg, "net_height", 1.55))
    court_length = float(getattr(arena_cfg, "court_length", 13.4))
    court_width = float(getattr(arena_cfg, "court_width", 6.1))

    dt_pred = float(getattr(cfg, "launch_prediction_dt", 0.01))
    max_time_pred = float(getattr(cfg, "launch_prediction_horizon_s", 3.0))
    net_margin = float(getattr(cfg, "launch_net_clearance_margin_m", 0.15))
    x_margin = float(getattr(cfg, "launch_landing_margin_x_m", 0.1))
    y_margin = float(getattr(cfg, "launch_landing_margin_y_m", 0.1))

    use_near_net_init_x = bool(getattr(cfg, "launch_use_near_net_init_x", True))
    stage_ball_init_x_range = tuple(float(v) for v in getattr(stage, "ball_init_x_range", (-6.0, -2.0)))
    
    if use_near_net_init_x and stage_ball_init_x_range == (-6.0, -2.0):
        init_x_range = tuple(float(v) for v in getattr(cfg, "launch_ball_init_x_range", (-4.5, -1.8)))
    else:
        init_x_range = stage_ball_init_x_range
    
    init_y_range = tuple(float(v) for v in getattr(stage, "ball_init_y_range", (-1, 1)))

    center_ball_init_z_on_net = bool(getattr(cfg, "launch_center_ball_init_z_on_net", True))
    z_half_span_m = float(getattr(cfg, "launch_ball_init_z_half_span_m", 0.3))
    if center_ball_init_z_on_net:
        init_z_range = (max(0.2, net_height - z_half_span_m), net_height + z_half_span_m)
    else:
        init_z_range = tuple(float(v) for v in getattr(stage, "ball_init_z_range", (1.35, 1.75)))

    cumulative_v_forward_range, cumulative_v_up_range = _build_cumulative_stage_velocity_ranges(stage)
    v_forward_range = tuple(float(v) for v in cumulative_v_forward_range)
    v_side_range = tuple(float(v) for v in getattr(stage, "ball_vel_y_range", (-1.5, 1.5)))
    v_up_range = tuple(float(v) for v in cumulative_v_up_range)

    boosted_vx_lo = float(getattr(cfg, "launch_vx_min_for_cross", 6.0))
    boosted_vx_hi = float(getattr(cfg, "launch_vx_max_for_cross", 12.0))
    vx_lo, vx_hi = v_forward_range
    vx_cross_lo = max(vx_lo, min(boosted_vx_lo, vx_hi))
    vx_cross_hi = max(vx_hi, boosted_vx_hi)
    drag_length_range = tuple(float(v) for v in getattr(cfg, "dr_drag_length_range", (3.8, 4.4)))
    hit_plane_z_range = tuple(float(v) for v in getattr(cfg, "launch_hit_plane_z_range", (1.0, 1.5)))

    landing_x_range = getattr(stage, "landing_x_range", None)
    landing_y_range = getattr(stage, "landing_y_range", None)

    return LaunchSamplingSpec(
        stage_id=int(stage.stage_id),
        stage_name=str(stage.name),
        net_height=net_height,
        court_length=court_length,
        court_width=court_width,
        dt_pred=dt_pred,
        max_time_pred=max_time_pred,
        net_margin=net_margin,
        x_margin=x_margin,
        y_margin=y_margin,
        init_x_range=init_x_range,
        init_y_range=init_y_range,
        init_z_range=init_z_range,
        v_forward_range=v_forward_range,
        v_side_range=v_side_range,
        v_up_range=v_up_range,
        vx_cross_range=(vx_cross_lo, vx_cross_hi),
        drag_length_range=drag_length_range,
        hit_plane_z_range=hit_plane_z_range,
        use_near_net_init_x=use_near_net_init_x,
        center_ball_init_z_on_net=center_ball_init_z_on_net,
        z_half_span_m=z_half_span_m,
        landing_x_range=landing_x_range,
        landing_y_range=landing_y_range,
    )


def sample_uniform(value_range: tuple[float, float], count: int, device: Any):
    if torch is None:
        raise RuntimeError("Torch is required for launch sampling.")
    lo, hi = float(value_range[0]), float(value_range[1])
    return lo + (hi - lo) * torch.rand(count, device=device)


def sample_launch_candidates(
    count: int,
    spec: LaunchSamplingSpec,
    device: Any,
    include_drag_length: bool = False,
) -> dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("Torch is required for launch sampling.")

    pos_local = torch.zeros((count, 3), device=device, dtype=torch.float32)
    pos_local[:, 0] = sample_uniform(spec.init_x_range, count, device)
    pos_local[:, 1] = sample_uniform(spec.init_y_range, count, device)
    pos_local[:, 2] = sample_uniform(spec.init_z_range, count, device)

    vel_local = torch.zeros((count, 3), device=device, dtype=torch.float32)
    vel_local[:, 0] = sample_uniform(spec.vx_cross_range, count, device)
    vel_local[:, 1] = sample_uniform(spec.v_side_range, count, device)
    vel_local[:, 2] = sample_uniform(spec.v_up_range, count, device)

    result = {
        "pos_local": pos_local,
        "vel_local": vel_local,
        "hit_plane_z": sample_uniform(spec.hit_plane_z_range, count, device).to(dtype=torch.float32),
    }
    if include_drag_length:
        result["drag_length_m"] = sample_uniform(spec.drag_length_range, count, device).to(dtype=torch.float32)
    return result


def compute_hit_plane_intersections(
    positions: "torch.Tensor",
    hit_plane_z: "torch.Tensor",
    dt: float,
    eps: float = 1.0e-6,
) -> dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("Torch is required for hit-plane intersection evaluation.")

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (T, N, 3), got {tuple(positions.shape)}")

    num_steps, num_envs, _ = positions.shape
    device = positions.device
    env_idx = torch.arange(num_envs, device=device)

    if hit_plane_z.shape != (num_envs,):
        raise ValueError(f"Expected hit_plane_z shape ({num_envs},), got {tuple(hit_plane_z.shape)}")

    if num_steps < 2:
        false_mask = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        nan_time = torch.full((num_envs,), float("nan"), device=device, dtype=positions.dtype)
        nan_point = torch.full((num_envs, 3), float("nan"), device=device, dtype=positions.dtype)
        return {
            "hit_plane_ok": false_mask,
            "hit_time_s": nan_time,
            "hit_point_local": nan_point,
        }

    z = positions[:, :, 2]
    target_z = hit_plane_z.unsqueeze(0)
    hit_mask = (z[:-1] >= target_z) & (z[1:] <= target_z)
    hit_plane_ok = hit_mask.any(dim=0)
    first_hit_idx = hit_mask.to(torch.int32).argmax(dim=0).clamp(0, num_steps - 2)

    pos0 = positions[first_hit_idx, env_idx]
    pos1 = positions[first_hit_idx + 1, env_idx]
    z0 = pos0[:, 2]
    z1 = pos1[:, 2]
    denom = z1 - z0
    denom = torch.where(
        denom.abs() < eps,
        torch.where(denom >= 0.0, torch.full_like(denom, eps), torch.full_like(denom, -eps)),
        denom,
    )
    alpha = ((hit_plane_z - z0) / denom).clamp(0.0, 1.0)
    hit_point_local = pos0 + alpha.unsqueeze(-1) * (pos1 - pos0)
    hit_time_s = (first_hit_idx.to(dtype=positions.dtype) + alpha) * float(dt)

    hit_point_local = torch.where(
        hit_plane_ok.unsqueeze(-1),
        hit_point_local,
        torch.full_like(hit_point_local, float("nan")),
    )
    hit_time_s = torch.where(hit_plane_ok, hit_time_s, torch.full_like(hit_time_s, float("nan")))

    return {
        "hit_plane_ok": hit_plane_ok,
        "hit_time_s": hit_time_s,
        "hit_point_local": hit_point_local,
    }


def evaluate_launch_candidates(
    pos_local: "torch.Tensor",
    vel_local: "torch.Tensor",
    drag_length_m: "torch.Tensor",
    spec: LaunchSamplingSpec,
    hit_plane_z: "torch.Tensor",
) -> dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("Torch is required for launch evaluation.")

    trajectory = predict_shuttle_trajectory(
        pos_start=pos_local,
        vel_start=vel_local,
        mass_kg=torch.ones_like(drag_length_m),
        drag_length_m=drag_length_m,
        dt=spec.dt_pred,
        max_time=spec.max_time_pred,
        gravity=9.81,
    )
    z_at_net, has_cross, landing_xy, has_landing = evaluate_shuttle_trajectory_constraints(
        positions=trajectory,
        net_x=0.0,
        ground_z=0.0,
    )
    hit_plane_data = compute_hit_plane_intersections(
        positions=trajectory,
        hit_plane_z=hit_plane_z,
        dt=spec.dt_pred,
    )

    x_min = 0.0 + spec.x_margin
    x_max = spec.court_length - spec.x_margin
    y_min = -spec.court_width * 0.5 + spec.y_margin
    y_max = spec.court_width * 0.5 - spec.y_margin

    if spec.landing_x_range is not None:
        lx_lo, lx_hi = spec.landing_x_range
        x_min = max(x_min, lx_lo)
        x_max = min(x_max, lx_hi)
    if spec.landing_y_range is not None:
        ly_lo, ly_hi = spec.landing_y_range
        y_min = max(y_min, min(ly_lo, ly_hi))
        y_max = min(y_max, max(ly_lo, ly_hi))

    net_ok = has_cross & (z_at_net > (spec.net_height + spec.net_margin))
    landing_x = landing_xy[:, 0]
    landing_y = landing_xy[:, 1]
    landing_x_short = has_landing & (landing_x < x_min)
    landing_x_long = has_landing & (landing_x > x_max)
    landing_y_out = has_landing & ((landing_y < y_min) | (landing_y > y_max))
    landing_ok = has_landing & ~(landing_x_short | landing_x_long | landing_y_out)
    hit_plane_ok = hit_plane_data["hit_plane_ok"]
    valid = net_ok & landing_ok & hit_plane_ok

    return {
        "valid": valid,
        "has_cross": has_cross,
        "z_at_net": z_at_net,
        "net_ok": net_ok,
        "has_landing": has_landing,
        "landing_xy": landing_xy,
        "landing_ok": landing_ok,
        "landing_x_short": landing_x_short,
        "landing_x_long": landing_x_long,
        "landing_y_out": landing_y_out,
        "hit_plane_z": hit_plane_z,
        "hit_plane_ok": hit_plane_ok,
        "hit_time_s": hit_plane_data["hit_time_s"],
        "hit_point_local": hit_plane_data["hit_point_local"],
        "no_cross": ~has_cross,
        "no_landing": ~has_landing,
        "no_hit_plane": ~hit_plane_ok,
    }


def launch_violation_score(
    eval_data: dict[str, "torch.Tensor"],
    spec: LaunchSamplingSpec,
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch is required for launch scoring.")

    z_at_net = eval_data["z_at_net"]
    has_cross = eval_data["has_cross"]
    has_landing = eval_data["has_landing"]
    landing_xy = eval_data["landing_xy"]

    target_net_z = spec.net_height + spec.net_margin
    x_min = 0.0 + spec.x_margin
    x_max = spec.court_length - spec.x_margin
    y_max = spec.court_width * 0.5 - spec.y_margin

    net_penalty = torch.where(
        has_cross,
        (target_net_z - z_at_net).clamp_min(0.0),
        torch.full_like(z_at_net, 5.0),
    )

    lx = landing_xy[:, 0]
    ly = landing_xy[:, 1]
    x_penalty = (x_min - lx).clamp_min(0.0) + (lx - x_max).clamp_min(0.0)
    y_penalty = (ly.abs() - y_max).clamp_min(0.0)
    landing_penalty = torch.where(has_landing, x_penalty + y_penalty, torch.full_like(x_penalty, 5.0))
    return net_penalty + landing_penalty


def build_launch_library_metadata(spec: LaunchSamplingSpec) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_LIBRARY_SCHEMA_VERSION,
        "stage_id": int(spec.stage_id),
        "stage_name": str(spec.stage_name),
        "net_height": float(spec.net_height),
        "court_length": float(spec.court_length),
        "court_width": float(spec.court_width),
        "dt_pred": float(spec.dt_pred),
        "max_time_pred": float(spec.max_time_pred),
        "net_margin": float(spec.net_margin),
        "x_margin": float(spec.x_margin),
        "y_margin": float(spec.y_margin),
        "init_x_range": [float(v) for v in spec.init_x_range],
        "init_y_range": [float(v) for v in spec.init_y_range],
        "init_z_range": [float(v) for v in spec.init_z_range],
        "v_forward_range": [float(v) for v in spec.v_forward_range],
        "v_side_range": [float(v) for v in spec.v_side_range],
        "v_up_range": [float(v) for v in spec.v_up_range],
        "vx_cross_range": [float(v) for v in spec.vx_cross_range],
        "drag_length_range": [float(v) for v in spec.drag_length_range],
        "hit_plane_z_range": [float(v) for v in spec.hit_plane_z_range],
        "use_near_net_init_x": bool(spec.use_near_net_init_x),
        "center_ball_init_z_on_net": bool(spec.center_ball_init_z_on_net),
        "z_half_span_m": float(spec.z_half_span_m),
        "landing_x_range": [float(v) for v in spec.landing_x_range] if spec.landing_x_range else None,
        "landing_y_range": [float(v) for v in spec.landing_y_range] if spec.landing_y_range else None,
    }


def compare_launch_library_metadata(actual: dict[str, Any], expected: dict[str, Any], float_tol: float = 1.0e-6) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"missing key: {key}")
            continue
        actual_value = actual[key]
        if not _metadata_values_match(actual_value, expected_value, float_tol=float_tol):
            mismatches.append(f"{key}: actual={actual_value!r}, expected={expected_value!r}")
    return mismatches


def validate_launch_library_payload(payload: Any) -> int:
    if torch is None:
        raise RuntimeError("Torch is required for trajectory-library payload validation.")
    if not isinstance(payload, dict):
        raise ValueError("Trajectory library payload must be a dict.")

    required_keys = {
        "pos_local",
        "vel_local",
        "drag_length_m",
        "net_clearance_z",
        "landing_xy",
        "hit_plane_z",
        "hit_time_s",
        "hit_point_local",
        "metadata",
    }
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        raise ValueError(f"Trajectory library payload is missing keys: {missing_keys}")

    pos_local = payload["pos_local"]
    vel_local = payload["vel_local"]
    drag_length_m = payload["drag_length_m"]
    net_clearance_z = payload["net_clearance_z"]
    landing_xy = payload["landing_xy"]
    hit_plane_z = payload["hit_plane_z"]
    hit_time_s = payload["hit_time_s"]
    hit_point_local = payload["hit_point_local"]
    metadata = payload["metadata"]

    if not isinstance(metadata, dict):
        raise ValueError("Trajectory library metadata must be a dict.")
    if metadata.get("schema_version") != LAUNCH_LIBRARY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory library schema_version: {metadata.get('schema_version')!r}, "
            f"expected {LAUNCH_LIBRARY_SCHEMA_VERSION}"
        )

    _require_float32_tensor("pos_local", pos_local, (None, 3))
    _require_float32_tensor("vel_local", vel_local, (pos_local.shape[0], 3))
    _require_float32_tensor("drag_length_m", drag_length_m, (pos_local.shape[0],))
    _require_float32_tensor("net_clearance_z", net_clearance_z, (pos_local.shape[0],))
    _require_float32_tensor("landing_xy", landing_xy, (pos_local.shape[0], 2))
    _require_float32_tensor("hit_plane_z", hit_plane_z, (pos_local.shape[0],))
    _require_float32_tensor("hit_time_s", hit_time_s, (pos_local.shape[0],))
    _require_float32_tensor("hit_point_local", hit_point_local, (pos_local.shape[0], 3))

    num_entries = int(pos_local.shape[0])
    if num_entries <= 0:
        raise ValueError("Trajectory library payload must contain at least one entry.")
    return num_entries


def _require_float32_tensor(name: str, tensor: Any, expected_shape: tuple[int | None, ...]) -> None:
    if torch is None:
        raise RuntimeError("Torch is required for tensor validation.")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    if tensor.ndim != len(expected_shape):
        raise ValueError(f"{name} must have rank {len(expected_shape)}, got shape {tuple(tensor.shape)}.")
    for dim_idx, expected_dim in enumerate(expected_shape):
        if expected_dim is None:
            continue
        if int(tensor.shape[dim_idx]) != int(expected_dim):
            raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}; expected {expected_shape}.")


def _metadata_values_match(actual: Any, expected: Any, float_tol: float) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except Exception:
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= float_tol
        except Exception:
            return False
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_metadata_values_match(a, e, float_tol=float_tol) for a, e in zip(actual, expected))
    return actual == expected
