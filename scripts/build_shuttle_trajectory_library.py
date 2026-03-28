from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from badminton_intercept.envs.intercept_env_cfg import InterceptEnvCfg
from badminton_intercept.envs.launch_sampler import (
    build_launch_library_metadata,
    build_launch_sampling_spec,
    evaluate_launch_candidates,
    get_curriculum_stages,
    resolve_launch_library_stage_paths,
    sample_launch_candidates,
    validate_launch_library_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline shuttle launch trajectory libraries.")
    parser.add_argument("--stage-counts", nargs=3, type=int, default=(200000, 500000, 300000))
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.stage_counts) != 3:
        raise ValueError("--stage-counts must contain exactly three values.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    cfg = InterceptEnvCfg()
    output_paths = resolve_launch_library_stage_paths(cfg)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_paths = tuple(output_dir / path.name for path in output_paths)

    output_dir = output_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] project_root={PROJECT_ROOT}")
    print(f"[build] output_dir={output_dir}")
    print(f"[build] device={device}")
    print(f"[build] seed={args.seed}")

    for stage, target_count, output_path in zip(get_curriculum_stages(), args.stage_counts, output_paths):
        payload = build_stage_library(cfg=cfg, stage=stage, target_count=int(target_count), batch_size=int(args.batch_size), device=device)
        torch.save(payload, output_path)
        print(f"[build] wrote stage {stage.stage_id} ({stage.name}) -> {output_path}")

    return 0


def build_stage_library(cfg: InterceptEnvCfg, stage, target_count: int, batch_size: int, device: torch.device) -> dict[str, object]:
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    spec = build_launch_sampling_spec(cfg, stage)
    pos_chunks: list[torch.Tensor] = []
    vel_chunks: list[torch.Tensor] = []
    drag_chunks: list[torch.Tensor] = []
    clearance_chunks: list[torch.Tensor] = []
    landing_chunks: list[torch.Tensor] = []
    hit_plane_chunks: list[torch.Tensor] = []
    hit_time_chunks: list[torch.Tensor] = []
    hit_point_chunks: list[torch.Tensor] = []

    total_valid = 0
    total_sampled = 0

    while total_valid < target_count:
        current_batch = max(batch_size, min(batch_size, target_count - total_valid))
        sampled = sample_launch_candidates(current_batch, spec, device=device, include_drag_length=True)
        eval_data = evaluate_launch_candidates(
            pos_local=sampled["pos_local"],
            vel_local=sampled["vel_local"],
            drag_length_m=sampled["drag_length_m"],
            spec=spec,
            hit_plane_z=sampled["hit_plane_z"],
        )

        valid_idx = torch.nonzero(eval_data["valid"], as_tuple=False).squeeze(-1)
        total_sampled += current_batch
        if valid_idx.numel() == 0:
            print(
                f"[build][stage {stage.stage_id}] sampled={total_sampled} valid={total_valid} "
                f"acceptance=0.0000"
            )
            continue

        take = min(target_count - total_valid, int(valid_idx.numel()))
        chosen_idx = valid_idx[:take]
        pos_chunks.append(sampled["pos_local"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        vel_chunks.append(sampled["vel_local"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        drag_chunks.append(sampled["drag_length_m"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        clearance = eval_data["z_at_net"].index_select(0, chosen_idx) - float(spec.net_height)
        clearance_chunks.append(clearance.to(device="cpu", dtype=torch.float32))
        landing_chunks.append(eval_data["landing_xy"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        hit_plane_chunks.append(sampled["hit_plane_z"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        hit_time_chunks.append(eval_data["hit_time_s"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        hit_point_chunks.append(eval_data["hit_point_local"].index_select(0, chosen_idx).to(device="cpu", dtype=torch.float32))
        total_valid += take

        acceptance = float(total_valid) / float(total_sampled)
        print(
            f"[build][stage {stage.stage_id}] sampled={total_sampled} valid={total_valid}/{target_count} "
            f"acceptance={acceptance:.4f}"
        )

    payload = {
        "pos_local": torch.cat(pos_chunks, dim=0).contiguous(),
        "vel_local": torch.cat(vel_chunks, dim=0).contiguous(),
        "drag_length_m": torch.cat(drag_chunks, dim=0).contiguous(),
        "net_clearance_z": torch.cat(clearance_chunks, dim=0).contiguous(),
        "landing_xy": torch.cat(landing_chunks, dim=0).contiguous(),
        "hit_plane_z": torch.cat(hit_plane_chunks, dim=0).contiguous(),
        "hit_time_s": torch.cat(hit_time_chunks, dim=0).contiguous(),
        "hit_point_local": torch.cat(hit_point_chunks, dim=0).contiguous(),
        "metadata": {
            **build_launch_library_metadata(spec),
            "num_entries": int(target_count),
            "num_sampled": int(total_sampled),
            "acceptance_rate": float(target_count) / float(total_sampled),
        },
    }
    validate_launch_library_payload(payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
