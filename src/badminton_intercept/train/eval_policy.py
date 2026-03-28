import argparse
from pathlib import Path
from typing import Any

from badminton_intercept.envs.intercept_env import InterceptEnv
from badminton_intercept.envs.intercept_env_cfg import DEFAULT_ENV_CFG


def _unpack_reset(reset_out: Any) -> Any:
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def run_eval(project_root: Path) -> None:
    env = InterceptEnv(cfg=DEFAULT_ENV_CFG, render_mode=None)
    obs = _unpack_reset(env.reset())
    obs_keys = list(obs.keys()) if isinstance(obs, dict) else [type(obs).__name__]
    print("[eval] env initialized")
    print(f"[eval] obs keys: {obs_keys}")
    print(f"[eval] project_root: {project_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate skeleton policy.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run_eval(project_root=args.project_root)


if __name__ == "__main__":
    main()
