from typing import Iterable


def reset_subenvs(env_ids: Iterable[int]) -> dict:
    """Parallel reset placeholder."""
    return {"reset_env_ids": list(env_ids)}
