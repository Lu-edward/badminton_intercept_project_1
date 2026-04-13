from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CurriculumStage:
    stage_id: int
    name: str
    ball_vel_x_range: tuple[float, float]
    ball_vel_z_range: tuple[float, float]
    uav_init_x_range: tuple[float, float] = (1.0, 3.0)
    uav_init_y_range: tuple[float, float] = (-1, 1)
    uav_init_z_range: tuple[float, float] = (1.0, 1.5)
    ball_init_x_range: tuple[float, float] = (-3.0, -2.0)
    ball_init_y_range: tuple[float, float] = (-1, 1)
    ball_init_z_range: tuple[float, float] = (1.35, 1.75)
    ball_vel_y_range: tuple[float, float] = (-1.5, 1.5)


class CurriculumManager:
    """Three-stage curriculum with iteration-streak promotion criteria."""

    def __init__(self, window_size: int = 100, promote_threshold: float = 0.85, promote_iteration_streak: int = 10):
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.promote_iteration_streak = promote_iteration_streak
        self._history: deque[int] = deque(maxlen=window_size)
        self._iteration_success_streak = 0

        self.stages = [
            CurriculumStage(
                stage_id=1,
                name="stage_i_follow",
                ball_vel_x_range=(2.0, 4.0),
                ball_vel_y_range=(-1.0, 1.0),
                ball_vel_z_range=(2.0, 6.0),
                uav_init_x_range=(1.5, 3.0),# 感觉初始坐标为1离得有点近了，先改成1.5试试
                uav_init_y_range=(-1.0, 1.0),
                uav_init_z_range=(1.0, 1.5),
                ball_init_x_range=(-3.0, -2.0),
                ball_init_y_range=(-1, 1),
                ball_init_z_range=(1.35, 1.75),
    
            ),
            CurriculumStage(
                stage_id=2,
                name="stage_ii_full_court",
                ball_vel_x_range=(4.0, 10.0),
                ball_vel_y_range=(-1.0, 1.0),
                ball_vel_z_range=(1.0, 6.0),
                uav_init_x_range=(1.0, 3.0),
                uav_init_y_range=(-1.0, 1.0),
                uav_init_z_range=(1.0, 1.5),
                ball_init_x_range=(-3.0, -2.0),
                ball_init_y_range=(-1, 1),
                ball_init_z_range=(1.35, 1.75),
                
            ),
            CurriculumStage(
                stage_id=3,
                name="stage_iii_fast_flat",
                ball_vel_x_range=(8.0, 14.0),
                ball_vel_y_range=(-1.0, 1.0),
                ball_vel_z_range=(0.2, 2.0),
                uav_init_x_range=(1.0, 3.0),
                uav_init_y_range=(-1.0, 1.0),
                uav_init_z_range=(1.0, 1.5),
                ball_init_x_range=(-3.0, -2.0),
                ball_init_y_range=(-1, 1),
                ball_init_z_range=(1.35, 1.75),
                
            ),
        ]
        self.stage_idx = 0
        self._logger: Optional[Callable[[str], None]] = None

    def set_logger(self, logger: Callable[[str], None]) -> None:
        """Set a logger callback for curriculum stage changes."""
        self._logger = logger

    def set_stage(self, stage_idx: int) -> None:
        """Set the curriculum stage directly (for resuming training from a specific stage)."""
        if 0 <= stage_idx < len(self.stages):
            self.stage_idx = stage_idx
            self._history.clear()  # Clear history when manually setting stage
            self._iteration_success_streak = 0
            
            # Log stage change if logger is set
            if self._logger is not None:
                stage = self.stages[stage_idx]
                msg = (f"[Curriculum] Manually set to stage {stage_idx + 1}: {stage.name} | "
                       f"uav_x={stage.uav_init_x_range}, y={stage.uav_init_y_range}, z={stage.uav_init_z_range}")
                self._logger(msg)

    def set_stage_by_id(self, stage_id: int) -> None:
        """Set the curriculum stage by 1-based stage id."""
        for idx, stage in enumerate(self.stages):
            if int(stage.stage_id) == int(stage_id):
                self.set_stage(idx)
                return
        raise ValueError(f"Invalid curriculum stage_id={stage_id}. Available ids: {[stage.stage_id for stage in self.stages]}")

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def get_state(self) -> dict:
        """Export a lightweight curriculum state for future resume/fine-tune workflows."""
        return {
            "stage_idx": int(self.stage_idx),
            "stage_id": int(self.current_stage.stage_id),
            "success_history": list(self._history),
            "window_size": int(self.window_size),
            "promote_threshold": float(self.promote_threshold),
            "promote_iteration_streak": int(self.promote_iteration_streak),
            "iteration_success_streak": int(self._iteration_success_streak),
        }

    def load_state(self, state: dict, strict: bool = True) -> None:
        """Restore curriculum state exported by :meth:`get_state`."""
        if not isinstance(state, dict):
            raise TypeError("Curriculum state must be a dict.")

        stage_idx = state.get("stage_idx")
        stage_id = state.get("stage_id")
        if stage_idx is not None:
            self.set_stage(int(stage_idx))
        elif stage_id is not None:
            self.set_stage_by_id(int(stage_id))
        elif strict:
            raise KeyError("Curriculum state must contain 'stage_idx' or 'stage_id'.")

        history = state.get("success_history")
        if history is not None:
            self._history.clear()
            for item in history[-self.window_size :]:
                self._history.append(1 if bool(item) else 0)
        self._iteration_success_streak = int(state.get("iteration_success_streak", 0))

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    @property
    def success_rate(self) -> float:
        if len(self._history) == 0:
            return 0.0
        return float(sum(self._history)) / float(len(self._history))

    def record_episode_outcomes(self, successes: list[bool]) -> None:
        for s in successes:
            self._history.append(1 if s else 0)

    def update(self, successes: list[bool]) -> bool:
        """Backward-compatible episode recorder. Promotion is handled per iteration."""
        self.record_episode_outcomes(successes)
        return False

    def update_iteration(self, iteration_success_rate: float) -> bool:
        """Promote only after a consecutive streak of successful training iterations."""
        if self.stage_idx >= len(self.stages) - 1:
            return False

        if float(iteration_success_rate) >= self.promote_threshold:
            self._iteration_success_streak += 1
        else:
            self._iteration_success_streak = 0

        if self._iteration_success_streak < self.promote_iteration_streak:
            return False

        old_stage_idx = self.stage_idx
        self.stage_idx += 1
        self._history.clear()
        self._iteration_success_streak = 0

        if self._logger is not None:
            new_stage = self.stages[self.stage_idx]
            msg = (
                f"[Curriculum] Promoted from stage {old_stage_idx + 1} to stage {self.stage_idx + 1}: {new_stage.name} | "
                f"iteration_success_rate={float(iteration_success_rate):.4f} >= threshold={self.promote_threshold} "
                f"for {self.promote_iteration_streak} consecutive iterations"
            )
            self._logger(msg)

        return True
