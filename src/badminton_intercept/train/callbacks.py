from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainCallbacks:
    enable_console_log: bool = True

    def on_train_start(self) -> None:
        if self.enable_console_log:
            print("[callbacks] training started")

    def on_iteration_end(self, iteration: int, metrics: dict[str, Any]) -> None:
        if self.enable_console_log:
            print(f"[callbacks] iter={iteration} metrics={metrics}")

    def on_train_end(self, output_dir: Path) -> None:
        if self.enable_console_log:
            print(f"[callbacks] training ended, output_dir={output_dir}")
