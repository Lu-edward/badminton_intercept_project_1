import math
import random
from dataclasses import dataclass


@dataclass
class LaunchState:
    speed_mps: float
    elevation_rad: float
    heading_rad: float


def sample_launch_state(speed_range=(8.0, 18.0), elevation_range_deg=(10.0, 55.0)) -> LaunchState:
    speed = random.uniform(*speed_range)
    elevation = math.radians(random.uniform(*elevation_range_deg))
    heading = random.uniform(-math.pi, math.pi)
    return LaunchState(speed_mps=speed, elevation_rad=elevation, heading_rad=heading)
