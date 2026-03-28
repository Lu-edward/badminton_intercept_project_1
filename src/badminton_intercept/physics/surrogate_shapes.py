from dataclasses import dataclass


@dataclass
class SurrogateShape:
    shape_type: str = "sphere"
    radius_m: float = 0.035


def create_shuttlecock_surrogate() -> SurrogateShape:
    return SurrogateShape()
