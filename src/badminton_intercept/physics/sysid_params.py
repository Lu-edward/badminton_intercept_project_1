from dataclasses import dataclass


@dataclass
class SysIdParams:
    mass_kg: float = 1.35
    inertia_xx: float = 0.021
    inertia_yy: float = 0.021
    inertia_zz: float = 0.038
    thrust_coeff: float = 1.0
    torque_coeff: float = 0.01


DEFAULT_SYSID = SysIdParams()
