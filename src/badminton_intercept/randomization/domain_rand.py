import random


def uniform_sample(low: float, high: float) -> float:
    return random.uniform(low, high)


def sample_domain_params() -> dict:
    """Sample domain-randomization parameters for one episode."""
    return {
        "racket_restitution": uniform_sample(0.75, 0.90),
        "shuttle_mass_kg": uniform_sample(0.0045, 0.0055),
        "drag_length_m": uniform_sample(3.8, 4.4),
        "drone_mass_scale": uniform_sample(0.95, 1.05),
        "drone_inertia_scale": uniform_sample(0.95, 1.05),
        "thrust_coeff_scale": uniform_sample(0.95, 1.05),
    }
