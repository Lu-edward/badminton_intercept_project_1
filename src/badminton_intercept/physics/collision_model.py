def get_restitution(is_sweet_spot: bool, sweet_spot: float = 0.82, edge: float = 0.64) -> float:
    """Return restitution for racket collision region."""
    return sweet_spot if is_sweet_spot else edge
