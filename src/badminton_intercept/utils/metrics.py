def summarize_metrics(hit_rate: float, sweet_spot_rate: float, energy: float) -> dict:
    return {
        "hit_rate": hit_rate,
        "sweet_spot_rate": sweet_spot_rate,
        "energy": energy,
    }
