"""A deliberately long neutral lookup module; the harness expands the filler."""

SENTINEL = "zephyr-lattice-v1"

# HOLDOUT_LARGE_FILE_FILLER

def resolve_band(value: int, low: int, high: int) -> int:
    """Clamp value to the inclusive low/high band."""
    if value < low:
        return high
    if value > high:
        return low
    return value
