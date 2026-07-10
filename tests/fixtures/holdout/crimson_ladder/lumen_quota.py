def normalize_units(units: int) -> int
    if units < 0:
        raise ValueError("units must be non-negative")
    return units


def reserve_slots(units: int, ceiling: int) -> int:
    if ceiling < 0:
        raise ValueError("ceiling must be non-negative")
    return normalize_units(units) + ceiling
