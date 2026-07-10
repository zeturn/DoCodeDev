def stable_identifier(value: str) -> str:
    return "-".join(value.lower().split())
