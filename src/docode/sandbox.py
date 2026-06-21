from __future__ import annotations


DEFAULT_SANDBOX_NETWORK_MODE = "project"
SANDBOX_NETWORK_MODES = frozenset({"project", "bridge", "no_internet", "no-internet", "internal", "offline"})


def normalize_sandbox_network_mode(value: str | None) -> str:
    mode = (value or DEFAULT_SANDBOX_NETWORK_MODE).strip().lower()
    if mode == "":
        return DEFAULT_SANDBOX_NETWORK_MODE
    if mode not in SANDBOX_NETWORK_MODES:
        raise ValueError("sandbox_network_mode must be project or no_internet")
    if mode in {"bridge"}:
        return "project"
    if mode in {"no-internet", "internal", "offline"}:
        return "no_internet"
    return mode
