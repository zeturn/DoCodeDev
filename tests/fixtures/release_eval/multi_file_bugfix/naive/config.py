DEFAULTS = {"timeout": 30, "retries": 3, "mode": "safe"}


def load(overrides=None):
    merged = dict(DEFAULTS)
    if overrides:
        merged.update(overrides)
    return merged
