DEFAULTS = {"timeout": 30, "retries": 3, "mode": "safe"}


def load(overrides=None):
    merged = dict(DEFAULTS)
    if overrides:
        # BUG: copies the default value instead of applying the override.
        for key in overrides:
            merged[key] = DEFAULTS[key]
    return merged
