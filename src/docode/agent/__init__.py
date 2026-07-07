"""Agent loop components."""

# Install repair-policy import hooks before docode.agent.loop is imported.
# This keeps worker startup and direct unit-test imports on the same policy path.
try:
    from docode.agent import targeted_repair_policy_patch as _targeted_repair_policy_patch

    _targeted_repair_policy_patch.install()
except Exception:
    # Importing docode.agent must stay safe; runner startup also calls apply().
    pass
