"""Agent loop components."""

# Runtime policy overrides for the targeted repair loop.
#
# Keep this small and import-safe: the patch only replaces helper functions in
# docode.agent.loop after that module is imported. This lets the repair policy be
# hardened without rewriting the large loop module in one commit.
try:
    from docode.agent import targeted_repair_policy_patch as _targeted_repair_policy_patch

    _targeted_repair_policy_patch.apply()
except Exception:
    # Never make importing docode.agent fail because a defensive policy patch
    # could not be installed. The normal loop code remains available.
    pass
