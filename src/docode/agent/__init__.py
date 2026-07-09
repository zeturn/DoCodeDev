"""Agent loop components."""

# Install runtime loop fixes at package import time.
#
# This keeps the patch narrowly scoped while avoiding a risky full rewrite of
# src/docode/agent/loop.py through the remote GitHub contents API. The patch is
# idempotent and only adjusts generic repair-control behavior.
try:  # pragma: no cover - import side-effect guard.
    from docode.agent import loop as _loop
    from docode.agent.runtime_fixes import apply_loop_runtime_fixes

    apply_loop_runtime_fixes(_loop)
except Exception:
    # Importing this package should never fail because a patch hook failed.
    # Tests that import the loop module directly will surface any real issue.
    pass
