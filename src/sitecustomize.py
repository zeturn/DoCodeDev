from __future__ import annotations

# Python imports sitecustomize automatically when this repository's src/
# directory is on PYTHONPATH. Use that hook to install small, generic runtime
# loop fixes even when callers import docode.agent.loop directly.
try:  # pragma: no cover - import-time safety hook.
    from docode.agent import loop as _loop
    from docode.agent.runtime_fixes import apply_loop_runtime_fixes

    apply_loop_runtime_fixes(_loop)
except Exception:
    # Do not make interpreter startup fail because an optional patch hook failed.
    pass
