"""DoCode autonomous coding runtime."""

__version__ = "0.1.0"

try:
    from docode._runtime_hotfix_source_pipeline import apply_runtime_hotfix

    apply_runtime_hotfix()
except Exception:
    pass
