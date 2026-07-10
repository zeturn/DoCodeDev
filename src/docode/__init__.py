"""DoCode autonomous coding runtime."""

__version__ = "0.1.0"
__runtime_hotfix_applied__ = False
__runtime_hotfix_error__: str | None = None

try:
    from docode._runtime_hotfix_source_pipeline import apply_runtime_hotfix

    apply_runtime_hotfix()
    __runtime_hotfix_applied__ = True
except Exception as exc:
    __runtime_hotfix_error__ = f"{type(exc).__name__}: {exc}"
