from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FinalizationState:
    changed_files: tuple[str, ...]
    required_files: tuple[str, ...] = ()
    diff: str = ""
    explicit_commands_fresh: bool = False
    semantic_contract_passed: bool = True
    repair_cleared: bool = True
    stale_verification: bool = False
    summary: str = ""
    exporter_succeeded: bool = False


@dataclass(frozen=True, slots=True)
class FinalizationDecision:
    ready: bool
    failures: tuple[str, ...]


class FinalizationController:
    def evaluate(self, state: FinalizationState) -> FinalizationDecision:
        failures: list[str] = []
        changed = set(state.changed_files)
        if not state.diff.strip() or not changed:
            failures.append("diff_empty")
        missing = [path for path in state.required_files if path not in changed]
        if missing:
            failures.append("required_files_missing:" + ",".join(missing))
        if changed and all(_generated_or_cache(path) for path in changed):
            failures.append("generated_or_cache_only")
        if re.search(r"(?im)^\s*(?:#|//)?\s*(?:TODO|FIXME)\b|placeholder|not implemented", state.diff):
            failures.append("placeholder_or_debug_marker")
        if not state.explicit_commands_fresh:
            failures.append("explicit_commands_stale")
        if not state.semantic_contract_passed:
            failures.append("semantic_contract_failed")
        if not state.repair_cleared:
            failures.append("repair_active")
        if state.stale_verification:
            failures.append("verification_stale")
        if not state.summary.strip():
            failures.append("summary_empty")
        if not state.exporter_succeeded:
            failures.append("artifact_export_failed")
        return FinalizationDecision(not failures, tuple(failures))


def _generated_or_cache(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(part in normalized.split("/") for part in ("__pycache__", "dist", "build", ".cache")) or normalized.endswith((".pyc", ".pyo"))
