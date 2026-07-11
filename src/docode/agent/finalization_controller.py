from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreExportFinalizationState:
    changed_files: tuple[str, ...]
    required_files: tuple[str, ...] = ()
    diff: str = ""
    explicit_commands_fresh: bool = False
    semantic_contract_passed: bool = True
    repair_cleared: bool = True
    stale_verification: bool = False
    task_graph_complete: bool = True
    review_passed: bool = True
    summary: str = ""


@dataclass(frozen=True, slots=True)
class ExportCompletionState:
    artifact_id: str | None
    artifact_count: int


@dataclass(frozen=True, slots=True)
class FinalizationDecision:
    ready: bool
    failures: tuple[str, ...]


class FinalizationController:
    def evaluate_pre_export(self, state: PreExportFinalizationState) -> FinalizationDecision:
        failures: list[str] = []
        changed = set(state.changed_files)
        if not state.diff.strip() or not changed:
            failures.append("diff_empty")
        missing = [path for path in state.required_files if path not in changed]
        if missing:
            failures.append("required_files_missing:" + ",".join(missing))
        if changed and all(_generated_or_cache(path) for path in changed):
            failures.append("generated_or_cache_only")
        added = "\n".join(line[1:] for line in state.diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        if re.search(r"(?im)^\s*(?:#|//)?\s*(?:TODO|FIXME)\b|placeholder|not implemented", added):
            failures.append("placeholder_or_debug_marker")
        if not state.explicit_commands_fresh:
            failures.append("explicit_commands_stale")
        if not state.semantic_contract_passed:
            failures.append("semantic_contract_failed")
        if not state.repair_cleared:
            failures.append("repair_active")
        if state.stale_verification:
            failures.append("verification_stale")
        if not state.task_graph_complete:
            failures.append("task_graph_incomplete")
        if not state.review_passed:
            failures.append("review_failed")
        if not state.summary.strip():
            failures.append("summary_empty")
        return FinalizationDecision(not failures, tuple(failures))

    def evaluate_export(self, state: ExportCompletionState) -> FinalizationDecision:
        failures = []
        if not state.artifact_id:
            failures.append("artifact_id_missing")
        if state.artifact_count <= 0:
            failures.append("artifact_files_missing")
        return FinalizationDecision(not failures, tuple(failures))

    def evaluate(self, state: "FinalizationState") -> FinalizationDecision:
        pre = self.evaluate_pre_export(PreExportFinalizationState(
            state.changed_files, state.required_files, state.diff, state.explicit_commands_fresh,
            state.semantic_contract_passed, state.repair_cleared, state.stale_verification,
            True, True, state.summary,
        ))
        if not pre.ready:
            return pre
        return self.evaluate_export(ExportCompletionState("compat" if state.exporter_succeeded else None, 1 if state.exporter_succeeded else 0))


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


def _generated_or_cache(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(part in normalized.split("/") for part in ("__pycache__", "dist", "build", ".cache")) or normalized.endswith((".pyc", ".pyo"))
