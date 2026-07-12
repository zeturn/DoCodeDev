from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from docode.agent.outcome import (
    BlockerSource,
    FinalizationBlocker,
    RequiredAction,
)

if TYPE_CHECKING:
    pass


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
    blockers: tuple[FinalizationBlocker, ...]

    @property
    def failures(self) -> tuple[str, ...]:
        """Backward-compatible accessor returning blocker codes."""
        return tuple(blocker.code for blocker in self.blockers)


class FinalizationController:
    def evaluate_pre_export(self, state: PreExportFinalizationState) -> FinalizationDecision:
        blockers: list[FinalizationBlocker] = []
        changed = set(state.changed_files)

        if not state.diff.strip() or not changed:
            blockers.append(_make(
                "diff_empty", BlockerSource.FINALIZATION,
                RequiredAction.EDIT_TARGET,
                message="No non-empty diff or changed files",
            ))

        missing = [path for path in state.required_files if path not in changed]
        if missing:
            blockers.append(_make(
                "required_files_missing", BlockerSource.FINALIZATION,
                RequiredAction.EDIT_TARGET,
                message=f"Required files not modified: {', '.join(missing)}",
                related_files=tuple(missing),
            ))

        if changed and all(_generated_or_cache(path) for path in changed):
            blockers.append(_make(
                "generated_or_cache_only", BlockerSource.FINALIZATION,
                RequiredAction.EDIT_TARGET,
                message="Only generated/cache files changed",
            ))

        added = "\n".join(line[1:] for line in state.diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        if re.search(r"(?im)^\s*(?:#|//)?\s*(?:TODO|FIXME)\b|placeholder|not implemented", added):
            blockers.append(_make(
                "placeholder_or_debug_marker", BlockerSource.FINALIZATION,
                RequiredAction.REMOVE_PLACEHOLDER,
                message="Placeholder/TODO/stub text in diff",
            ))

        if not state.explicit_commands_fresh:
            blockers.append(_make(
                "explicit_commands_stale", BlockerSource.VERIFICATION_SCHEDULER,
                RequiredAction.RUN_REQUIRED_COMMAND,
                message="Explicit verification commands are stale",
            ))

        if not state.semantic_contract_passed:
            blockers.append(_make(
                "semantic_contract_failed", BlockerSource.QUALITY_GATE,
                RequiredAction.REPAIR_SEMANTIC_FAILURE,
                message="Semantic contract / quality gate failed",
            ))

        if not state.repair_cleared:
            blockers.append(_make(
                "repair_active", BlockerSource.REPAIR_COORDINATOR,
                RequiredAction.CONTINUE_REPAIR,
                message="Repair still active",
            ))

        if state.stale_verification:
            blockers.append(_make(
                "verification_stale", BlockerSource.VERIFICATION_SCHEDULER,
                RequiredAction.RUN_REQUIRED_COMMAND,
                message="Verification evidence is stale",
            ))

        if not state.task_graph_complete:
            blockers.append(_make(
                "task_graph_incomplete", BlockerSource.TASK_GRAPH,
                RequiredAction.COMPLETE_TASK_NODE,
                message="Task graph incomplete",
            ))

        if not state.review_passed:
            blockers.append(_make(
                "review_failed", BlockerSource.REVIEW,
                RequiredAction.REPAIR_REVIEW_FINDING,
                message="Independent review failed",
            ))

        if not state.summary.strip():
            blockers.append(_make(
                "summary_empty", BlockerSource.FINALIZATION,
                RequiredAction.PROVIDE_FINAL_SUMMARY,
                message="No final summary provided",
            ))

        return FinalizationDecision(not blockers, tuple(blockers))

    def evaluate_export(self, state: ExportCompletionState) -> FinalizationDecision:
        blockers: list[FinalizationBlocker] = []
        if not state.artifact_id:
            blockers.append(_make(
                "artifact_id_missing", BlockerSource.EXPORT,
                RequiredAction.RETRY_EXPORT,
                message="Artifact ID missing",
            ))
        if state.artifact_count <= 0:
            blockers.append(_make(
                "artifact_files_missing", BlockerSource.EXPORT,
                RequiredAction.RETRY_EXPORT,
                message="No artifact files produced",
            ))
        return FinalizationDecision(not blockers, tuple(blockers))

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


# ── adapter helpers ──────────────────────────────────────────────────────


def _make(
    code: str,
    source: BlockerSource,
    action: RequiredAction,
    message: str = "",
    related_files: tuple[str, ...] = (),
    related_commands: tuple[str, ...] = (),
    related_node_ids: tuple[str, ...] = (),
) -> FinalizationBlocker:
    return FinalizationBlocker(
        code=code,
        source=source,
        message=message or code,
        required_action=action,
        related_files=related_files,
        related_commands=related_commands,
        related_node_ids=related_node_ids,
    )
