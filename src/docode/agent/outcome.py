"""Structured, immutable contracts for describing agent-step results.

Every external action (tool call, controller action, rejection, final
candidate, model failure) produces one :class:`StepOutcome`.  The outcome
answers:

* what was done
* whether it succeeded
* whether it produced real progress
* what new evidence was added
* why the task cannot yet finish (via structured :class:`FinalizationBlocker`)
* what the agent must do next

All contracts are frozen, slotted, JSON-serialisable and have stable
canonical SHA-256 fingerprints.  This module depends only on the Python
standard library and does NOT import ``AgentState``, ``TaskGraph`` or any
runtime component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import total_ordering
import hashlib
import json
from typing import Any


# ── Enums ───────────────────────────────────────────────────────────────


class RequiredAction(str, Enum):
    """What the agent MUST do next to unblock progress."""

    NONE = "none"

    # observation / investigation
    INSPECT_TARGET = "inspect_target"

    # editing
    EDIT_TARGET = "edit_target"
    REMOVE_PLACEHOLDER = "remove_placeholder"

    # verification commands
    RUN_REQUIRED_COMMAND = "run_required_command"

    # repair
    CONTINUE_REPAIR = "continue_repair"
    REPAIR_SEMANTIC_FAILURE = "repair_semantic_failure"
    REPAIR_REVIEW_FINDING = "repair_review_finding"

    # task-graph completion
    COMPLETE_TASK_NODE = "complete_task_node"

    # finalisation
    PROVIDE_FINAL_SUMMARY = "provide_final_summary"
    RETRY_EXPORT = "retry_export"

    # escalation
    CHOOSE_DIFFERENT_ACTION = "choose_different_action"
    STOP_NON_CONVERGENT = "stop_non_convergent"


class BlockerSource(str, Enum):
    """Which Runtime V2 subsystem is responsible for a blocker."""

    WORKFLOW = "workflow"
    TASK_GRAPH = "task_graph"
    VERIFICATION_SCHEDULER = "verification_scheduler"
    VERIFIER = "verifier"
    QUALITY_GATE = "quality_gate"
    REVIEW = "review"
    REPAIR_COORDINATOR = "repair_coordinator"
    FINALIZATION = "finalization"
    EXPORT = "export"
    NO_PROGRESS = "no_progress"
    TOOL = "tool"
    MODEL = "model"


class OutcomeKind(str, Enum):
    """The category of action that produced a :class:`StepOutcome`."""

    TOOL = "tool"
    CONTROLLER_ACTION = "controller_action"
    DECISION_REJECTED = "decision_rejected"
    MODEL_FAILURE = "model_failure"

    QUALITY_GATE = "quality_gate"
    VERIFICATION = "verification"
    REVIEW = "review"
    FINALIZATION = "finalization"
    EXPORT = "export"


# ── Internal helpers ────────────────────────────────────────────────────


def _stable_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalise a tuple of strings: strip, drop empty, deduplicate, sort."""
    return tuple(
        sorted(
            {
                str(value).strip()
                for value in values
                if str(value).strip()
            }
        )
    )


def _canonical_json_fingerprint(payload: dict[str, Any]) -> str:
    """Return a 64-char lowercase SHA-256 hex fingerprint of *payload*."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ── Blocker priority table ──────────────────────────────────────────────

_BLOCKER_SOURCE_PRIORITY: dict[BlockerSource, int] = {
    BlockerSource.NO_PROGRESS: 0,
    BlockerSource.REPAIR_COORDINATOR: 1,
    BlockerSource.VERIFICATION_SCHEDULER: 2,
    BlockerSource.QUALITY_GATE: 3,
    BlockerSource.VERIFIER: 4,
    BlockerSource.REVIEW: 5,
    BlockerSource.TASK_GRAPH: 6,
    BlockerSource.WORKFLOW: 7,
    BlockerSource.FINALIZATION: 8,
    BlockerSource.EXPORT: 9,
    BlockerSource.TOOL: 10,
    BlockerSource.MODEL: 11,
}


# ── FinalizationBlocker ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FinalizationBlocker:
    """A structured reason the agent cannot yet finish the task.

    Two blockers with the same ``code``, ``source``, ``required_action``
    and normalised tuple fields produce the **same** fingerprint regardless
    of ``message`` or ``evidence_refs``.  The *message* is for human
    display only.
    """

    code: str
    source: BlockerSource
    message: str
    required_action: RequiredAction

    related_files: tuple[str, ...] = ()
    related_commands: tuple[str, ...] = ()
    related_node_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    retryable: bool = True

    def __post_init__(self) -> None:
        # --- code ---
        code = str(self.code).strip()
        if not code:
            raise ValueError("blocker code must not be empty")
        object.__setattr__(self, "code", code)

        # --- message ---
        message = str(self.message).strip()
        if not message:
            message = code
        object.__setattr__(self, "message", message)

        # --- required_action ---
        if not isinstance(self.required_action, RequiredAction):
            object.__setattr__(
                self,
                "required_action",
                RequiredAction(str(self.required_action)),
            )

        # --- normalise tuples ---
        object.__setattr__(
            self,
            "related_files",
            tuple(
                value.replace("\\", "/")
                for value in _stable_strings(self.related_files)
            ),
        )
        object.__setattr__(
            self,
            "related_commands",
            _stable_strings(self.related_commands),
        )
        object.__setattr__(
            self,
            "related_node_ids",
            _stable_strings(self.related_node_ids),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _stable_strings(self.evidence_refs),
        )

    # -- fingerprint -------------------------------------------------------

    def fingerprint(self) -> str:
        """64-char hex SHA-256 committed only to the blocker's semantic identity.

        ``message`` and ``evidence_refs`` are intentionally excluded so
        that two blockers that describe the same logical obstacle share
        the same fingerprint.
        """
        payload: dict[str, Any] = {
            "code": self.code,
            "source": self.source.value,
            "required_action": self.required_action.value,
            "related_files": list(self.related_files),
            "related_commands": list(self.related_commands),
            "related_node_ids": list(self.related_node_ids),
            "retryable": self.retryable,
        }
        return _canonical_json_fingerprint(payload)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary representation."""
        return {
            "code": self.code,
            "source": self.source.value,
            "message": self.message,
            "required_action": self.required_action.value,
            "related_files": list(self.related_files),
            "related_commands": list(self.related_commands),
            "related_node_ids": list(self.related_node_ids),
            "evidence_refs": list(self.evidence_refs),
            "retryable": self.retryable,
            "fingerprint": self.fingerprint(),
        }


# ── StepOutcome ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """The unified, immutable result of one agent / controller action.

    One action → one ``StepOutcome``.  Intermediate stage events
    (quality_gate, verifier, review …) may continue to emit their own
    granular repository steps, but the *final* outcome for the action
    is a single instance of this contract.
    """

    kind: OutcomeKind
    action_key: str
    success: bool

    progress: bool
    progress_reasons: tuple[str, ...] = ()

    state_fingerprint_before: str = ""
    state_fingerprint_after: str = ""

    workspace_changed: bool = False
    evidence_added: tuple[str, ...] = ()
    completed_node_ids: tuple[str, ...] = ()
    invalidated_node_ids: tuple[str, ...] = ()

    blockers: tuple[FinalizationBlocker, ...] = ()
    next_required_action: RequiredAction = RequiredAction.NONE

    retryable: bool = True
    failure_class: str | None = None

    def __post_init__(self) -> None:
        # --- action_key ---
        key = str(self.action_key).strip()
        if not key:
            raise ValueError("action_key must not be empty")
        object.__setattr__(self, "action_key", key)

        # --- normalise string tuples ---
        object.__setattr__(
            self,
            "progress_reasons",
            _stable_strings(self.progress_reasons),
        )
        object.__setattr__(
            self,
            "evidence_added",
            _stable_strings(self.evidence_added),
        )
        object.__setattr__(
            self,
            "completed_node_ids",
            _stable_strings(self.completed_node_ids),
        )
        object.__setattr__(
            self,
            "invalidated_node_ids",
            _stable_strings(self.invalidated_node_ids),
        )

        # --- deduplicate & sort blockers by fingerprint ---
        if self.blockers:
            seen: dict[str, FinalizationBlocker] = {}
            for blocker in self.blockers:
                fp = blocker.fingerprint()
                if fp not in seen:
                    seen[fp] = blocker
            object.__setattr__(
                self,
                "blockers",
                tuple(seen[fp] for fp in sorted(seen)),
            )

        # --- next_required_action ---
        if not isinstance(self.next_required_action, RequiredAction):
            object.__setattr__(
                self,
                "next_required_action",
                RequiredAction(str(self.next_required_action)),
            )

    # -- primary blocker ---------------------------------------------------

    def primary_blocker(self) -> FinalizationBlocker | None:
        """Return the most important blocker, or *None*.

        Priority is source-based (``NO_PROGRESS`` > ``REPAIR_COORDINATOR``
        > … > ``MODEL``) with ``code`` and ``fingerprint`` as tie-breaks.
        """
        if not self.blockers:
            return None
        return min(
            self.blockers,
            key=lambda blocker: (
                _BLOCKER_SOURCE_PRIORITY[blocker.source],
                blocker.code,
                blocker.fingerprint(),
            ),
        )

    def effective_required_action(self) -> RequiredAction:
        """The action the agent should take next.

        If ``next_required_action`` is set it takes precedence; otherwise
        falls back to the primary blocker's suggestion.
        """
        if self.next_required_action != RequiredAction.NONE:
            return self.next_required_action
        primary = self.primary_blocker()
        if primary is not None:
            return primary.required_action
        return RequiredAction.NONE

    # -- fingerprint -------------------------------------------------------

    def fingerprint(self) -> str:
        """64-char hex SHA-256 representing the complete outcome identity.

        This is a **data-contract** fingerprint used for testing.  Future
        no-progress logic must use ``state_progress_fingerprint`` (Phase 3)
        rather than this value alone.
        """
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "action_key": self.action_key,
            "success": self.success,
            "progress": self.progress,
            "progress_reasons": list(self.progress_reasons),
            "state_fingerprint_before": self.state_fingerprint_before,
            "state_fingerprint_after": self.state_fingerprint_after,
            "workspace_changed": self.workspace_changed,
            "evidence_added": list(self.evidence_added),
            "completed_node_ids": list(self.completed_node_ids),
            "invalidated_node_ids": list(self.invalidated_node_ids),
            "blocker_fingerprints": [
                blocker.fingerprint() for blocker in self.blockers
            ],
            "effective_required_action": self.effective_required_action().value,
            "retryable": self.retryable,
            "failure_class": self.failure_class,
        }
        return _canonical_json_fingerprint(payload)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary representation."""
        primary = self.primary_blocker()
        return {
            "kind": self.kind.value,
            "action_key": self.action_key,
            "success": self.success,
            "progress": self.progress,
            "progress_reasons": list(self.progress_reasons),
            "state_fingerprint_before": self.state_fingerprint_before,
            "state_fingerprint_after": self.state_fingerprint_after,
            "workspace_changed": self.workspace_changed,
            "evidence_added": list(self.evidence_added),
            "completed_node_ids": list(self.completed_node_ids),
            "invalidated_node_ids": list(self.invalidated_node_ids),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "primary_blocker": primary.to_dict() if primary else None,
            "next_required_action": self.next_required_action.value,
            "effective_required_action": self.effective_required_action().value,
            "retryable": self.retryable,
            "failure_class": self.failure_class,
            "fingerprint": self.fingerprint(),
        }
