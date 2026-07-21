from __future__ import annotations

from enum import Enum
from docode.agent.repair_planner import RepairAction


class RepairPhase(str, Enum):
    LOCATE_PRODUCER = "locate_producer"
    INSPECT_ALLOWED = "inspect_allowed"
    EDIT_REQUIRED = "edit_required"
    RERUN_PRODUCER = "rerun_producer"
    RERUN_VALIDATOR = "rerun_validator"
    RESOLVED = "resolved"
    NON_CONVERGENT = "non_convergent"


class RepairCoordinator:
    def __init__(self, maximum_attempts: int = 3) -> None:
        self.maximum_attempts = maximum_attempts
        self.phase = RepairPhase.INSPECT_ALLOWED
        self._attempts: dict[str, int] = {}
        self.transitions: list[dict[str, object]] = []

    def _transition(self, phase: RepairPhase, reason: str, evidence_refs: list[str] | None = None) -> RepairPhase:
        previous = self.phase
        self.phase = phase
        self.transitions.append({"previous_phase": previous.value, "new_phase": phase.value, "reason": reason, "evidence_refs": list(evidence_refs or [])})
        return phase

    def activate(self, action: RepairAction) -> RepairPhase:
        attempt = self._attempts.get(action.signature, 0) + 1
        self._attempts[action.signature] = attempt
        if action.artifact_ownership == "generated" and not action.target_files:
            return self._transition(RepairPhase.LOCATE_PRODUCER, "generated artifact producer is unknown", action.evidence_refs)
        return self._transition(RepairPhase.NON_CONVERGENT if attempt >= self.maximum_attempts else RepairPhase.EDIT_REQUIRED, "repair activated", action.evidence_refs)

    def attempt_count(self, signature: str) -> int:
        return self._attempts.get(signature, 0)

    def record_edit(self) -> RepairPhase:
        if self.phase != RepairPhase.NON_CONVERGENT:
            self._transition(RepairPhase.RERUN_PRODUCER, "repair target edited")
        return self.phase

    def record_producer(self, passed: bool, artifact_improved: bool = False) -> RepairPhase:
        if self.phase == RepairPhase.NON_CONVERGENT:
            return self.phase
        self._transition(RepairPhase.RERUN_VALIDATOR if passed else RepairPhase.EDIT_REQUIRED, "producer command passed" if passed else "producer command failed")
        if artifact_improved:
            self._attempts.clear()
        return self.phase

    def record_validator(self, passed: bool) -> RepairPhase:
        self._transition(RepairPhase.RESOLVED if passed else RepairPhase.EDIT_REQUIRED, "validator passed" if passed else "validator failed")
        if passed:
            self._attempts.clear()
        return self.phase

    def resolve_if_verified(self, *, scheduler_fresh: bool, quality_passed: bool, review_passed: bool, edit_epoch: int) -> bool:
        if self.phase in {RepairPhase.INSPECT_ALLOWED, RepairPhase.RESOLVED}:
            return True
        if self.phase == RepairPhase.NON_CONVERGENT or not (scheduler_fresh and quality_passed and review_passed):
            return False
        self._attempts.clear()
        self._transition(RepairPhase.RESOLVED, "current revision passed scheduler, quality, and review", [f"edit_epoch:{edit_epoch}"])
        return True
