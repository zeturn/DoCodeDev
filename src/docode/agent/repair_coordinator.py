from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RepairPhase(str, Enum):
    INSPECT_ALLOWED = "inspect_allowed"
    EDIT_REQUIRED = "edit_required"
    RERUN_PRODUCER = "rerun_producer"
    RERUN_VALIDATOR = "rerun_validator"
    RESOLVED = "resolved"
    NON_CONVERGENT = "non_convergent"


@dataclass(slots=True)
class RepairAction:
    failure_class: str
    signature: str
    target_files: list[str]
    evidence_files: list[str]
    artifact_path: str = ""
    producer_command: str = ""
    validator_command: str = ""
    instruction: str = ""
    attempt: int = 1
    maximum_attempts: int = 3


class RepairCoordinator:
    def __init__(self, maximum_attempts: int = 3) -> None:
        self.maximum_attempts = maximum_attempts
        self.phase = RepairPhase.INSPECT_ALLOWED
        self._attempts: dict[str, int] = {}

    def activate(self, action: RepairAction) -> RepairPhase:
        attempt = self._attempts.get(action.signature, 0) + 1
        self._attempts[action.signature] = attempt
        action.attempt = attempt
        action.maximum_attempts = self.maximum_attempts
        self.phase = RepairPhase.NON_CONVERGENT if attempt >= self.maximum_attempts else RepairPhase.EDIT_REQUIRED
        return self.phase

    def record_edit(self) -> RepairPhase:
        if self.phase != RepairPhase.NON_CONVERGENT:
            self.phase = RepairPhase.RERUN_PRODUCER
        return self.phase

    def record_producer(self, passed: bool, artifact_improved: bool = False) -> RepairPhase:
        if self.phase == RepairPhase.NON_CONVERGENT:
            return self.phase
        self.phase = RepairPhase.RERUN_VALIDATOR if passed else RepairPhase.EDIT_REQUIRED
        if artifact_improved:
            self._attempts.clear()
        return self.phase

    def record_validator(self, passed: bool) -> RepairPhase:
        self.phase = RepairPhase.RESOLVED if passed else RepairPhase.EDIT_REQUIRED
        if passed:
            self._attempts.clear()
        return self.phase
