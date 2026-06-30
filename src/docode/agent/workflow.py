from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from docode.agent.state import AgentState
from docode.agent.stuck import git_status_clean
from docode.agent.task_contract import TaskContract


class WorkflowPhase(str, Enum):
    INSPECT = "INSPECT"
    PLAN = "PLAN"
    EDIT_REQUIRED = "EDIT_REQUIRED"
    TEST_REQUIRED = "TEST_REQUIRED"
    VERIFY_READY = "VERIFY_READY"
    FINAL_READY = "FINAL_READY"


@dataclass(frozen=True, slots=True)
class WorkflowSnapshot:
    phase: WorkflowPhase
    diff_exists: bool
    tests_run: bool
    final_allowed: bool
    reason: str
    required_action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "diff_exists": self.diff_exists,
            "tests_run": self.tests_run,
            "final_allowed": self.final_allowed,
            "reason": self.reason,
            "required_action": self.required_action,
        }


@dataclass(frozen=True, slots=True)
class FinalGate:
    allowed: bool
    reason: str
    detail: str
    snapshot: WorkflowSnapshot
    repair_mode: str | None = None


def workflow_snapshot(state: AgentState, git_status_output: str) -> WorkflowSnapshot:
    diff_exists = not git_status_clean(git_status_output)
    tests_run = required_commands_satisfied(state)
    if state.inspection is None:
        return WorkflowSnapshot(
            phase=WorkflowPhase.INSPECT,
            diff_exists=diff_exists,
            tests_run=tests_run,
            final_allowed=False,
            reason="inspection_missing",
            required_action="inspect the repository before planning or editing",
        )
    if not diff_exists:
        return WorkflowSnapshot(
            phase=WorkflowPhase.EDIT_REQUIRED,
            diff_exists=False,
            tests_run=tests_run,
            final_allowed=False,
            reason="no_diff",
            required_action="modify a target file and confirm git_status shows a change",
        )
    if state.repair_mode == "must_edit":
        return WorkflowSnapshot(
            phase=WorkflowPhase.EDIT_REQUIRED,
            diff_exists=True,
            tests_run=tests_run,
            final_allowed=False,
            reason="repair_mode_requires_edit",
            required_action="make or confirm an edit with an allowed repair tool before final_candidate",
        )
    if required_commands(state.task_contract) and not tests_run:
        return WorkflowSnapshot(
            phase=WorkflowPhase.TEST_REQUIRED,
            diff_exists=True,
            tests_run=False,
            final_allowed=False,
            reason="required_tests_missing",
            required_action="run the required verification command before final_candidate",
        )
    return WorkflowSnapshot(
        phase=WorkflowPhase.FINAL_READY,
        diff_exists=True,
        tests_run=tests_run,
        final_allowed=True,
        reason="ready",
        required_action="submit final_candidate for verifier review",
    )


def final_candidate_gate(state: AgentState, git_status_output: str) -> FinalGate:
    snapshot = workflow_snapshot(state, git_status_output)
    if snapshot.final_allowed:
        return FinalGate(allowed=True, reason="final_allowed", detail=snapshot.required_action, snapshot=snapshot)
    if snapshot.reason == "repair_mode_requires_edit":
        return FinalGate(
            allowed=False,
            reason="repair_mode_final_forbidden",
            detail="final_candidate is blocked while repair_mode=must_edit. Modify a target file and confirm git_status first.",
            snapshot=snapshot,
        )
    if snapshot.reason == "no_diff":
        return FinalGate(
            allowed=False,
            reason="final_candidate_clean_git_status",
            detail=(
                "Final candidate rejected before verification: git status is clean. "
                "You must modify files with edit_file/write_file/apply_patch first."
            ),
            snapshot=snapshot,
            repair_mode="must_edit",
        )
    if snapshot.reason == "required_tests_missing":
        commands = ", ".join(required_commands(state.task_contract))
        return FinalGate(
            allowed=False,
            reason="final_candidate_tests_missing",
            detail=f"Final candidate rejected before verification: run required verification first: {commands}",
            snapshot=snapshot,
        )
    return FinalGate(
        allowed=False,
        reason=f"workflow_not_ready:{snapshot.reason}",
        detail=snapshot.required_action,
        snapshot=snapshot,
    )


def required_commands_satisfied(state: AgentState) -> bool:
    commands = required_commands(state.task_contract)
    if not commands:
        return True
    return all(command_was_run(state, command) for command in commands)


def required_commands(task_contract: TaskContract | None) -> list[str]:
    return list(task_contract.must_run_commands) if task_contract is not None else []


def command_was_run(state: AgentState, command: str) -> bool:
    expected = normalize_command(command)
    for message in state.messages:
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = normalize_command(str(metadata.get("command") or ""))
        if observed and (observed == expected or expected in observed):
            return True
        tool = str(message.get("tool") or "")
        if tool == "run_tests" and ("test" in expected or "pytest" in expected or "unittest" in expected):
            return True
    return False


def normalize_command(command: str) -> str:
    return " ".join(command.strip().split())
