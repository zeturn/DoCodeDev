from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from docode.agent.state import AgentState
from docode.agent.task_contract import TaskContract


EDIT_TOOLS = {"edit_file", "write_file", "apply_patch", "replace_in_file"}
FOCUSED_READ_TOOLS = {"read_file", "read_file_range", "read_symbol"}


class WorkflowPhase(str, Enum):
    INSPECT = "INSPECT"
    PLAN = "PLAN"
    EDIT_REQUIRED = "EDIT_REQUIRED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"
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
    missing_commands: list[str] | None = None
    required_tests_attempted: bool = False
    required_tests_passed: bool = False
    active_repair_required: bool = False
    latest_test_failure_signature: str | None = None
    allowed_next_tools: list[str] | None = None
    rerun_after_patch: str | None = None
    target_file: str | None = None
    target_file_modified_after_repair: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "phase": self.phase.value,
            "diff_exists": self.diff_exists,
            "tests_run": self.tests_run,
            "required_tests_attempted": self.required_tests_attempted,
            "required_tests_passed": self.required_tests_passed,
            "final_allowed": self.final_allowed,
            "reason": self.reason,
            "required_action": self.required_action,
            "active_repair_required": self.active_repair_required,
            "target_file_modified_after_repair": self.target_file_modified_after_repair,
        }
        if self.missing_commands:
            payload["missing_commands"] = self.missing_commands
        if self.latest_test_failure_signature:
            payload["latest_test_failure_signature"] = self.latest_test_failure_signature
        if self.allowed_next_tools:
            payload["allowed_next_tools"] = self.allowed_next_tools
        if self.rerun_after_patch:
            payload["rerun_after_patch"] = self.rerun_after_patch
        if self.target_file:
            payload["target_file"] = self.target_file
        return payload


@dataclass(frozen=True, slots=True)
class FinalGate:
    allowed: bool
    reason: str
    detail: str
    snapshot: WorkflowSnapshot
    repair_mode: str | None = None


def workflow_snapshot(state: AgentState, git_status_output: str) -> WorkflowSnapshot:
    diff_exists = meaningful_diff_exists(git_status_output) and successful_edit_tool_called(state)
    missing_commands = missing_required_commands(state)
    tests_run = not missing_commands
    required_tests_attempted = required_commands_attempted(state)
    required_tests_passed = required_commands_passed(state)
    latest_failure_signature = latest_failed_required_command_signature(state)
    if state.inspection is None:
        return WorkflowSnapshot(
            phase=WorkflowPhase.INSPECT,
            diff_exists=diff_exists,
            tests_run=tests_run,
            required_tests_attempted=required_tests_attempted,
            required_tests_passed=required_tests_passed,
            final_allowed=False,
            reason="inspection_missing",
            required_action="inspect the repository before planning or editing",
        )
    if not diff_exists:
        return WorkflowSnapshot(
            phase=WorkflowPhase.EDIT_REQUIRED,
            diff_exists=False,
            tests_run=tests_run,
            required_tests_attempted=required_tests_attempted,
            required_tests_passed=required_tests_passed,
            final_allowed=False,
            reason="no_diff",
            required_action="modify a target file and confirm git_status shows a change",
        )
    if state.repair_mode == "must_edit":
        return WorkflowSnapshot(
            phase=WorkflowPhase.EDIT_REQUIRED,
            diff_exists=True,
            tests_run=tests_run,
            required_tests_attempted=required_tests_attempted,
            required_tests_passed=required_tests_passed,
            final_allowed=False,
            reason="repair_mode_requires_edit",
            required_action="make or confirm an edit with an allowed repair tool before final_candidate",
        )
    if required_commands(state.task_contract) and not tests_run:
        return WorkflowSnapshot(
            phase=WorkflowPhase.TEST_REQUIRED,
            diff_exists=True,
            tests_run=False,
            required_tests_attempted=required_tests_attempted,
            required_tests_passed=False,
            final_allowed=False,
            reason="required_tests_missing",
            required_action=f"run this exact verification command before final_candidate: {missing_commands[0]}",
            missing_commands=missing_commands,
            latest_test_failure_signature=latest_failure_signature,
        )
    if required_commands(state.task_contract) and required_tests_attempted and not required_tests_passed:
        next_command = next_required_command(state)
        return WorkflowSnapshot(
            phase=WorkflowPhase.REPAIR_REQUIRED,
            diff_exists=True,
            tests_run=True,
            required_tests_attempted=True,
            required_tests_passed=False,
            final_allowed=False,
            reason="required_tests_failed",
            required_action=f"repair the failing source file, then rerun the exact verification command: {next_command}",
            latest_test_failure_signature=latest_failure_signature,
            rerun_after_patch=next_command or None,
        )
    return WorkflowSnapshot(
        phase=WorkflowPhase.FINAL_READY,
        diff_exists=True,
        tests_run=tests_run,
        required_tests_attempted=required_tests_attempted,
        required_tests_passed=required_tests_passed,
        final_allowed=True,
        reason="ready",
        required_action="submit final_candidate for verifier review",
    )


def targeted_repair_workflow_allowed_tools(state: AgentState, modified: bool) -> list[str]:
    if modified:
        return ["run_command", "git_status", "git_diff"]
    phase = getattr(state, "targeted_repair_phase", None)
    if phase == "edit_forced":
        return ["edit_file", "apply_patch", "write_file", "replace_in_file"]
    return ["read_symbol", "read_file_range", "read_file", "edit_file", "apply_patch", "write_file", "replace_in_file"]


def targeted_repair_required_action(state: AgentState, target_file: str | None, modified: bool, rerun_commands: list[str]) -> str:
    target = target_file or "the target file"
    if modified:
        return f"rerun after patch: {rerun_commands[0] if rerun_commands else 'the failing command'}"
    if getattr(state, "targeted_repair_phase", None) == "edit_forced":
        return f"modify {target} now with edit_file/apply_patch/write_file; do not read more context"
    return f"inspect {target} with read_symbol/read_file_range if needed, then modify it before rerunning tests"


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
        commands = ", ".join(snapshot.missing_commands or required_commands(state.task_contract))
        return FinalGate(
            allowed=False,
            reason="final_candidate_tests_missing",
            detail=f"Final candidate rejected before verification: run the remaining required verification command(s) exactly: {commands}",
            snapshot=snapshot,
        )
    return FinalGate(
        allowed=False,
        reason=f"workflow_not_ready:{snapshot.reason}",
        detail=snapshot.required_action,
        snapshot=snapshot,
    )


def required_commands_satisfied(state: AgentState) -> bool:
    return required_commands_passed(state)


def missing_required_commands(state: AgentState) -> list[str]:
    return [command for command in required_commands(state.task_contract) if not command_was_attempted(state, command)]


def required_commands_passed(state: AgentState) -> bool:
    commands = required_commands(state.task_contract)
    return all(command_was_run(state, command) for command in commands)


def next_required_command(state: AgentState) -> str:
    commands = required_commands(state.task_contract)
    if not commands:
        return ""
    for command in commands:
        if not command_was_run(state, command):
            return command
    return commands[0]


def required_commands(task_contract: TaskContract | None) -> list[str]:
    return list(task_contract.must_run_commands) if task_contract is not None else []


def command_was_run(state: AgentState, command: str) -> bool:
    expected = normalize_command(command)
    for message in state.messages:
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = normalize_command(str(metadata.get("command") or ""))
        if observed and commands_equivalent(observed, expected):
            return True
        tool = str(message.get("tool") or "")
        if tool == "run_tests" and ("test" in expected or "pytest" in expected or "unittest" in expected):
            return True
    return False


def command_was_attempted(state: AgentState, command: str) -> bool:
    expected = normalize_command(command)
    for message in state.messages:
        if message.get("role") != "tool":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = normalize_command(str(metadata.get("command") or ""))
        if observed and commands_equivalent(observed, expected):
            return True
        tool = str(message.get("tool") or "")
        if tool == "run_tests" and ("test" in expected or "pytest" in expected or "unittest" in expected):
            return True
    return False


def required_commands_attempted(state: AgentState) -> bool:
    commands = required_commands(state.task_contract)
    if not commands:
        return False
    expected = [normalize_command(command) for command in commands]
    for message in state.messages:
        if message.get("role") != "tool":
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = normalize_command(str(metadata.get("command") or ""))
        if observed and any(commands_equivalent(observed, command) for command in expected):
            return True
    return False


def latest_failed_required_command_signature(state: AgentState) -> str | None:
    commands = [normalize_command(command) for command in required_commands(state.task_contract)]
    for message in reversed(state.messages):
        if message.get("role") != "tool" or int(message.get("exit_code") or 0) == 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        observed = normalize_command(str(metadata.get("command") or ""))
        if not observed or not any(commands_equivalent(observed, command) for command in commands):
            continue
        output = str(message.get("output") or "")
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith(("AssertionError:", "AttributeError:", "KeyError:", "FAIL:", "ERROR:")):
                return stripped[:160]
        return observed
    return None


def successful_edit_tool_called(state: AgentState) -> bool:
    return any(
        message.get("role") == "tool"
        and message.get("tool") in EDIT_TOOLS
        and int(message.get("exit_code") or 0) == 0
        for message in state.messages
    )


def normalize_command(command: str) -> str:
    cleaned = " ".join(command.strip().split())
    for suffix in (" 2>&1", " 1>&2"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def commands_equivalent(observed: str, expected: str) -> bool:
    observed = normalize_command(observed)
    expected = normalize_command(expected)
    if observed == expected:
        return True
    if is_compound_shell_command(observed):
        return compound_command_contains_success_segment(observed, expected)
    if observed.endswith(" -v") and observed[: -3] == expected:
        return True
    if expected.endswith(" -v") and expected[: -3] == observed:
        return True
    unittest = "python3 -m unittest discover -s tests"
    if observed.startswith(unittest) and expected == unittest:
        extra = observed[len(unittest) :].strip()
        return extra in {"", "-v"}
    return False


def is_compound_shell_command(command: str) -> bool:
    return any(token in command for token in (";", "&&", "||", "|", "`", "$("))


def compound_command_contains_success_segment(observed: str, expected: str) -> bool:
    if any(token in observed for token in (";", "||", "|", "`", "$(")):
        return False
    if "&&" not in observed:
        return False
    return any(commands_equivalent(segment.strip(), expected) for segment in observed.split("&&") if segment.strip())


def target_file_modified_after_repair_start(state: AgentState) -> bool:
    action = state.active_repair_action or {}
    targets = {str(path).replace("\\", "/").strip().removeprefix("/workspace/") for path in action.get("target_files") or [] if str(path)}
    rerun_commands = [str(command) for command in action.get("rerun_commands") or [] if str(command)]
    if not targets:
        return successful_edit_tool_called(state)
    for message in reversed(state.messages[state.active_repair_started_at :]):
        if message.get("role") != "tool":
            continue
        tool = str(message.get("tool") or "")
        if tool == "run_command" and int(message.get("exit_code") or 0) != 0 and rerun_commands:
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            observed = str(metadata.get("command") or "")
            if any(commands_equivalent(observed, command) for command in rerun_commands):
                return False
        if tool not in EDIT_TOOLS or int(message.get("exit_code") or 0) != 0:
            continue
        metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        path = str(metadata.get("path") or "").replace("\\", "/").strip()
        if path.startswith("/workspace/"):
            path = path[len("/workspace/") :]
        if tool == "apply_patch" or path in targets:
            return True
    return False


def meaningful_diff_exists(git_status_output: str) -> bool:
    return any(meaningful_change_path(path) for path in changed_paths_from_status(git_status_output))


def changed_paths_from_status(status: str) -> list[str]:
    paths: list[str] = []
    for raw_line in status.splitlines():
        marker, path = parse_status_line(raw_line)
        if path and (marker == "??" or marker.strip()) and meaningful_change_path(path):
            paths.append(path)
    return paths


def parse_status_line(raw_line: str) -> tuple[str, str]:
    line = strip_ansi(raw_line).rstrip()
    if not line:
        return "", ""
    if line.startswith("?? "):
        return "??", line[3:].strip()
    if len(line) >= 4 and line[2] == " ":
        marker = line[:2]
        path = line[3:].strip()
    elif len(line) >= 3 and line[1] == " ":
        marker = line[:1]
        path = line[2:].strip()
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return "", ""
        marker, path = parts[0], parts[1].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1].strip()
    return marker, path


def meaningful_change_path(path: str) -> bool:
    normalized = strip_ansi(path).strip().replace("\\", "/")
    if not normalized:
        return False
    parts = normalized.split("/")
    return not (
        normalized in {".docode_probe", ".docode_probe_api"}
        or normalized.startswith(".docode_probe")
        or "__pycache__" in parts
        or normalized.endswith((".pyc", ".pyo"))
        or normalized.startswith(".git/")
    )


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)
