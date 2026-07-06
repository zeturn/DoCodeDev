from __future__ import annotations

import re
from dataclasses import dataclass

from docode.agent.state import AgentState


EDIT_TOOLS = {"edit_file", "write_file", "apply_patch", "replace_in_file"}
REPAIR_ALLOWED_TOOLS = {"read_file", "edit_file", "write_file", "apply_patch", "replace_in_file", "git_status", "git_diff"}


@dataclass(frozen=True, slots=True)
class StuckSignal:
    stuck: bool
    reason: str = ""
    repair_instruction: str = ""


class StuckDetector:
    def evaluate(self, *, state: AgentState, latest_git_status: str) -> StuckSignal:
        if state.iteration >= 4 and git_status_clean(latest_git_status) and not edit_tool_called(state):
            return StuckSignal(
                stuck=True,
                reason="no_diff_after_multiple_iterations",
                repair_instruction=(
                    "You have not produced any file changes. The next action must be an edit_file, "
                    "write_file, or apply_patch call that changes the target file. Do not call run_command "
                    "or final_candidate until git_status shows a modified file."
                ),
            )
        return StuckSignal(stuck=False)


def git_status_clean(output: str | None) -> bool:
    for raw_line in (output or "").splitlines():
        line = strip_ansi(raw_line).rstrip()
        if len(line) < 4:
            continue
        marker = line[:2]
        path = line[3:].strip().replace("\\", "/")
        if not path or not (marker == "??" or marker.strip()):
            continue
        if path in {".docode_probe", ".docode_probe_api"} or path.startswith(".docode_probe") or path.startswith(".git/"):
            continue
        return False
    return True


def edit_tool_called(state: AgentState) -> bool:
    return any(message.get("role") == "tool" and message.get("tool") in EDIT_TOOLS for message in state.messages)


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)
