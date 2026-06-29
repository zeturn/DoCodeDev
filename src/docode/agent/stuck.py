from __future__ import annotations

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
        if state.iteration >= 6 and git_status_clean(latest_git_status) and not edit_tool_called(state):
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
    return not (output or "").strip()


def edit_tool_called(state: AgentState) -> bool:
    return any(message.get("role") == "tool" and message.get("tool") in EDIT_TOOLS for message in state.messages)
