from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from docode.agent.output import prompt_safe_output
from docode.agent.inspector import ProjectInspection
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob


@dataclass(slots=True)
class AgentState:
    job: CodingJob
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    tool_calls_count: int = 0
    llm_tokens_used: int = 0
    llm_cost_used: float = 0.0
    started_monotonic: float = field(default_factory=monotonic)
    consecutive_failures: int = 0
    inspection: ProjectInspection | None = None
    task_contract: TaskContract | None = None
    latest_git_status: ToolResult | None = None
    repair_mode: str | None = None
    stuck_count: int = 0
    quality_gate_passed: bool = False
    quality_gate_attempts: int = 0
    last_quality_gate: dict[str, Any] | None = None
    active_repair_action: dict[str, Any] | None = None
    active_repair_started_at: int = 0
    targeted_repair_phase: str | None = None
    targeted_repair_inspections: int = 0
    targeted_repair_edits: int = 0
    repair_action_attempts: int = 0
    failure_signatures: dict[str, int] = field(default_factory=dict)
    last_failed_command: str | None = None

    def add_observation(self, content: str) -> None:
        self.messages.append({"role": "system", "kind": "observation", "content": content})

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_calls_count += 1
        prompt_output = prompt_safe_output(result.output)
        metadata = dict(result.metadata or {})
        if prompt_output.truncated:
            metadata["prompt_output_truncated"] = True
            metadata["original_output_lines"] = prompt_output.original_lines
            metadata["original_output_bytes"] = prompt_output.original_bytes
        self.messages.append(
            {
                "role": "tool",
                "tool": result.tool,
                "exit_code": result.exit_code,
                "output": prompt_output.text,
                "truncated": result.truncated or prompt_output.truncated,
                "metadata": metadata,
            }
        )
        if result.tool in {"edit_file", "write_file", "replace_in_file", "apply_patch"} and result.ok:
            self.quality_gate_passed = False
        self.consecutive_failures = self.consecutive_failures + 1 if not result.ok else 0

    def add_feedback(self, content: str) -> None:
        self.messages.append({"role": "system", "kind": "feedback", "content": content})
        self.consecutive_failures += 1
