from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from docode.agent.output import prompt_safe_output
from docode.agent.inspector import ProjectInspection
from docode.agent.task_contract import TaskContract
from docode.agent.artifact_contract import ArtifactSemanticContract
from docode.agent.failure_taxonomy import TerminalResult
from docode.agent.finalization_controller import FinalizationController
from docode.agent.no_progress import NoProgressTracker
from docode.agent.outcome import FinalizationBlocker, StepOutcome
from docode.agent.profiles import TaskProfile
from docode.agent.repair_coordinator import RepairCoordinator
from docode.agent.task_graph import TaskGraph
from docode.agent.task_graph import TaskStatus
from docode.agent.verification_scheduler import VerificationScheduler
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
    profile: TaskProfile | None = None
    artifact_contract: ArtifactSemanticContract | None = None
    verification_scheduler: VerificationScheduler | None = None
    repair_coordinator: RepairCoordinator | None = None
    repository_context: Any | None = None
    task_graph: TaskGraph | None = None
    finalization_controller: FinalizationController | None = None
    terminal_result: TerminalResult | None = None
    edit_epoch: int = 0
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
    source_inspection_auto_attempted_urls: set[str] = field(default_factory=set)
    source_inspection_excerpt_presented: bool = False
    terminal_repair_reason: str | None = None

    # structured outcome & no-progress (Runtime V2)
    active_blocker: FinalizationBlocker | None = None
    last_outcome: StepOutcome | None = None
    recent_outcomes: list[StepOutcome] = field(default_factory=list)
    no_progress_tracker: NoProgressTracker = field(default_factory=NoProgressTracker)
    terminal_no_progress_reason: str | None = None

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
            self.edit_epoch += 1
            if self.verification_scheduler is not None:
                self.verification_scheduler.mark_edit()
            if self.repair_coordinator is not None and self.repair_mode == "targeted_repair":
                self.targeted_repair_phase = self.repair_coordinator.record_edit().value
            if self.task_graph is not None:
                paths = [str(metadata.get("path") or "").replace("\\", "/")]
                paths.extend(str(item).replace("\\", "/") for item in metadata.get("paths", []) if item)
                paths = [path for path in paths if path]
                implement = self.task_graph.nodes.get("implement")
                targets = [item.replace("\\", "/") for item in (implement.target_files if implement else [])]
                relevant = [path for path in paths if not targets or any(path == item or path.endswith("/" + item) for item in targets)]
                if implement is not None and relevant:
                    self.task_graph.set_status("implement", TaskStatus.DONE, reason="task-relevant edit succeeded", evidence_refs=[f"edit:{self.edit_epoch}:{path}" for path in relevant])
                if "verify" in self.task_graph.nodes:
                    self.task_graph.set_status("verify", TaskStatus.PENDING)
                if "review" in self.task_graph.nodes:
                    self.task_graph.set_status("review", TaskStatus.PENDING)
        elif result.tool in {"read_file", "read_file_range", "read_symbol", "search"} and result.ok and self.task_graph is not None:
            path = str(metadata.get("path") or metadata.get("file") or "").replace("\\", "/")
            understand = self.task_graph.nodes.get("understand")
            targets = [item.replace("\\", "/") for item in (understand.target_files if understand else [])]
            if understand is not None and path and targets and any(path == item or path.endswith("/" + item) for item in targets):
                self.task_graph.set_status("understand", TaskStatus.DONE, reason="relevant repository target inspected", evidence_refs=[f"read:{path}"])
        if result.tool == "run_command" and self.verification_scheduler is not None:
            command = str(metadata.get("command") or "")
            if command:
                self.verification_scheduler.record(command, result.ok)
                if self.repair_coordinator is not None and self.repair_mode == "targeted_repair":
                    node = next((item for item in self.verification_scheduler.commands if item.command == command), None)
                    if node is not None and node.kind == "producer":
                        self.targeted_repair_phase = self.repair_coordinator.record_producer(result.ok).value
                    elif node is not None and node.kind == "validator":
                        self.targeted_repair_phase = self.repair_coordinator.record_validator(result.ok).value
                if self.task_graph is not None and "verify" in self.task_graph.nodes:
                    complete = self.verification_scheduler.next_command() is None
                    status = TaskStatus.DONE if complete else (TaskStatus.PENDING if result.ok else TaskStatus.BLOCKED)
                    refs = [f"command:{self.edit_epoch}:{command}"] if status == TaskStatus.DONE else []
                    self.task_graph.set_status("verify", status, reason="scheduler command result", evidence_refs=refs)
        self.consecutive_failures = self.consecutive_failures + 1 if not result.ok else 0

    def add_feedback(self, content: str) -> None:
        self.messages.append({"role": "system", "kind": "feedback", "content": content})
        if feedback_counts_as_consecutive_failure(content):
            self.consecutive_failures += 1


def feedback_counts_as_consecutive_failure(content: str) -> bool:
    """Return whether feedback should consume the hard consecutive-failure budget.

    Rejections that are deterministic workflow control-flow guidance should not be
    counted as hard failures. Tool execution failures still count through
    add_tool_result, and broader stop policies continue to cap iterations, tool
    calls, runtime, and token usage.
    """

    text = (content or "").lower()
    targeted_repair_markers = (
        "active repair:",
        "active targeted repair",
        "repair_mode=targeted_repair",
        "targeted_repair_wrong_action",
        "targeted_repair_exploration_limit",
        "targeted_repair_tool_forbidden",
        "requires modifying",
        "do not run tests again until",
        "rerun after patch",
    )
    if any(marker in text for marker in targeted_repair_markers) and (
        "targeted" in text or "active repair" in text or "before running commands" in text
    ):
        return False

    required_command_markers = (
        "test_required_exact_command_control",
        "test_required_tool_forbidden",
        "run this exact verification command before final_candidate",
        "run_command now with exactly",
        "unavailable_tool_requested",
    )
    if any(marker in text for marker in required_command_markers):
        return False

    return True
