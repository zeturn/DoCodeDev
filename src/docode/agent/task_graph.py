from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class TaskNode:
    id: str
    goal: str
    target_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    acceptance_criteria: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    failure_reason: str | None = None
    attempt_count: int = 0


@dataclass(frozen=True, slots=True)
class TaskNodeTransition:
    node_id: str
    previous_status: TaskStatus
    new_status: TaskStatus
    reason: str
    evidence_refs: tuple[str, ...] = ()


class TaskGraph:
    def __init__(self, nodes: list[TaskNode] | None = None) -> None:
        self.nodes = {node.id: node for node in nodes or []}
        self.transitions: list[TaskNodeTransition] = []

    def add_or_update(self, node: TaskNode) -> None:
        self.nodes[node.id] = node

    def set_status(self, node_id: str, status: TaskStatus, *, reason: str = "", evidence_refs: list[str] | None = None) -> None:
        node = self.nodes[node_id]
        refs = [ref for ref in evidence_refs or [] if ref]
        if status == TaskStatus.DONE and not refs and not node.evidence_refs:
            raise ValueError(f"task node {node_id} cannot complete without evidence")
        previous = node.status
        now = datetime.now(timezone.utc).isoformat()
        if previous == TaskStatus.PENDING and status != TaskStatus.PENDING:
            node.started_at = node.started_at or now
        node.attempt_count += 1
        node.status = status
        node.evidence_refs.extend(ref for ref in refs if ref not in node.evidence_refs)
        node.failure_reason = reason if status in {TaskStatus.BLOCKED, TaskStatus.FAILED} else None
        if status == TaskStatus.DONE:
            node.completed_at = now
        elif previous == TaskStatus.DONE:
            node.completed_at = None
        if previous != status or refs:
            self.transitions.append(TaskNodeTransition(node_id, previous, status, reason, tuple(refs)))

    def complete(self) -> bool:
        return bool(self.nodes) and all(node.status == TaskStatus.DONE and node.evidence_refs for node in self.nodes.values())

    def ready(self) -> list[TaskNode]:
        return [node for node in self.nodes.values() if node.status == TaskStatus.PENDING and all(self.nodes[item].status == TaskStatus.DONE for item in node.dependencies if item in self.nodes)]
