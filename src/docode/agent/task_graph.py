from __future__ import annotations

from dataclasses import dataclass, field
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


class TaskGraph:
    def __init__(self, nodes: list[TaskNode] | None = None) -> None:
        self.nodes = {node.id: node for node in nodes or []}

    def add_or_update(self, node: TaskNode) -> None:
        self.nodes[node.id] = node

    def set_status(self, node_id: str, status: TaskStatus) -> None:
        self.nodes[node_id].status = status

    def ready(self) -> list[TaskNode]:
        return [node for node in self.nodes.values() if node.status == TaskStatus.PENDING and all(self.nodes[item].status == TaskStatus.DONE for item in node.dependencies if item in self.nodes)]
