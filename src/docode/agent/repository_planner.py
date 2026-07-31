from __future__ import annotations

from .repository_index import RepositoryIndex
from .task_graph import TaskGraph, TaskNode


class RepositoryPlanner:
    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    def initial_graph(self, instruction: str, verification: list[str] | None = None) -> TaskGraph:
        ranked = [path for path, score in self.index.rank_files(instruction) if score > 0][:8]
        tests = [path for path in ranked if "test" in path.lower()]
        implementation = [path for path in ranked if path not in tests]
        nodes = [TaskNode("understand", "Identify interfaces, definitions, callers, and tests", ranked)]
        if implementation:
            nodes.append(TaskNode("implement", "Apply the requested implementation changes", implementation, ["understand"]))
        nodes.append(TaskNode("verify", "Run explicit verification and review changed-file impact", tests, ["implement"] if implementation else ["understand"], verification or []))
        return TaskGraph(nodes)
