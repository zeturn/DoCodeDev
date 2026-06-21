from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandResult:
    output: str
    exit_code: int
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True, slots=True)
class FileResult:
    content: str
    path: str | None = None
    file_name: str | None = None
    bytes_read: int | None = None
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool: str
    output: str
    exit_code: int = 0
    metadata: dict[str, Any] | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True, slots=True)
class ProjectSandbox:
    project_id: str
    sandbox_id: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class AgentSession:
    session_id: str
    raw: dict[str, Any] | None = None
