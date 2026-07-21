from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    path: str
    is_file: bool = True


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    line: int | None
    text: str


class WorkspaceReader(Protocol):
    async def list_files(self, path: str = ".") -> list[WorkspaceEntry]: ...
    async def read_file(self, path: str, max_bytes: int = 256_000) -> str: ...
    async def search(self, query: str, path: str = ".") -> list[SearchHit]: ...


class DoBoxWorkspaceReader:
    def __init__(self, tools: object) -> None:
        self.tools = tools

    async def list_files(self, path: str = ".") -> list[WorkspaceEntry]:
        result = await self.tools.list_files(path)
        if not result.ok:
            return []
        return [WorkspaceEntry(line.strip().replace("\\", "/")) for line in result.output.splitlines() if line.strip()]

    async def read_file(self, path: str, max_bytes: int = 256_000) -> str:
        result = await self.tools.read_file(path)
        if not result.ok:
            return ""
        return str(result.output or "").encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")

    async def search(self, query: str, path: str = ".") -> list[SearchHit]:
        result = await self.tools.search(query, path)
        if not result.ok:
            return []
        hits: list[SearchHit] = []
        for line in result.output.splitlines():
            parts = line.split(":", 2)
            number = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
            hits.append(SearchHit(parts[0], number, parts[-1]))
        return hits


class LocalWorkspaceReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    async def list_files(self, path: str = ".") -> list[WorkspaceEntry]:
        base = (self.root / path).resolve()
        return [WorkspaceEntry(item.relative_to(self.root).as_posix()) for item in base.rglob("*") if item.is_file()]

    async def read_file(self, path: str, max_bytes: int = 256_000) -> str:
        try:
            return (self.root / path).read_bytes()[:max_bytes].decode("utf-8")
        except (OSError, UnicodeError):
            return ""

    async def search(self, query: str, path: str = ".") -> list[SearchHit]:
        hits: list[SearchHit] = []
        for entry in await self.list_files(path):
            for number, line in enumerate((await self.read_file(entry.path)).splitlines(), 1):
                if query in line:
                    hits.append(SearchHit(entry.path, number, line))
        return hits
