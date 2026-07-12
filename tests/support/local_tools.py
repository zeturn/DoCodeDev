"""Shared deterministic/local tool implementations for holdout and diagnostic suites.

This module centralises ``DiagnosticLocalTools`` and the path/command helpers that were
previously defined inline in ``tests/test_real_llm_diagnostic_suite.py``. Keeping them in
``tests.support`` lets both the holdout harness and the diagnostic suite import the same
implementation instead of importing across ``test_*.py`` modules.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from docode.dobox.tools import ToolDefinition
from docode.dobox.types import ToolResult

from tests.support.path_utils import normalize_path


class DiagnosticLocalTools:
    def __init__(self, workspace: Path, *, test_command: str = "python -m unittest discover -s tests") -> None:
        self.workspace = workspace.resolve()
        self.test_command = test_command
        self.initial_files = self.snapshot_files()
        self.command_results: list[ToolResult] = []

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("run_command", "Run a shell command in the local fixture workspace.", {"command": "string", "cwd": "string"}, self.run_command),
            ToolDefinition("read_file", "Read a file from the fixture workspace.", {"path": "string"}, self.read_file),
            ToolDefinition("read_file_range", "Read a 1-based inclusive line range from a file.", {"path": "string", "start_line": "integer", "end_line": "integer"}, self.read_file_range),
            ToolDefinition("write_file", "Write a file in the fixture workspace.", {"path": "string", "content": "string"}, self.write_file),
            ToolDefinition("edit_file", "Replace exact text in an existing file.", {"path": "string", "old_text": "string", "new_text": "string", "expected_occurrences": "integer"}, self.edit_file),
            ToolDefinition("replace_in_file", "Replace exact text using find/replace arguments.", {"path": "string", "find": "string", "replace": "string", "expected_occurrences": "integer"}, self.replace_in_file),
            ToolDefinition("apply_patch", "Apply a unified diff patch in the fixture workspace.", {"patch": "string"}, self.apply_patch),
            ToolDefinition("list_files", "List files under a workspace path.", {"path": "string"}, self.list_files),
            ToolDefinition("search", "Search fixture files for text.", {"query": "string", "path": "string"}, self.search),
            ToolDefinition("git_status", "Return git porcelain-like status from fixture snapshots.", {}, self.git_status),
            ToolDefinition("git_diff", "Return a unified diff from fixture snapshots.", {}, self.git_diff),
            ToolDefinition("run_tests", "Run the detected unittest command.", {}, self.run_tests),
            ToolDefinition("run_build", "Report that no build command is detected.", {}, self.run_build),
            ToolDefinition("run_lint", "Report that no lint command is detected.", {}, self.run_lint),
        ]

    def set_detected_command(self, name: str, command: str | None) -> None:
        if name == "test" and command:
            self.test_command = command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        for definition in self.definitions():
            if definition.name != tool_name:
                continue
            if tool_name in {"git_status", "git_diff", "run_tests", "run_build", "run_lint"}:
                return await definition.handler()
            allowed = {key: value for key, value in args.items() if key in definition.parameters}
            return await definition.handler(**allowed)
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    async def list_files(self, path: str = ".") -> ToolResult:
        base = safe_workspace_path(self.workspace, normalize_path(path) or ".")
        if not base.exists():
            return ToolResult(tool="list_files", output=f"{path} not found", exit_code=1, metadata={"path": normalize_path(path)})
        if base.is_file():
            paths = [base.relative_to(self.workspace).as_posix()]
        else:
            paths = sorted(file.relative_to(self.workspace).as_posix() for file in base.rglob("*") if file.is_file() and "__pycache__" not in file.parts)
        return ToolResult(tool="list_files", output="\n".join(paths) + ("\n" if paths else ""))

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists() or not target.is_file():
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=target.read_text(encoding="utf-8"), metadata={"path": normalized})

    async def read_file_range(self, path: str, start_line: int = 1, end_line: int = 120) -> ToolResult:
        result = await self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(tool="read_file_range", output=result.output, exit_code=result.exit_code, metadata=result.metadata)
        lines = result.output.splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), max(start, int(end_line)))
        output = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start - 1 : end], start=start))
        return ToolResult(tool="read_file_range", output=output, metadata=result.metadata)

    async def write_file(self, path: str, content: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(tool="write_file", output=f"wrote {normalized}", metadata={"path": normalized})

    async def edit_file(self, path: str, old_text: str, new_text: str, expected_occurrences: int | None = None) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists():
            return ToolResult(tool="edit_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        current = target.read_text(encoding="utf-8")
        occurrences = current.count(old_text)
        if occurrences == 0:
            return ToolResult(tool="edit_file", output="old_text not found", exit_code=1, metadata={"path": normalized})
        if expected_occurrences is not None and int(expected_occurrences) > 0 and occurrences != int(expected_occurrences):
            return ToolResult(tool="edit_file", output=f"expected {expected_occurrences} occurrences, found {occurrences}", exit_code=1, metadata={"path": normalized})
        target.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
        return ToolResult(tool="edit_file", output=f"edited {normalized}", metadata={"path": normalized})

    async def replace_in_file(self, path: str, find: str, replace: str, expected_occurrences: int | None = None) -> ToolResult:
        result = await self.edit_file(path, find, replace, expected_occurrences)
        return ToolResult(tool="replace_in_file", output=result.output, exit_code=result.exit_code, metadata=result.metadata, truncated=result.truncated)

    async def apply_patch(self, patch: str) -> ToolResult:
        completed = subprocess.run(
            ["git", "apply", "--no-index", "--whitespace=nowarn", "-"],
            input=patch,
            cwd=self.workspace,
            text=True,
            capture_output=True,
            check=False,
        )
        paths = []
        for match in re.finditer(r"^\+\+\+\s+(?:b/)?(.+)$", patch, flags=re.MULTILINE):
            path = match.group(1).strip()
            if path != "/dev/null" and path not in paths:
                paths.append(path)
        return ToolResult(
            tool="apply_patch",
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            metadata={"paths": paths, "patch_bytes": len(patch.encode("utf-8")), "stderr": completed.stderr},
        )

    async def search(self, query: str, path: str = ".") -> ToolResult:
        root = safe_workspace_path(self.workspace, normalize_path(path) or ".")
        matches: list[str] = []
        files = [root] if root.is_file() else [file for file in root.rglob("*") if file.is_file()]
        for file in files:
            if "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(f"{file.relative_to(self.workspace).as_posix()}:{idx}:{line}")
        return ToolResult(tool="search", output="\n".join(matches[:200]))

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        workspace_cwd = self.workspace if cwd in {"", ".", "/workspace"} else safe_workspace_path(self.workspace, normalize_path(cwd))
        executable_command = python_portable_command(command)
        completed = subprocess.run(
            executable_command,
            cwd=workspace_cwd,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=45,
        )
        result = ToolResult(
            tool="run_command",
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            metadata={"command": command, "executed_command": executable_command},
        )
        self.command_results.append(result)
        return result

    async def git_status(self) -> ToolResult:
        changed = self.changed_files()
        return ToolResult(tool="git_status", output="".join(f" M {path}\n" for path in changed))

    async def git_diff(self) -> ToolResult:
        parts: list[str] = []
        current = self.snapshot_files()
        for path in sorted(set(self.initial_files) | set(current)):
            before = self.initial_files.get(path, "").splitlines(keepends=True)
            after = current.get(path, "").splitlines(keepends=True)
            if before == after:
                continue
            parts.append(f"diff --git a/{path} b/{path}\n")
            parts.extend(difflib.unified_diff(before, after, fromfile=f"a/{path}", tofile=f"b/{path}"))
        return ToolResult(tool="git_diff", output="".join(parts))

    async def run_tests(self) -> ToolResult:
        result = await self.run_command(self.test_command)
        return ToolResult(tool="run_tests", output=result.output, exit_code=result.exit_code, metadata={"detected": True, "command": self.test_command})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return self.test_command

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None

    def changed_files(self) -> list[str]:
        current = self.snapshot_files()
        return [path for path in sorted(set(self.initial_files) | set(current)) if self.initial_files.get(path) != current.get(path)]

    def snapshot_files(self) -> dict[str, str]:
        return {
            file.relative_to(self.workspace).as_posix(): file.read_text(encoding="utf-8", errors="surrogateescape")
            for file in self.workspace.rglob("*")
            if file.is_file() and "__pycache__" not in file.parts and not file.name.endswith(".pyc")
        }


def safe_workspace_path(workspace: Path, path: str) -> Path:
    normalized = normalize_path(path).lstrip("/")
    target = (workspace / normalized).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes workspace: {path}")
    return target


def python_portable_command(command: str) -> str:
    """Rewrite ``python``/``python3`` to the running interpreter on every platform.

    Unlike the partial ``tests.support.path_utils.python_portable_command`` (which only
    rewrites ``python3`` -> ``python`` on Windows), this helper always resolves the
    command to ``sys.executable`` so the deterministic local tools run the same
    interpreter that started the test process on Linux, macOS and Windows.
    """
    stripped = command.strip()
    for executable in ("python3", "python"):
        if stripped == executable or stripped.startswith(executable + " "):
            return f'"{sys.executable}"{stripped[len(executable):]}'
    return command
