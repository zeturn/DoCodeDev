from __future__ import annotations

import inspect
import posixpath
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .client import DoBoxClient
from .types import FileResult, ToolResult


WORKSPACE_ROOT = "/workspace"
ToolCallable = Callable[..., Awaitable[ToolResult]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolCallable

    def input_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        signature = inspect.signature(self.handler)
        for name, value in self.parameters.items():
            properties[name] = _parameter_schema(value)
            parameter = signature.parameters.get(name)
            if parameter is not None and parameter.default is inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


@dataclass(frozen=True, slots=True)
class WeavCompatibleToolSpec:
    """Fallback spec with the shape expected by weav-style tool registries."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[dict[str, Any]]]


class LocalToolRegistry:
    """Small ToolRegistry-compatible fallback for local development and tests."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, spec: Any) -> None:
        self._tools[str(spec.name)] = spec

    def list(self) -> list[Any]:
        return list(self._tools.values())

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = self._tools[name]
        return await _call_weav_handler(spec.handler, args or {})


class DoBoxTools:
    def __init__(
        self,
        client: DoBoxClient,
        project_id: str,
        *,
        agent_session_id: str | None = None,
        command_timeout_seconds: int = 120,
        output_limit_bytes: int = 1_000_000,
    ) -> None:
        self.client = client
        self.project_id = project_id
        self.agent_session_id = agent_session_id
        self.command_timeout_seconds = command_timeout_seconds
        self.output_limit_bytes = output_limit_bytes

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("run_command", "Run a shell command inside the project sandbox.", {"command": "string", "cwd": "string"}, self.run_command),
            ToolDefinition("read_file", "Read a file under /workspace.", {"path": "string"}, self.read_file),
            ToolDefinition("write_file", "Write a file under /workspace.", {"path": "string", "content": "string"}, self.write_file),
            ToolDefinition("list_files", "List files under a workspace path.", {"path": "string"}, self.list_files),
            ToolDefinition("search", "Search project files for text.", {"query": "string", "path": "string"}, self.search),
            ToolDefinition("git_status", "Return git porcelain status.", {}, self.git_status),
            ToolDefinition("git_diff", "Return git diff.", {}, self.git_diff),
            ToolDefinition("git_commit", "Commit all workspace changes with a message.", {"message": "string"}, self.git_commit),
            ToolDefinition("run_tests", "Run detected tests for the project.", {}, self.run_tests),
            ToolDefinition("run_build", "Run a detected build command for the project.", {}, self.run_build),
            ToolDefinition("run_lint", "Run a detected lint command for the project.", {}, self.run_lint),
            ToolDefinition("preview", "Create or fetch a preview URL for a sandbox service port.", {"port": "integer"}, self.preview),
            ToolDefinition("logs", "Read recent sandbox logs for debugging.", {"tail": "integer"}, self.logs),
        ]

    async def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        for definition in self.definitions():
            if definition.name == tool_name:
                return await definition.handler(**args)
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        cwd_error = workspace_path_error(cwd, label="cwd")
        if cwd_error:
            return rejected_tool_result("run_command", cwd_error, {"command": command, "cwd": cwd})
        result = await self.client.run_command(
            self.project_id,
            ["bash", "-lc", command],
            cwd=cwd,
            timeout_sec=self.command_timeout_seconds,
            output_limit=self.output_limit_bytes,
            agent_session_id=self.agent_session_id,
        )
        return self._compress("run_command", result.output, result.exit_code, {"command": command, "cwd": cwd}, truncated=result.truncated)

    async def read_file(self, path: str) -> ToolResult:
        path_error = workspace_path_error(path)
        if path_error:
            return rejected_tool_result("read_file", path_error, {"path": path})
        file_result = await self.client.read_file(self.project_id, path, agent_session_id=self.agent_session_id)
        if isinstance(file_result, FileResult):
            metadata: dict[str, Any] = {"path": path}
            if file_result.path is not None:
                metadata["resolved_path"] = file_result.path
            if file_result.file_name is not None:
                metadata["file_name"] = file_result.file_name
            if file_result.bytes_read is not None:
                metadata["bytes"] = file_result.bytes_read
            return self._compress("read_file", file_result.content, 0, metadata, truncated=file_result.truncated)
        return self._compress("read_file", str(file_result), 0, {"path": path})

    async def write_file(self, path: str, content: str) -> ToolResult:
        path_error = workspace_path_error(path)
        if path_error:
            return rejected_tool_result("write_file", path_error, {"path": path})
        await self.client.write_file(self.project_id, path, content, agent_session_id=self.agent_session_id)
        return ToolResult(tool="write_file", output=f"wrote {len(content.encode('utf-8'))} bytes", metadata={"path": path})

    async def list_files(self, path: str = ".") -> ToolResult:
        path_error = workspace_path_error(path)
        if path_error:
            return rejected_tool_result("list_files", path_error, {"path": path})
        result = await self.client.list_files(self.project_id, path, agent_session_id=self.agent_session_id)
        return self._compress("list_files", result.output, result.exit_code, {"path": path}, truncated=result.truncated)

    async def search(self, query: str, path: str = ".") -> ToolResult:
        path_error = workspace_path_error(path)
        if path_error:
            return rejected_tool_result("search", path_error, {"query": query, "path": path})
        result = await self.client.search(self.project_id, query, path, agent_session_id=self.agent_session_id)
        return self._compress("search", result.output, result.exit_code, {"query": query, "path": path}, truncated=result.truncated)

    async def git_status(self) -> ToolResult:
        result = await self.client.git_status(self.project_id, agent_session_id=self.agent_session_id)
        return self._compress("git_status", result.output, result.exit_code, truncated=result.truncated)

    async def git_diff(self) -> ToolResult:
        if hasattr(self.client, "git_diff_result"):
            result = await self.client.git_diff_result(self.project_id, agent_session_id=self.agent_session_id)
            return self._compress("git_diff", result.output, result.exit_code, truncated=result.truncated)
        diff = await self.client.git_diff(self.project_id, agent_session_id=self.agent_session_id)
        return self._compress("git_diff", diff, 0)

    async def git_commit(self, message: str) -> ToolResult:
        result = await self.client.git_commit(self.project_id, message, agent_session_id=self.agent_session_id)
        return self._compress("git_commit", result.output, result.exit_code, {"message": message}, truncated=result.truncated)

    async def preview(self, port: int) -> ToolResult:
        data = await self.client.preview(self.project_id, port, agent_session_id=self.agent_session_id)
        output = preview_output(data)
        return self._compress("preview", output, 0, {"port": port})

    async def logs(self, tail: int = 200) -> ToolResult:
        logs = await self.client.logs(self.project_id, str(tail), agent_session_id=self.agent_session_id)
        return self._compress("logs", logs, 0, {"tail": tail})

    async def run_tests(self) -> ToolResult:
        command = await self.detect_test_command()
        if command is None:
            return ToolResult(tool="run_tests", output="no test command detected", exit_code=0, metadata={"detected": False})
        result = await self.run_command(command, "/workspace")
        return ToolResult(tool="run_tests", output=result.output, exit_code=result.exit_code, metadata={"command": command, "detected": True}, truncated=result.truncated)

    async def run_build(self) -> ToolResult:
        command = await self.detect_build_command()
        if command is None:
            return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})
        result = await self.run_command(command, "/workspace")
        return ToolResult(tool="run_build", output=result.output, exit_code=result.exit_code, metadata={"command": command, "detected": True}, truncated=result.truncated)

    async def run_lint(self) -> ToolResult:
        command = await self.detect_lint_command()
        if command is None:
            return ToolResult(tool="run_lint", output="no lint command detected", exit_code=0, metadata={"detected": False})
        result = await self.run_command(command, "/workspace")
        return ToolResult(tool="run_lint", output=result.output, exit_code=result.exit_code, metadata={"command": command, "detected": True}, truncated=result.truncated)

    async def detect_test_command(self) -> str | None:
        checks = [
            (package_script_exists_command("test"), "npm test"),
            ("test -f pyproject.toml || test -f pytest.ini", "pytest"),
            ("test -f go.mod", "go test ./..."),
            ("test -f Cargo.toml", "cargo test"),
        ]
        return await self._detect_command(checks)

    async def detect_build_command(self) -> str | None:
        checks = [
            (package_script_exists_command("build"), "npm run build"),
            ("test -f go.mod", "go build ./..."),
            ("test -f Cargo.toml", "cargo build"),
        ]
        return await self._detect_command(checks)

    async def detect_lint_command(self) -> str | None:
        checks = [
            (package_script_exists_command("lint"), "npm run lint"),
            ("test -f pyproject.toml && command -v ruff >/dev/null 2>&1", "ruff check ."),
            ("test -f Cargo.toml && command -v cargo-clippy >/dev/null 2>&1", "cargo clippy --all-targets -- -D warnings"),
        ]
        return await self._detect_command(checks)

    async def _detect_command(self, checks: list[tuple[str, str]]) -> str | None:
        for exists_command, test_command in checks:
            result = await self.client.run_command(
                self.project_id,
                ["bash", "-lc", exists_command],
                cwd="/workspace",
                timeout_sec=10,
                agent_session_id=self.agent_session_id,
            )
            if result.exit_code == 0:
                return test_command
        return None

    def _compress(
        self,
        tool: str,
        output: str,
        exit_code: int,
        metadata: dict[str, Any] | None = None,
        *,
        truncated: bool = False,
    ) -> ToolResult:
        encoded = output.encode("utf-8")
        if len(encoded) <= self.output_limit_bytes:
            return ToolResult(tool=tool, output=output, exit_code=exit_code, metadata=metadata, truncated=truncated)
        clipped = encoded[: self.output_limit_bytes].decode("utf-8", errors="replace")
        return ToolResult(tool=tool, output=clipped, exit_code=exit_code, metadata=metadata, truncated=True)


def register_dobox_tools(registry: Any, tools: DoBoxTools) -> None:
    """Register DoBox tools into a Weav ToolRegistry-like object.

    This intentionally uses duck typing because the exact weav-core registry
    implementation may vary by installed version.
    """

    for definition in tools.definitions():
        spec = build_weav_tool_spec(definition)
        if hasattr(registry, "register"):
            _register_tool(registry.register, spec)
        elif hasattr(registry, "add"):
            _register_tool(registry.add, spec)


def build_dobox_tool_registry(tools: DoBoxTools, registry: Any | None = None) -> Any:
    """Build a ToolRegistry and register all DoBox project sandbox tools."""

    if registry is None:
        registry = try_create_weav_tool_registry() or LocalToolRegistry()
    register_dobox_tools(registry, tools)
    return registry


def build_weav_tool_spec(definition: ToolDefinition) -> Any:
    async def handler(args: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        merged_args = dict(args or {})
        merged_args.update(kwargs)
        result = await definition.handler(**merged_args)
        return tool_result_to_weav_output(result)

    schema = definition.input_schema()
    spec_cls = try_import_weav_tool_spec()
    if spec_cls is not None:
        for kwargs in (
            {"name": definition.name, "description": definition.description, "input_schema": schema, "handler": handler},
            {"name": definition.name, "description": definition.description, "parameters": schema, "handler": handler},
            {"name": definition.name, "description": definition.description, "schema": schema, "handler": handler},
        ):
            try:
                return spec_cls(**kwargs)
            except TypeError:
                continue
        try:
            return spec_cls(definition.name, definition.description, schema, handler)
        except TypeError:
            pass
    return WeavCompatibleToolSpec(definition.name, definition.description, schema, handler)


def tool_result_to_weav_output(result: ToolResult) -> dict[str, Any]:
    return {
        "tool": result.tool,
        "ok": result.ok,
        "exit_code": result.exit_code,
        "content": result.output,
        "truncated": result.truncated,
        "metadata": result.metadata or {},
    }


def workspace_path_error(path: str, *, label: str = "path") -> str | None:
    if not isinstance(path, str):
        return f"{label} must be a string"
    if "\x00" in path:
        return f"{label} must not contain NUL bytes"
    if not path.strip():
        return f"{label} must not be empty"

    normalized = posixpath.normpath(path.strip())
    if normalized in {".", WORKSPACE_ROOT}:
        return None
    if normalized.startswith("/"):
        if normalized.startswith(WORKSPACE_ROOT + "/"):
            return None
        return f"{label} must stay under {WORKSPACE_ROOT}"
    if normalized == ".." or normalized.startswith("../"):
        return f"{label} must stay under {WORKSPACE_ROOT}"
    return None


def rejected_tool_result(tool: str, reason: str, metadata: dict[str, Any]) -> ToolResult:
    return ToolResult(tool=tool, output=f"rejected: {reason}", exit_code=2, metadata={**metadata, "rejected": True, "reason": reason})


def try_create_weav_tool_registry() -> Any | None:
    try:
        from weav_ai_core import ToolRegistry
    except Exception:
        return None
    try:
        return ToolRegistry()
    except Exception:
        return None


def try_import_weav_tool_spec() -> Any | None:
    try:
        from weav_ai_core import ToolSpec
    except Exception:
        return None
    return ToolSpec


def _register_tool(register: Callable[..., Any], spec: Any) -> None:
    try:
        register(spec)
        return
    except TypeError:
        pass
    try:
        register(spec.name, spec.handler)
        return
    except TypeError:
        pass
    register(spec.name, spec.description, spec.input_schema, spec.handler)


async def _call_weav_handler(handler: Callable[..., Awaitable[dict[str, Any]]], args: dict[str, Any]) -> dict[str, Any]:
    try:
        return await handler(args)
    except TypeError:
        return await handler(**args)


def _parameter_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in {"string", str}:
        return {"type": "string"}
    if value in {"integer", int}:
        return {"type": "integer"}
    if value in {"number", float}:
        return {"type": "number"}
    if value in {"boolean", bool}:
        return {"type": "boolean"}
    return {"type": "string", "description": str(value)}


def package_script_exists_command(script: str) -> str:
    return (
        "test -f package.json && "
        "node -e \"const p=require('./package.json'); "
        f"process.exit(p.scripts && p.scripts['{script}'] ? 0 : 1)\""
    )


def preview_output(data: dict[str, Any]) -> str:
	for key in ("url", "preview_url", "proxy_url"):
		value = data.get(key)
		if value:
			return str(value)
	if data.get("status") == "preview_descriptor":
		port = data.get("port")
		message = data.get("message") or "Preview descriptor returned without a proxy URL."
		return f"preview port {port}: {message}"
	return str(data)
