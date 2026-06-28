from __future__ import annotations

from typing import Any

from docode.dobox.tools import (
    LocalToolRegistry,
    ToolDefinition,
    build_weav_tool_spec,
    register_dobox_tools,
    tool_result_to_weav_output,
    try_create_weav_tool_registry,
)
from docode.dobox.types import ToolResult


def docode_tool_to_weav_spec(definition: ToolDefinition) -> Any:
    return build_weav_tool_spec(definition)


def docode_result_to_weav_output(result: ToolResult) -> dict[str, Any]:
    return tool_result_to_weav_output(result)


def weav_output_to_docode_result(output: Any, *, tool_name: str = "weav_tool") -> ToolResult:
    if isinstance(output, ToolResult):
        return output
    if isinstance(output, dict):
        tool = str(output.get("tool") or output.get("name") or tool_name)
        content = output.get("content", output.get("output", output.get("text", "")))
        exit_code = int(output.get("exit_code", 0 if output.get("ok", True) else 1))
        metadata = output.get("metadata")
        return ToolResult(
            tool=tool,
            output=str(content),
            exit_code=exit_code,
            metadata=dict(metadata) if isinstance(metadata, dict) else None,
            truncated=bool(output.get("truncated", False)),
        )
    content = getattr(output, "content", getattr(output, "output", getattr(output, "text", output)))
    tool = str(getattr(output, "tool", getattr(output, "name", tool_name)))
    exit_code = int(getattr(output, "exit_code", 0 if getattr(output, "ok", True) else 1))
    metadata = getattr(output, "metadata", None)
    return ToolResult(
        tool=tool,
        output=str(content),
        exit_code=exit_code,
        metadata=dict(metadata) if isinstance(metadata, dict) else None,
        truncated=bool(getattr(output, "truncated", False)),
    )


def register_agent_tools(registry: Any, tools: Any) -> None:
    register_dobox_tools(registry, tools)


def build_agent_tool_registry(tools: Any, registry: Any | None = None) -> Any:
    if registry is None:
        registry = try_create_weav_tool_registry() or LocalToolRegistry()
    register_agent_tools(registry, tools)
    return registry
