from __future__ import annotations

from typing import Any, Protocol

from docode.dobox.tools import ToolDefinition
from docode.dobox.types import ToolResult


class AgentToolset(Protocol):
    def definitions(self) -> list[ToolDefinition]: ...

    async def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult: ...


class CompositeAgentTools:
    def __init__(self, primary: Any, *extras: Any) -> None:
        self.primary = primary
        self.extras = [toolset for toolset in extras if toolset is not None]
        self._toolsets = [primary, *self.extras]

    def definitions(self) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        seen: set[str] = set()
        for toolset in self._toolsets:
            for definition in toolset.definitions():
                if definition.name in seen:
                    raise ValueError(f"duplicate tool name: {definition.name}")
                seen.add(definition.name)
                definitions.append(definition)
        return definitions

    async def call(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        for toolset in self._toolsets:
            if any(definition.name == tool_name for definition in toolset.definitions()):
                return await toolset.call(tool_name, args)
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)
