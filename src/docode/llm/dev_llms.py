from __future__ import annotations

from docode.dobox.tools import ToolDefinition

from .decision import AgentDecision


class ScriptedDecisionLLM:
    """Deterministic development LLM for smoke tests and local end-to-end runs."""

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.calls = 0

    async def decide(self, *, system: str, messages: list[dict[str, object]], tools: list[ToolDefinition], context: str) -> AgentDecision:
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={
                    "path": "DOCODE_RESULT.md",
                    "content": f"# DoCode Result\n\nInstruction: {self.instruction}\n\nStatus: implemented by scripted development agent.\n",
                },
            )
        return AgentDecision(type="final_candidate", summary="Created DOCODE_RESULT.md and verified the workspace.")
