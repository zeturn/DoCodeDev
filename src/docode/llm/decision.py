from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from docode.dobox.tools import ToolDefinition

from .provider_compat import call_provider
from .usage import LLMUsageMeter


@dataclass(frozen=True, slots=True)
class AgentDecision:
    type: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    summary: str | None = None
    verification: str | None = None
    no_test_reason: str | None = None
    remaining_risks: list[str] | None = None


class DecisionLLM(Protocol):
    async def decide(self, *, system: str, messages: list[dict[str, Any]], tools: list[ToolDefinition], context: str) -> AgentDecision: ...


class DoCodeDecisionAdapter:
    def __init__(self, provider_client: Any, model: str, usage_meter: LLMUsageMeter | None = None) -> None:
        self.provider_client = provider_client
        self.model = model
        self.usage_meter = usage_meter

    async def decide(self, *, system: str, messages: list[dict[str, Any]], tools: list[ToolDefinition], context: str) -> AgentDecision:
        prompt = self._format_prompt(system, messages, tools, context)
        result = await call_provider(self.provider_client, prompt, self.model)
        if self.usage_meter is not None:
            self.usage_meter.record_provider_call(prompt=prompt, result=result)
        return parse_decision(result.text)

    def _format_prompt(self, system: str, messages: list[dict[str, Any]], tools: list[ToolDefinition], context: str) -> str:
        tool_specs = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema(),
            }
            for tool in tools
        ]
        return (
            f"{system}\n\nAvailable tools JSON schema:\n{json.dumps(tool_specs, ensure_ascii=False)}\n\n"
            "Respond as JSON: {\"type\":\"tool_call\",\"tool_name\":\"...\",\"args\":{...}} "
            "or {\"type\":\"final_candidate\",\"summary\":\"...\",\"verification\":\"...\","
            "\"no_test_reason\":null,\"remaining_risks\":[]}.\n\n"
            f"Context:\n{context}"
        )


WeavDecisionLLM = DoCodeDecisionAdapter


def parse_decision(raw: str) -> AgentDecision:
    data = parse_json_object(raw)
    decision_type = str(data.get("type", ""))
    if decision_type == "tool_call":
        return AgentDecision(type="tool_call", tool_name=str(data["tool_name"]), args=dict(data.get("args") or {}))
    if decision_type == "final_candidate":
        risks = data.get("remaining_risks") or []
        if not isinstance(risks, list):
            risks = [str(risks)]
        no_test_reason = data.get("no_test_reason")
        return AgentDecision(
            type="final_candidate",
            summary=str(data.get("summary") or ""),
            verification=str(data.get("verification") or ""),
            no_test_reason=str(no_test_reason) if no_test_reason else None,
            remaining_risks=[str(risk) for risk in risks if str(risk)],
        )
    raise ValueError(f"unsupported decision type: {decision_type}")


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data
