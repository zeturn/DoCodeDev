from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Protocol

from docode.dobox.tools import ToolDefinition

from .provider_compat import call_provider
from .usage import LLMUsageMeter


TOOL_DECISION_TYPES = {
    "read_file",
    "read_file_range",
    "read_symbol",
    "list_files",
    "search",
    "write_file",
    "edit_file",
    "replace_in_file",
    "apply_patch",
    "run_command",
    "run_tests",
    "run_build",
    "run_lint",
    "git_status",
    "git_diff",
    "web_search",
    "fetch_url",
    "inspect_source",
    "preview",
    "logs",
}


@dataclass(frozen=True, slots=True)
class AgentDecision:
    type: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    summary: str | None = None
    verification: str | None = None
    no_test_reason: str | None = None
    remaining_risks: list[str] | None = None
    reasoning: str | None = None
    reasoning_records: list[dict[str, object]] | None = None


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
        decision = parse_decision(result.text)
        if result.reasoning or result.reasoning_records:
            return replace(
                decision,
                reasoning=result.reasoning,
                reasoning_records=list(result.reasoning_records or []),
            )
        return decision

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
    decision_type = str(data.get("type") or data.get("action") or "").strip()
    reasoning, reasoning_records = parse_decision_reasoning(data)
    if decision_type == "tool_call":
        return AgentDecision(
            type="tool_call",
            tool_name=str(data["tool_name"]),
            args=dict(data.get("args") or {}),
            reasoning=reasoning,
            reasoning_records=reasoning_records,
        )
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
            reasoning=reasoning,
            reasoning_records=reasoning_records,
        )
    if decision_type == "tool":
        tool = data.get("tool")
        if isinstance(tool, dict):
            tool_name = str(tool.get("tool_name") or tool.get("name") or "")
            if tool_name:
                return AgentDecision(
                    type="tool_call",
                    tool_name=tool_name,
                    args=dict(tool.get("input") or tool.get("args") or {}),
                    reasoning=reasoning,
                    reasoning_records=reasoning_records,
                )
        tool_name = str(data.get("tool_name") or data.get("name") or "")
        if tool_name:
            return AgentDecision(
                type="tool_call",
                tool_name=tool_name,
                args=dict(data.get("input") or data.get("args") or {}),
                reasoning=reasoning,
                reasoning_records=reasoning_records,
            )
    if decision_type and isinstance(data.get("args"), dict):
        return AgentDecision(
            type="tool_call",
            tool_name=decision_type,
            args=dict(data.get("args") or {}),
            reasoning=reasoning,
            reasoning_records=reasoning_records,
        )
    if decision_type in TOOL_DECISION_TYPES:
        args = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "type",
                "action",
                "tool",
                "tool_name",
                "name",
                "reasoning_summary",
                "reasoning",
                "thinking_summary",
                "thinking",
                "thought_summary",
                "thoughts",
            }
        }
        return AgentDecision(type="tool_call", tool_name=decision_type, args=args, reasoning=reasoning, reasoning_records=reasoning_records)
    if isinstance(data.get("tool"), dict):
        tool = data["tool"]
        tool_name = str(tool.get("tool_name") or tool.get("name") or "")
        if tool_name:
            return AgentDecision(
                type="tool_call",
                tool_name=tool_name,
                args=dict(tool.get("input") or tool.get("args") or {}),
                reasoning=reasoning,
                reasoning_records=reasoning_records,
            )
    raise ValueError(f"unsupported decision type: {decision_type}")


def parse_decision_reasoning(data: dict[str, Any]) -> tuple[str | None, list[dict[str, object]] | None]:
    records: list[dict[str, object]] = []
    for key in ("reasoning_summary", "reasoning", "thinking_summary", "thinking", "thought_summary", "thoughts"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            records.append({"type": key, "text": value.strip()[:4000], "source": "decision_json"})
        elif isinstance(value, list):
            for item in value:
                text = str(item.get("text") or item.get("summary") or "") if isinstance(item, dict) else str(item)
                if text.strip():
                    records.append({"type": key, "text": text.strip()[:4000], "source": "decision_json"})
    if not records:
        return None, None
    text = "\n\n".join(str(record["text"]) for record in records)
    return text, records


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data
