from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from docode.agent.output import prompt_safe_output
from docode.agent.quality_gate import QualityGateResult
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult
from docode.llm.decision import parse_json_object
from docode.llm.provider_compat import call_provider
from docode.llm.usage import LLMUsageMeter


@dataclass(frozen=True, slots=True)
class ReviewResult:
    passed: bool
    confidence: float
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_plan: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "independent_review",
            "passed": self.passed,
            "confidence": self.confidence,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "repair_plan": self.repair_plan,
            "reason": self.reason,
        }


class CodeReviewer(Protocol):
    async def review(
        self,
        *,
        instruction: str,
        task_contract: TaskContract | None,
        quality: QualityGateResult,
        recent_tool_results: list[ToolResult],
        final_summary: str,
    ) -> ReviewResult: ...


class IndependentReviewer:
    def __init__(
        self,
        provider_client: Any,
        model: str,
        usage_meter: LLMUsageMeter | None = None,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.provider_client = provider_client
        self.model = model
        self.usage_meter = usage_meter
        self.timeout_seconds = timeout_seconds

    async def review(
        self,
        *,
        instruction: str,
        task_contract: TaskContract | None,
        quality: QualityGateResult,
        recent_tool_results: list[ToolResult],
        final_summary: str,
    ) -> ReviewResult:
        prompt = format_reviewer_prompt(
            instruction=instruction,
            task_contract=task_contract,
            quality=quality,
            recent_tool_results=recent_tool_results,
            final_summary=final_summary,
        )
        try:
            result = await asyncio.wait_for(call_provider(self.provider_client, prompt, self.model), timeout=self.timeout_seconds)
        except TimeoutError:
            return ReviewResult(
                passed=True,
                confidence=0.0,
                warnings=["independent reviewer timed out; deterministic quality gate and verifier still ran"],
                reason="reviewer_timeout",
            )
        if self.usage_meter is not None:
            self.usage_meter.record_provider_call(prompt=prompt, result=result)
        return parse_review_result(result.text)


def parse_review_result(raw: str) -> ReviewResult:
    data = parse_json_object(raw)
    return ReviewResult(
        passed=bool(data.get("passed", False)),
        confidence=clamp_confidence(data.get("confidence", 0.0)),
        blocking_issues=string_list(data.get("blocking_issues")),
        warnings=string_list(data.get("warnings")),
        repair_plan=string_list(data.get("repair_plan")),
        reason=str(data.get("reason") or ""),
    )


def format_reviewer_prompt(
    *,
    instruction: str,
    task_contract: TaskContract | None,
    quality: QualityGateResult,
    recent_tool_results: list[ToolResult],
    final_summary: str,
) -> str:
    payload = {
        "instruction": instruction,
        "task_contract": {
            "must_modify_files": task_contract.must_modify_files,
            "must_run_commands": task_contract.must_run_commands,
        }
        if task_contract is not None
        else None,
        "final_summary": final_summary,
        "quality_gate": quality.to_dict(),
        "recent_tool_results": [tool_result_for_review(result) for result in recent_tool_results[-8:]],
    }
    return (
        "You are DoCode's independent quality reviewer.\n\n"
        "Review the task instruction, task contract, diff, verification outputs, and artifact samples.\n"
        "Find only blocking issues that affect correctness, reproducibility, dependency safety, or output usefulness.\n"
        "Ignore style-only suggestions.\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "passed": true,\n'
        '  "confidence": 0.86,\n'
        '  "blocking_issues": [],\n'
        '  "warnings": [],\n'
        '  "repair_plan": [],\n'
        '  "reason": ""\n'
        "}\n\n"
        f"Review input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def tool_result_for_review(result: ToolResult) -> dict[str, Any]:
    output = prompt_safe_output(result.output)
    command = result.metadata.get("command") if result.metadata else None
    return {
        "tool": result.tool,
        "command": command,
        "exit_code": result.exit_code,
        "truncated": result.truncated or output.truncated,
        "output": output.text,
    }


def string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def clamp_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
