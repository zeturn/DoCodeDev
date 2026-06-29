from __future__ import annotations

import json
from typing import Any

from docode.agent.output import prompt_safe_output
from docode.agent.verifier import VerifierJudgement
from docode.dobox.types import ToolResult

from .decision import parse_json_object
from .provider_compat import call_provider
from .usage import LLMUsageMeter


class WeavVerifierJudge:
    def __init__(self, provider_client: Any, model: str, usage_meter: LLMUsageMeter | None = None) -> None:
        self.provider_client = provider_client
        self.model = model
        self.usage_meter = usage_meter

    async def judge(
        self,
        *,
        instruction: str,
        status: ToolResult | None = None,
        diff: str,
        tests: ToolResult,
        build: ToolResult,
        lint: ToolResult,
        smoke: ToolResult | None = None,
    ) -> VerifierJudgement:
        prompt = self._format_prompt(instruction=instruction, status=status, diff=diff, tests=tests, build=build, lint=lint, smoke=smoke)
        result = await call_provider(self.provider_client, prompt, self.model)
        if self.usage_meter is not None:
            self.usage_meter.record_provider_call(prompt=prompt, result=result)
        return parse_verifier_judgement(result.text)

    def _format_prompt(
        self,
        *,
        instruction: str,
        status: ToolResult | None = None,
        diff: str,
        tests: ToolResult,
        build: ToolResult,
        lint: ToolResult,
        smoke: ToolResult | None = None,
    ) -> str:
        status_section = f"Git status:\n{tool_result_for_prompt(status)}\n\n" if status is not None else ""
        smoke_section = f"\n\nSmoke:\n{tool_result_for_prompt(smoke)}" if smoke is not None else ""
        return (
            "You are DoCode's independent verifier. Review whether the code diff satisfies the user's instruction.\n"
            "You must consider the diff and the verification command outputs. Respond only as JSON with this shape:\n"
            "{\"passed\":true,\"confidence\":0.86,\"reason\":\"...\",\"required_fixes\":[]}.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"{status_section}"
            f"Diff:\n{truncate_for_prompt(diff, 30000)}\n\n"
            f"Tests:\n{tool_result_for_prompt(tests)}\n\n"
            f"Build:\n{tool_result_for_prompt(build)}\n\n"
            f"Lint:\n{tool_result_for_prompt(lint)}"
            f"{smoke_section}"
        )


def parse_verifier_judgement(raw: str) -> VerifierJudgement:
    data = parse_json_object(raw)
    required_fixes = data.get("required_fixes") or []
    if not isinstance(required_fixes, list):
        required_fixes = [str(required_fixes)]
    return VerifierJudgement(
        passed=bool(data.get("passed", False)),
        confidence=clamp_confidence(data.get("confidence", 0.0)),
        reason=str(data.get("reason") or ""),
        required_fixes=[str(fix) for fix in required_fixes if str(fix)],
    )


def clamp_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def tool_result_for_prompt(result: ToolResult) -> str:
    command = result.metadata.get("command") if result.metadata else None
    detected = result.metadata.get("detected") if result.metadata else None
    output = prompt_safe_output(result.output)
    return json.dumps(
        {
            "tool": result.tool,
            "command": command,
            "detected": detected,
            "exit_code": result.exit_code,
            "truncated": result.truncated or output.truncated,
            "original_output_lines": output.original_lines if output.truncated else None,
            "original_output_bytes": output.original_bytes if output.truncated else None,
            "output": output.text,
        },
        ensure_ascii=False,
    )


def truncate_for_prompt(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + "\n<truncated>"
