from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from docode.agent.output import prompt_safe_output
from docode.agent.verifier import VerifierJudgement
from docode.dobox.tools import DoBoxTools, ToolDefinition, build_dobox_tool_registry
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob

from .credentials import APICredCredentialResolver


@dataclass(frozen=True, slots=True)
class AgentDecision:
    type: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class DecisionLLM(Protocol):
    async def decide(self, *, system: str, messages: list[dict[str, Any]], tools: list[ToolDefinition], context: str) -> AgentDecision: ...


class LocalLLMRouter:
    """Small LLMRouter-compatible fallback for local development and tests."""

    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}

    def register(self, provider: str, client: Any) -> None:
        self.providers[provider] = client

    def get(self, provider: str) -> Any | None:
        return self.providers.get(provider)


@dataclass(slots=True)
class DocodeRuntime:
    provider: str
    model: str
    llm: DecisionLLM
    router: Any = field(repr=False)
    tools: Any = field(repr=False)
    provider_client: Any | None = field(default=None, repr=False)
    usage_meter: LLMUsageMeter = field(default_factory=lambda: LLMUsageMeter(), repr=False)


@dataclass(slots=True)
class LLMUsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    estimated: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_text_call(self, *, prompt: str, response: str, cost: float = 0.0) -> None:
        self.calls += 1
        self.prompt_tokens += estimate_tokens(prompt)
        self.completion_tokens += estimate_tokens(response)
        self.cost += cost

    def record_provider_call(self, *, prompt: str, result: ProviderCallResult) -> None:
        if result.prompt_tokens is None and result.completion_tokens is None and result.total_tokens is None:
            self.record_text_call(prompt=prompt, response=result.text, cost=result.cost or 0.0)
            return

        prompt_tokens = result.prompt_tokens if result.prompt_tokens is not None else (0 if result.total_tokens is not None else estimate_tokens(prompt))
        if result.completion_tokens is not None:
            completion_tokens = result.completion_tokens
        elif result.total_tokens is not None:
            completion_tokens = max(0, result.total_tokens - prompt_tokens)
        else:
            completion_tokens = estimate_tokens(result.text)

        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost += result.cost or 0.0
        self.estimated = False

    def snapshot(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "estimated": self.estimated,
        }


class WeavDecisionLLM:
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
        tool_names = ", ".join(tool.name for tool in tools)
        return (
            f"{system}\n\nAvailable tools: {tool_names}\n\n"
            "Respond as JSON: {\"type\":\"tool_call\",\"tool_name\":\"...\",\"args\":{...}} "
            "or {\"type\":\"final_candidate\",\"summary\":\"...\"}.\n\n"
            f"Context:\n{context}\n\nMessages:\n{json.dumps(messages[-20:], ensure_ascii=False)}"
        )


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
    ) -> VerifierJudgement:
        prompt = self._format_prompt(instruction=instruction, status=status, diff=diff, tests=tests, build=build, lint=lint)
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
    ) -> str:
        status_section = f"Git status:\n{tool_result_for_prompt(status)}\n\n" if status is not None else ""
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
        )


class ScriptedDecisionLLM:
    """Deterministic development LLM for smoke tests and local end-to-end runs."""

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.calls = 0

    async def decide(self, *, system: str, messages: list[dict[str, Any]], tools: list[ToolDefinition], context: str) -> AgentDecision:
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


async def build_docode_llm(job: CodingJob, resolver: APICredCredentialResolver) -> DecisionLLM:
    runtime = await build_docode_runtime(job, resolver)
    return runtime.llm


async def build_docode_runtime(job: CodingJob, resolver: APICredCredentialResolver, dobox_tools: DoBoxTools | None = None) -> DocodeRuntime:
    tool_registry = build_dobox_tool_registry(dobox_tools) if dobox_tools is not None else None
    usage_meter = LLMUsageMeter()
    if job.provider in {"scripted", "dev"} or job.model == "scripted":
        return DocodeRuntime(
            provider="scripted",
            model="scripted",
            llm=ScriptedDecisionLLM(job.instruction),
            router=LocalLLMRouter(),
            tools=tool_registry,
            usage_meter=usage_meter,
        )

    credential = await resolver.resolve(user_id=job.user_id, provider=job.provider, model=job.model)
    provider_client = build_provider_client(credential.provider, credential.api_key, credential.base_url)
    router = create_llm_router()
    register_provider(router, credential.provider, provider_client)
    return DocodeRuntime(
        provider=credential.provider,
        model=credential.model,
        llm=WeavDecisionLLM(provider_client, credential.model, usage_meter),
        router=router,
        tools=tool_registry,
        provider_client=provider_client,
        usage_meter=usage_meter,
    )


def build_provider_client(provider: str, api_key: str | None, base_url: str | None) -> Any:
    from weav_ai_providers import build_provider

    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return build_provider(provider, **kwargs)


async def call_provider(client: Any, prompt: str, model: str) -> ProviderCallResult:
    config = provider_completion_config(model)
    if hasattr(client, "acomplete"):
        try:
            response = await client.acomplete(prompt=prompt, model=model)
        except TypeError:
            response = await client.acomplete(prompt, config)
        return provider_call_result(response)
    if hasattr(client, "complete"):
        try:
            response = client.complete(prompt=prompt, model=model)
        except TypeError:
            response = client.complete(prompt, config)
            if hasattr(response, "__await__"):
                response = await response
        return provider_call_result(response)
    if hasattr(client, "achat"):
        try:
            response = await client.achat(messages=[{"role": "user", "content": prompt}], model=model)
        except TypeError:
            response = await client.achat([{"role": "user", "content": prompt}], config)
        return provider_call_result(response)
    if hasattr(client, "chat"):
        try:
            response = client.chat(messages=[{"role": "user", "content": prompt}], model=model)
        except TypeError:
            response = client.chat([{"role": "user", "content": prompt}], config)
            if hasattr(response, "__await__"):
                response = await response
        return provider_call_result(response)
    raise RuntimeError("provider client does not expose a supported chat/completion method")


async def call_provider_text(client: Any, prompt: str, model: str) -> str:
    return (await call_provider(client, prompt, model)).text


def provider_completion_config(model: str) -> Any:
    try:
        from weav_provider_router.base import CompletionConfig
    except Exception:
        return {"model": model}
    return CompletionConfig(model=model, temperature=0.0)


def provider_call_result(response: Any) -> ProviderCallResult:
    if isinstance(response, str):
        return ProviderCallResult(text=response)

    usage = extract_usage(response)
    text = extract_text(response)
    if text is None:
        text = str(response)
    return ProviderCallResult(
        text=text,
        prompt_tokens=int_or_none(usage.get("prompt_tokens")),
        completion_tokens=int_or_none(usage.get("completion_tokens")),
        total_tokens=int_or_none(usage.get("total_tokens")),
        cost=float_or_none(usage.get("cost")),
    )


def create_llm_router() -> Any:
    router_cls = import_llm_router()
    if router_cls is None:
        return LocalLLMRouter()
    try:
        return router_cls()
    except Exception:
        return LocalLLMRouter()


def import_llm_router() -> Any | None:
    try:
        from weav_ai_core import LLMRouter

        return LLMRouter
    except Exception:
        pass
    try:
        from weav_ai_core.llm import LLMRouter

        return LLMRouter
    except Exception:
        return None


def register_provider(router: Any, provider: str, client: Any) -> None:
    if hasattr(router, "register"):
        try:
            router.register(provider, client)
            return
        except TypeError:
            router.register(client)
            return
    if hasattr(router, "providers"):
        router.providers[provider] = client
        return
    raise RuntimeError("LLM router does not expose a supported register method")


def parse_decision(raw: str) -> AgentDecision:
    data = parse_json_object(raw)
    decision_type = str(data.get("type", ""))
    if decision_type == "tool_call":
        return AgentDecision(type="tool_call", tool_name=str(data["tool_name"]), args=dict(data.get("args") or {}))
    if decision_type == "final_candidate":
        return AgentDecision(type="final_candidate", summary=str(data.get("summary") or ""))
    raise ValueError(f"unsupported decision type: {decision_type}")


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


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


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


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative provider-agnostic estimate for billing guardrails when the
    # provider adapter does not expose structured usage metadata.
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def extract_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        direct = first_present(value, "output_text", "text", "content")
        if direct is not None:
            return content_to_text(direct)
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            return extract_text(choices[0])
        message = value.get("message")
        if message is not None:
            return extract_text(message)
        data = value.get("data")
        if data is not None:
            return extract_text(data)
        output = value.get("output")
        if output is not None:
            return content_to_text(output)
        return None

    for attr in ("output_text", "text", "content"):
        if hasattr(value, attr):
            return content_to_text(getattr(value, attr))
    if hasattr(value, "choices"):
        choices = getattr(value, "choices")
        if isinstance(choices, list) and choices:
            return extract_text(choices[0])
    if hasattr(value, "message"):
        return extract_text(getattr(value, "message"))
    return None


def content_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = extract_text(item)
            if text is not None:
                parts.append(text)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        direct = first_present(value, "text", "content", "value")
        return content_to_text(direct) if direct is not None else None
    return str(value)


def extract_usage(value: Any) -> dict[str, object]:
    usage = get_field(value, "usage") or get_field(value, "usage_metadata") or {}
    result: dict[str, object] = {}
    for target, candidates in {
        "prompt_tokens": ("prompt_tokens", "input_tokens", "prompt", "input"),
        "completion_tokens": ("completion_tokens", "output_tokens", "completion", "output"),
        "total_tokens": ("total_tokens", "tokens", "total"),
        "cost": ("cost", "total_cost", "amount"),
    }.items():
        result[target] = first_present(usage, *candidates)
    for target in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        if result.get(target) is None:
            result[target] = get_field(value, target)
    return result


def first_present(value: Any, *keys: str) -> object | None:
    for key in keys:
        candidate = get_field(value, key)
        if candidate is not None:
            return candidate
    return None


def get_field(value: Any, key: str) -> object | None:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, key):
        return getattr(value, key)
    return None


def int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
