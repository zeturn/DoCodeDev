from __future__ import annotations

import sys
import types
from unittest import IsolatedAsyncioTestCase

import httpx

from docode.dobox.tools import DoBoxTools
from docode.dobox.tools import ToolDefinition
from docode.agent.tools import CompositeAgentTools
from docode.web.tools import WebTools, WebToolsConfig
from docode.llm.credentials import ProviderCredential
from docode.llm.runtime import (
    LLMUsageMeter,
    LocalLLMRouter,
    OpenAICompatibleChatClient,
    ProviderCallResult,
    ProviderUnavailableError,
    ScriptedDecisionLLM,
    WeavDecisionLLM,
    WeavVerifierJudge,
    build_runtime_policy,
    build_docode_llm,
    build_docode_runtime,
    call_provider,
    build_provider_client,
    estimate_tokens,
    parse_verifier_judgement,
    provider_call_result,
)
from docode.llm import runtime as runtime_surface
from docode.storage.models import CodingJob, new_id
from docode.dobox.types import ToolResult


class RuntimeResolver:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.proxy_active = False

    async def resolve(self, *, user_id: str, provider: str, model: str) -> ProviderCredential:
        self.resolve_calls += 1
        return ProviderCredential(provider=provider, model=model, api_key="secret-key", base_url="https://llm.example/v1")


class ProxyRuntimeResolver(RuntimeResolver):
    def __init__(self) -> None:
        super().__init__()
        self.proxy_active = True


class RuntimeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ("weav_ai_core", "weav_ai_core.llm", "weav_ai_providers", "weav_ai_runtime")
        }

    async def asyncTearDown(self) -> None:
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    async def test_build_docode_runtime_registers_provider_and_tools(self) -> None:
        install_weav_stubs()
        resolver = RuntimeResolver()
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="change code", provider="openai", model="gpt-test")
        dobox_tools = DoBoxTools(object(), "project-1")
        agent_tools = CompositeAgentTools(dobox_tools, WebTools(WebToolsConfig(openai_api_key="key-1")))

        runtime = await build_docode_runtime(job, resolver, agent_tools)

        self.assertEqual(runtime.provider, "openai")
        self.assertEqual(runtime.model, "gpt-test")
        self.assertEqual(resolver.resolve_calls, 1)
        self.assertEqual(
            runtime.router.providers["openai"],
            {"provider": "openai", "kwargs": {"api_key": "secret-key", "base_url": "https://llm.example/v1"}},
        )
        self.assertIsNotNone(runtime.tools.get("run_command"))
        self.assertIsNotNone(runtime.tools.get("web_search"))
        self.assertIsNotNone(runtime.tools.get("fetch_url"))
        self.assertNotIn("secret-key", repr(runtime))

    async def test_openai_provider_falls_back_without_weav_package(self) -> None:
        sys.modules.pop("weav_ai_providers", None)

        client = build_provider_client("openai", "secret-key", "https://llm.example/v1")

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual(client.base_url, "https://llm.example/v1")

    async def test_openai_compatible_provider_falls_back_without_weav_package(self) -> None:
        sys.modules.pop("weav_ai_providers", None)

        client = build_provider_client("deepseek", "secret-key", "https://llm.example/v1")

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual(client.base_url, "https://llm.example/v1")

    async def test_build_docode_runtime_falls_back_without_weav_runtime(self) -> None:
        sys.modules.pop("weav_ai_runtime", None)
        sys.modules.pop("weav_ai_providers", None)
        resolver = RuntimeResolver()
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="change code", provider="openai", model="gpt-test")

        runtime = await build_docode_runtime(job, resolver, DoBoxTools(object(), "project-1"))

        self.assertEqual(runtime.provider, "openai")
        self.assertEqual(runtime.model, "gpt-test")
        self.assertEqual(resolver.resolve_calls, 1)
        self.assertIsInstance(runtime.router, LocalLLMRouter)
        self.assertIsInstance(runtime.provider_client, OpenAICompatibleChatClient)
        self.assertEqual(runtime.provider_client.base_url, "https://llm.example/v1")
        self.assertIsNotNone(runtime.tools.get("run_command"))

    async def test_build_docode_runtime_uses_proxy_fallback_without_weav_router(self) -> None:
        install_weav_stubs()
        sys.modules.pop("weav_provider_router", None)
        resolver = ProxyRuntimeResolver()
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="change code", provider="openrouter", model="tencent/hy3:free")

        runtime = await build_docode_runtime(job, resolver, DoBoxTools(object(), "project-1"))

        self.assertEqual(runtime.provider, "openrouter")
        self.assertEqual(runtime.model, "tencent/hy3:free")
        self.assertEqual(resolver.resolve_calls, 1)
        self.assertIsInstance(runtime.router, LocalLLMRouter)
        self.assertIsInstance(runtime.provider_client, OpenAICompatibleChatClient)
        self.assertEqual(runtime.provider_client.base_url, "https://llm.example/v1")

    async def test_scripted_runtime_does_not_resolve_credentials(self) -> None:
        resolver = RuntimeResolver()
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="script it", provider="scripted", model="scripted")

        runtime = await build_docode_runtime(job, resolver)
        llm = await build_docode_llm(job, resolver)

        self.assertIsInstance(runtime.router, LocalLLMRouter)
        self.assertIsInstance(runtime.llm, ScriptedDecisionLLM)
        self.assertIsInstance(llm, ScriptedDecisionLLM)
        self.assertEqual(resolver.resolve_calls, 0)

    async def test_weav_verifier_judge_parses_structured_judgement(self) -> None:
        class Provider:
            def complete(self, *, prompt, model):
                self.prompt = prompt
                self.model = model
                return '{"passed": true, "confidence": 1.7, "reason": "Looks good.", "required_fixes": []}'

        provider = Provider()
        usage = LLMUsageMeter()
        judge = WeavVerifierJudge(provider, "gpt-test", usage)

        judgement = await judge.judge(
            instruction="update readme",
            status=ToolResult(tool="git_status", output=" M README.md\n"),
            diff="diff --git a/README.md b/README.md\n+done\n",
            tests=ToolResult(tool="run_tests", output="ok", metadata={"command": "pytest", "detected": True}),
            build=ToolResult(tool="run_build", output="ok"),
            lint=ToolResult(tool="run_lint", output="ok"),
        )

        self.assertTrue(judgement.passed)
        self.assertEqual(judgement.confidence, 1.0)
        self.assertIn("Instruction:", provider.prompt)
        self.assertIn("Git status:", provider.prompt)
        self.assertIn("M README.md", provider.prompt)
        self.assertEqual(provider.model, "gpt-test")
        self.assertGreater(usage.total_tokens, 0)

    async def test_parse_verifier_judgement_accepts_wrapped_json(self) -> None:
        judgement = parse_verifier_judgement('result:\n{"passed": false, "confidence": 0.2, "reason": "missing", "required_fixes": "add tests"}')

        self.assertFalse(judgement.passed)
        self.assertEqual(judgement.required_fixes, ["add tests"])

    async def test_weav_decision_llm_records_estimated_usage(self) -> None:
        class Provider:
            def complete(self, *, prompt, model):
                self.prompt = prompt
                self.model = model
                return '{"type": "final_candidate", "summary": "done"}'

        async def read_file(path: str):
            _ = path

        usage = LLMUsageMeter()
        provider = Provider()
        llm = WeavDecisionLLM(provider, "gpt-test", usage)

        decision = await llm.decide(
            system="system",
            messages=[],
            tools=[ToolDefinition("read_file", "Read a file.", {"path": "string"}, read_file)],
            context="context",
        )

        self.assertEqual(decision.type, "final_candidate")
        self.assertIn("Available tools JSON schema", provider.prompt)
        self.assertIn('"input_schema"', provider.prompt)
        self.assertIn('"read_file"', provider.prompt)
        self.assertNotIn("Messages:", provider.prompt)
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.prompt_tokens, estimate_tokens(provider.prompt))
        self.assertGreater(usage.total_tokens, 0)

    async def test_final_candidate_supports_structured_verification_fields(self) -> None:
        decision = runtime_surface.parse_decision(
            '{"type":"final_candidate","summary":"done","verification":"pytest passed",'
            '"no_test_reason":"No automated test is appropriate for this config-only change; manual verification was performed.",'
            '"remaining_risks":["depends on prod config"]}'
        )

        self.assertEqual(decision.summary, "done")
        self.assertEqual(decision.verification, "pytest passed")
        self.assertIn("No automated test", decision.no_test_reason)
        self.assertEqual(decision.remaining_risks, ["depends on prod config"])

    async def test_parse_decision_accepts_tool_name_as_type(self) -> None:
        decision = runtime_surface.parse_decision('{"type":"run_command","args":{"command":"python3 cli.py"}}')

        self.assertEqual(decision.type, "tool_call")
        self.assertEqual(decision.tool_name, "run_command")
        self.assertEqual(decision.args, {"command": "python3 cli.py"})

    def test_runtime_all_excludes_legacy_provider_helpers(self) -> None:
        self.assertIn("build_docode_runtime", runtime_surface.__all__)
        self.assertNotIn("call_provider_legacy", runtime_surface.__all__)
        self.assertIn("call_provider_legacy", runtime_surface.LEGACY_RUNTIME_EXPORTS)
        self.assertTrue(hasattr(runtime_surface, "call_provider_legacy"))

    async def test_weav_decision_llm_uses_provider_reported_usage(self) -> None:
        class Provider:
            def chat(self, *, messages, model):
                self.messages = messages
                self.model = model
                return {
                    "choices": [{"message": {"content": '{"type": "final_candidate", "summary": "done"}'}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    "cost": 0.03,
                }

        usage = LLMUsageMeter()
        provider = Provider()
        llm = WeavDecisionLLM(provider, "gpt-test", usage)

        decision = await llm.decide(system="system", messages=[], tools=[], context="context")

        self.assertEqual(decision.summary, "done")
        self.assertEqual(provider.model, "gpt-test")
        self.assertEqual(usage.prompt_tokens, 11)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertFalse(usage.estimated)

    async def test_call_provider_falls_back_when_runtime_uses_incompatible_config_shape(self) -> None:
        runtime = types.ModuleType("weav_ai_runtime")

        async def incompatible_call_llm_provider(client, *, prompt, model, purpose=None):
            _ = client, prompt, model, purpose
            raise AttributeError("'dict' object has no attribute 'model'")

        runtime.call_llm_provider = incompatible_call_llm_provider
        sys.modules["weav_ai_runtime"] = runtime

        class Provider:
            def complete(self, *args, **kwargs):
                if kwargs:
                    raise TypeError("positional config required")
                prompt, config = args
                self.prompt = prompt
                self.model = config.model
                return '{"type": "final_candidate", "summary": "fallback"}'

        result = await call_provider(Provider(), "prompt", "gpt-test")

        self.assertEqual(result.text, '{"type": "final_candidate", "summary": "fallback"}')

    async def test_call_provider_retries_transient_provider_5xx(self) -> None:
        sys.modules.pop("weav_ai_runtime", None)

        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, prompt, model):
                _ = prompt, model
                self.calls += 1
                if self.calls < 3:
                    request = httpx.Request("POST", "http://localhost:8103/v1/chat/completions")
                    response = httpx.Response(503, request=request, text='{"error":{"code":"no_upstream_capacity"}}')
                    raise httpx.HTTPStatusError("service unavailable", request=request, response=response)
                return '{"type": "final_candidate", "summary": "recovered"}'

        provider = Provider()

        result = await call_provider(provider, "prompt", "gpt-test", max_attempts=3, retry_delays=(0.0, 0.0))

        self.assertEqual(provider.calls, 3)
        self.assertEqual(result.text, '{"type": "final_candidate", "summary": "recovered"}')

    async def test_call_provider_raises_provider_unavailable_after_retry_budget(self) -> None:
        sys.modules.pop("weav_ai_runtime", None)

        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, *, prompt, model):
                _ = prompt, model
                self.calls += 1
                request = httpx.Request("POST", "http://localhost:8103/v1/chat/completions")
                response = httpx.Response(502, request=request, text="bad gateway")
                raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

        provider = Provider()

        with self.assertRaises(ProviderUnavailableError) as raised:
            await call_provider(provider, "prompt", "gpt-test", max_attempts=2, retry_delays=(0.0,))

        self.assertEqual(provider.calls, 2)
        self.assertEqual(raised.exception.category, "provider_upstream_unavailable")
        self.assertEqual(raised.exception.attempts, 2)

    async def test_provider_call_result_extracts_sdk_style_object(self) -> None:
        install_weav_stubs()

        class Message:
            content = '{"type": "final_candidate", "summary": "object"}'

        class Choice:
            message = Message()

        class Usage:
            input_tokens = 5
            output_tokens = 3
            total_tokens = 8

        class Response:
            choices = [Choice()]
            usage = Usage()

        result = provider_call_result(Response())

        self.assertIsInstance(result, ProviderCallResult)
        self.assertEqual(result.text, '{"type": "final_candidate", "summary": "object"}')
        self.assertEqual(result.prompt_tokens, 5)
        self.assertEqual(result.completion_tokens, 3)

    async def test_provider_call_result_accepts_runtime_llm_call_result_shape(self) -> None:
        install_weav_stubs()

        class Usage:
            tokens = 21
            cost = 0.04

        class RuntimeResult:
            text = '{"type": "final_candidate", "summary": "runtime"}'
            usage = Usage()
            tool_calls = []
            raw = {"id": "call_1"}

        result = provider_call_result(RuntimeResult())

        self.assertEqual(result.text, '{"type": "final_candidate", "summary": "runtime"}')
        self.assertEqual(result.total_tokens, 21)
        self.assertEqual(result.cost, 0.04)
        self.assertEqual(result.raw, {"id": "call_1"})

    async def test_build_runtime_policy_maps_job_budget_and_provider(self) -> None:
        install_weav_stubs()
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="change code",
            provider="apicred",
            model="gpt-test",
            max_llm_tokens=1234,
            max_llm_cost=0.5,
        )

        policy = build_runtime_policy(job)

        self.assertIsNotNone(policy)
        self.assertEqual(policy.purpose, "docode")
        self.assertEqual(policy.max_tokens, 1234)
        self.assertEqual(policy.max_cost, 0.5)
        self.assertEqual(policy.allowed_providers, ["openai"])
        self.assertEqual(policy.fallback_chain[0].provider, "openai")


def install_weav_stubs() -> None:
    core = types.ModuleType("weav_ai_core")
    llm = types.ModuleType("weav_ai_core.llm")

    class LLMRouter:
        def __init__(self) -> None:
            self.providers = {}

        def register(self, name, provider) -> None:
            self.providers[name] = provider

    core.LLMRouter = LLMRouter
    llm.LLMRouter = LLMRouter

    providers = types.ModuleType("weav_ai_providers")

    def build_provider(provider, **kwargs):
        return {"provider": provider, "kwargs": kwargs}

    providers.build_provider = build_provider

    runtime = types.ModuleType("weav_ai_runtime")

    class AIRuntimeContext:
        def __init__(self, tenant=None, user_id=None, purpose=None) -> None:
            self.tenant = tenant
            self.user_id = user_id
            self.purpose = purpose

    class ModelSpec:
        def __init__(self, provider, model) -> None:
            self.provider = provider
            self.model = model

    class RuntimePolicy:
        def __init__(self, *, purpose, max_tokens=None, max_cost=None, allowed_providers=None, denied_models=None, fallback_chain=None) -> None:
            self.purpose = purpose
            self.max_tokens = max_tokens
            self.max_cost = max_cost
            self.allowed_providers = list(allowed_providers or [])
            self.denied_models = list(denied_models or [])
            self.fallback_chain = list(fallback_chain or [])

    class UsageRecord:
        def __init__(self, tokens=0, cost=0.0, prompt_tokens=None, completion_tokens=None, provider=None, model=None, purpose=None) -> None:
            self.tokens = tokens
            self.cost = cost
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.provider = provider
            self.model = model
            self.purpose = purpose

    class LLMCallResult:
        def __init__(self, text, tool_calls=None, usage=None, raw=None) -> None:
            self.text = text
            self.tool_calls = list(tool_calls or [])
            self.usage = usage
            self.raw = raw

    def normalize_llm_call_result(response, **kwargs):
        _ = kwargs
        if isinstance(response, LLMCallResult):
            return response
        if hasattr(response, "text") and hasattr(response, "usage"):
            usage = response.usage
            return LLMCallResult(
                response.text,
                tool_calls=getattr(response, "tool_calls", []),
                usage=UsageRecord(
                    tokens=getattr(usage, "tokens", getattr(usage, "total_tokens", 0)),
                    cost=getattr(usage, "cost", 0.0),
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                ),
                raw=getattr(response, "raw", response),
            )
        if isinstance(response, str):
            return LLMCallResult(response, raw=response)
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        tokens = None
        prompt_tokens = None
        completion_tokens = None
        if isinstance(usage, dict):
            tokens = usage.get("total_tokens") or usage.get("tokens")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        elif usage is not None:
            tokens = getattr(usage, "total_tokens", getattr(usage, "tokens", None))
            prompt_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
            completion_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None))
        text = None
        if isinstance(response, dict):
            text = response.get("choices", [{}])[0].get("message", {}).get("content")
        if text is None:
            text = getattr(response, "text", None) or getattr(getattr(response, "choices", [None])[0], "message", None).content
        return LLMCallResult(
            text,
            usage=UsageRecord(tokens=tokens or 0, cost=getattr(usage, "cost", 0.0), prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            raw=getattr(response, "raw", response),
        )

    async def call_llm_provider(client, *, prompt, model, provider=None, purpose=None, config=None):
        _ = provider, purpose, config
        if hasattr(client, "acomplete"):
            try:
                response = await client.acomplete(prompt=prompt, model=model)
            except TypeError:
                response = await client.acomplete(prompt, {"model": model, "temperature": 0.0})
            return normalize_llm_call_result(response)
        if hasattr(client, "complete"):
            try:
                response = client.complete(prompt=prompt, model=model)
            except TypeError:
                response = client.complete(prompt, {"model": model, "temperature": 0.0})
            if hasattr(response, "__await__"):
                response = await response
            return normalize_llm_call_result(response)
        if hasattr(client, "achat"):
            try:
                response = await client.achat(messages=[{"role": "user", "content": prompt}], model=model)
            except TypeError:
                response = await client.achat([{"role": "user", "content": prompt}], {"model": model, "temperature": 0.0})
            return normalize_llm_call_result(response)
        if hasattr(client, "chat"):
            try:
                response = client.chat(messages=[{"role": "user", "content": prompt}], model=model)
            except TypeError:
                response = client.chat([{"role": "user", "content": prompt}], {"model": model, "temperature": 0.0})
            if hasattr(response, "__await__"):
                response = await response
            return normalize_llm_call_result(response)
        raise RuntimeError("provider client does not expose a supported chat/completion method")

    class AIRuntime:
        def __init__(self, *, context, credentials, model_catalog=None, usage_sink=None) -> None:
            self.context = context
            self.credentials = credentials
            self.model_catalog = model_catalog
            self.usage_sink = usage_sink

        async def resolve_model_async(self, provider=None, model=None):
            return await self.model_catalog.resolve_model(self.context, provider=provider, model=model)

        async def build_router_async(self):
            router = LLMRouter()
            for provider in ("openai", "anthropic", "google", "ollama", "deepseek", "qwen", "zhipu"):
                api_key = await self.credentials.get_api_key(provider, self.context)
                base_url = await self.credentials.get_base_url(provider, self.context)
                if not api_key and provider != "ollama":
                    continue
                kwargs = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                router.register(provider, build_provider(provider, **kwargs))
            return router

    runtime.AIRuntime = AIRuntime
    runtime.AIRuntimeContext = AIRuntimeContext
    runtime.LLMCallResult = LLMCallResult
    runtime.ModelSpec = ModelSpec
    runtime.RuntimePolicy = RuntimePolicy
    runtime.UsageRecord = UsageRecord
    runtime.call_llm_provider = call_llm_provider
    runtime.normalize_llm_call_result = normalize_llm_call_result
    sys.modules["weav_ai_core"] = core
    sys.modules["weav_ai_core.llm"] = llm
    sys.modules["weav_ai_providers"] = providers
    sys.modules["weav_ai_runtime"] = runtime
