from __future__ import annotations

import sys
import types
from unittest import IsolatedAsyncioTestCase

from docode.dobox.tools import DoBoxTools
from docode.agent.tools import CompositeAgentTools
from docode.web.tools import WebTools, WebToolsConfig
from docode.llm.credentials import ProviderCredential
from docode.llm.runtime import (
    LLMUsageMeter,
    LocalLLMRouter,
    OpenAICompatibleChatClient,
    ProviderCallResult,
    GitHubTrendingCrawlerDecisionLLM,
    ScriptedDecisionLLM,
    WeavDecisionLLM,
    WeavVerifierJudge,
    build_docode_llm,
    build_docode_runtime,
    build_provider_client,
    estimate_tokens,
    parse_verifier_judgement,
    provider_call_result,
)
from docode.storage.models import CodingJob, new_id
from docode.dobox.types import ToolResult


class RuntimeResolver:
    def __init__(self) -> None:
        self.resolve_calls = 0

    async def resolve(self, *, user_id: str, provider: str, model: str) -> ProviderCredential:
        self.resolve_calls += 1
        return ProviderCredential(provider=provider, model=model, api_key="secret-key", base_url="https://llm.example/v1")


class RuntimeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ("weav_ai_core", "weav_ai_core.llm", "weav_ai_providers")}

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

    async def test_scripted_runtime_does_not_resolve_credentials(self) -> None:
        resolver = RuntimeResolver()
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="script it", provider="scripted", model="scripted")

        runtime = await build_docode_runtime(job, resolver)
        llm = await build_docode_llm(job, resolver)

        self.assertIsInstance(runtime.router, LocalLLMRouter)
        self.assertIsInstance(runtime.llm, ScriptedDecisionLLM)
        self.assertIsInstance(llm, ScriptedDecisionLLM)
        self.assertEqual(resolver.resolve_calls, 0)

    async def test_github_trending_template_uses_objective_id_from_instruction(self) -> None:
        llm = GitHubTrendingCrawlerDecisionLLM(
            "Build crawler\nObjective id: obj_github_trending_abc123\nTarget: GitHub Trending"
        )

        self.assertIn('OBJECTIVE_ID = "obj_github_trending_abc123"', llm.files["crawler.py"])

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

        usage = LLMUsageMeter()
        provider = Provider()
        llm = WeavDecisionLLM(provider, "gpt-test", usage)

        decision = await llm.decide(system="system", messages=[], tools=[], context="context")

        self.assertEqual(decision.type, "final_candidate")
        self.assertEqual(usage.calls, 1)
        self.assertEqual(usage.prompt_tokens, estimate_tokens(provider.prompt))
        self.assertGreater(usage.total_tokens, 0)

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

    async def test_provider_call_result_extracts_sdk_style_object(self) -> None:
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
    sys.modules["weav_ai_core"] = core
    sys.modules["weav_ai_core.llm"] = llm
    sys.modules["weav_ai_providers"] = providers
