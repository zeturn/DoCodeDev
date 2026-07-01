from __future__ import annotations

"""
Compatibility import surface for DoCode LLM runtime primitives.

New code should import focused modules directly:
- docode.llm.decision for AgentDecision and decision parsing
- docode.llm.runtime_builder for runtime assembly
- docode.llm.usage for usage metering
- docode.llm.verifier_judge for verifier LLM judging

Legacy provider compatibility helpers remain importable from this module for
older callers, but are intentionally omitted from __all__.
"""

from .decision import AgentDecision, DecisionLLM, DoCodeDecisionAdapter, WeavDecisionLLM, parse_decision, parse_json_object
from .dev_llms import ScriptedDecisionLLM
from .provider_compat import (
    LocalLLMRouter,
    OpenAICompatibleChatClient,
    ProviderCallResult,
    ProviderUnavailableError,
    build_provider_client,
    call_provider,
    classify_provider_exception,
    call_provider_legacy,
    call_provider_text,
    content_to_text,
    create_llm_router,
    extract_text,
    extract_usage,
    first_present,
    float_or_none,
    get_field,
    import_llm_router,
    int_or_none,
    normalize_provider_response,
    provider_call_result,
    provider_completion_config,
    provider_result_from_runtime_result,
    register_provider,
)
from .runtime_builder import (
    DocodeRuntime,
    build_docode_llm,
    build_docode_runtime,
    build_runtime_policy,
    build_runtime_router_async,
    build_weav_runtime,
    registered_provider,
    resolve_model_async,
)
from .usage import LLMUsageMeter, estimate_tokens
from .verifier_judge import WeavVerifierJudge, clamp_confidence, parse_verifier_judgement, tool_result_for_prompt, truncate_for_prompt

__all__ = [
    "AgentDecision",
    "DecisionLLM",
    "DoCodeDecisionAdapter",
    "DocodeRuntime",
    "LLMUsageMeter",
    "WeavDecisionLLM",
    "WeavVerifierJudge",
    "build_docode_llm",
    "build_docode_runtime",
]

LEGACY_RUNTIME_EXPORTS = [
    "LocalLLMRouter",
    "OpenAICompatibleChatClient",
    "ProviderCallResult",
    "ProviderUnavailableError",
    "ScriptedDecisionLLM",
    "build_provider_client",
    "build_runtime_policy",
    "build_runtime_router_async",
    "build_weav_runtime",
    "call_provider",
    "classify_provider_exception",
    "call_provider_legacy",
    "call_provider_text",
    "clamp_confidence",
    "content_to_text",
    "create_llm_router",
    "estimate_tokens",
    "extract_text",
    "extract_usage",
    "first_present",
    "float_or_none",
    "get_field",
    "import_llm_router",
    "int_or_none",
    "normalize_provider_response",
    "parse_decision",
    "parse_json_object",
    "parse_verifier_judgement",
    "provider_call_result",
    "provider_completion_config",
    "provider_result_from_runtime_result",
    "registered_provider",
    "register_provider",
    "resolve_model_async",
    "tool_result_for_prompt",
    "truncate_for_prompt",
]
