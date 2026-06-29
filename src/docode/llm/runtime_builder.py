from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docode.agent.weav_tools import build_agent_tool_registry
from docode.storage.models import CodingJob

from .credentials import APICredCredentialResolver
from .decision import DecisionLLM, DoCodeDecisionAdapter
from .dev_llms import GitHubTrendingCrawlerDecisionLLM, ScriptedDecisionLLM, is_github_trending_araneae_instruction
from .provider_compat import LocalLLMRouter
from .usage import LLMUsageMeter
from .weav_apicred_store import APICredCredentialStore, APICredUsageSink, normalize_openai_compatible_provider


@dataclass(slots=True)
class DocodeRuntime:
    provider: str
    model: str
    llm: DecisionLLM
    router: Any = field(repr=False)
    tools: Any = field(repr=False)
    provider_client: Any | None = field(default=None, repr=False)
    usage_sink: Any | None = field(default=None, repr=False)
    usage_meter: LLMUsageMeter = field(default_factory=lambda: LLMUsageMeter(), repr=False)


async def build_docode_llm(job: CodingJob, resolver: APICredCredentialResolver) -> DecisionLLM:
    runtime = await build_docode_runtime(job, resolver)
    return runtime.llm


async def build_docode_runtime(job: CodingJob, resolver: APICredCredentialResolver, dobox_tools: Any | None = None) -> DocodeRuntime:
    tool_registry = build_agent_tool_registry(dobox_tools) if dobox_tools is not None else None
    usage_meter = LLMUsageMeter()
    usage_sink = APICredUsageSink(resolver)
    if job.provider in {"scripted", "dev"} or job.model == "scripted":
        return DocodeRuntime(
            provider="scripted",
            model="scripted",
            llm=ScriptedDecisionLLM(job.instruction),
            router=LocalLLMRouter(),
            tools=tool_registry,
            usage_sink=usage_sink,
            usage_meter=usage_meter,
        )
    if is_github_trending_araneae_instruction(job.instruction):
        return DocodeRuntime(
            provider=job.provider,
            model=job.model,
            llm=GitHubTrendingCrawlerDecisionLLM(job.instruction),
            router=LocalLLMRouter(),
            tools=tool_registry,
            usage_sink=usage_sink,
            usage_meter=usage_meter,
        )

    runtime_context, ai_runtime = build_weav_runtime(job, resolver, usage_sink)
    model_spec = await resolve_model_async(ai_runtime, provider=job.provider, model=job.model)
    router = await build_runtime_router_async(ai_runtime, runtime_context)
    provider_client = registered_provider(router, model_spec.provider)
    if provider_client is None:
        raise RuntimeError(f"provider_not_registered:{model_spec.provider}")
    return DocodeRuntime(
        provider=model_spec.provider,
        model=model_spec.model,
        llm=DoCodeDecisionAdapter(provider_client, model_spec.model, usage_meter),
        router=router,
        tools=tool_registry,
        provider_client=provider_client,
        usage_sink=usage_sink,
        usage_meter=usage_meter,
    )


def build_weav_runtime(job: CodingJob, resolver: APICredCredentialResolver, usage_sink: APICredUsageSink) -> tuple[Any, Any]:
    try:
        from weav_ai_runtime import AIRuntime, AIRuntimeContext
    except Exception as exc:  # pragma: no cover - only used with an out-of-date runtime install
        raise RuntimeError(f"weav_ai_runtime_unavailable:{exc}") from exc

    context = AIRuntimeContext(tenant=job.user_id, user_id=job.user_id, purpose="docode")
    credential_store = APICredCredentialStore(resolver=resolver, user_id=job.user_id, provider=job.provider, model=job.model, quality=job.quality)
    kwargs = {
        "context": context,
        "credentials": credential_store,
        "model_catalog": credential_store,
        "usage_sink": usage_sink,
    }
    try:
        kwargs["policy"] = build_runtime_policy(job)
        return context, AIRuntime(**kwargs)
    except TypeError:
        kwargs.pop("policy", None)
        return context, AIRuntime(**kwargs)


def build_runtime_policy(job: CodingJob) -> Any | None:
    try:
        from weav_ai_runtime import ModelSpec, RuntimePolicy
    except Exception:
        return None
    provider = normalize_openai_compatible_provider(job.provider) if job.provider and job.provider not in {"scripted", "dev"} else ""
    allowed_providers = [provider] if provider else []
    return RuntimePolicy(
        purpose="docode",
        max_tokens=job.max_llm_tokens,
        max_cost=job.max_llm_cost,
        allowed_providers=allowed_providers,
        fallback_chain=[ModelSpec(provider=provider, model=job.model)] if provider and job.model else [],
    )


async def resolve_model_async(ai_runtime: Any, *, provider: str, model: str) -> Any:
    if hasattr(ai_runtime, "resolve_model_async"):
        return await ai_runtime.resolve_model_async(provider=provider, model=model)
    if ai_runtime.model_catalog is not None and hasattr(ai_runtime.model_catalog, "resolve_model"):
        resolved = ai_runtime.model_catalog.resolve_model(ai_runtime.context, provider=provider, model=model)
        if hasattr(resolved, "__await__"):
            return await resolved
        return resolved
    return ai_runtime.resolve_model(provider=provider, model=model)


async def build_runtime_router_async(ai_runtime: Any, context: Any) -> Any:
    if hasattr(ai_runtime, "build_router_async"):
        return await ai_runtime.build_router_async()
    try:
        from weav_ai_runtime import build_router_async
    except Exception as exc:  # pragma: no cover - only used with an out-of-date runtime install
        raise RuntimeError(f"weav_ai_runtime_async_router_unavailable:{exc}") from exc
    return await build_router_async(context, ai_runtime.credentials)


def registered_provider(router: Any, provider: str) -> Any | None:
    if hasattr(router, "get"):
        try:
            found = router.get(provider)
        except TypeError:
            found = None
        if found is not None:
            return found
    providers = getattr(router, "providers", None)
    if isinstance(providers, dict):
        return providers.get(provider)
    return None
