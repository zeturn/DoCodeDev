from __future__ import annotations

from dataclasses import dataclass, field

from docode.defaults import DEFAULT_MODEL, DEFAULT_PROVIDER, DEFAULT_QUALITY
from docode.llm.credentials import APICredCredentialResolver, ProviderCredential
from docode.llm.model_policy import ModelOption, best_option_for_quality, normalize_quality

try:
    from weav_ai_runtime import AIRuntimeContext, ModelSpec, UsageRecord
except Exception:  # pragma: no cover - import guard for partially upgraded dev envs
    @dataclass(frozen=True, slots=True)
    class AIRuntimeContext:
        tenant: str | None = None
        user_id: str | None = None
        purpose: str | None = None

    @dataclass(frozen=True, slots=True)
    class ModelSpec:
        provider: str
        model: str

    @dataclass(frozen=True, slots=True)
    class UsageRecord:
        tokens: int
        cost: float
        provider: str | None = None
        model: str | None = None
        tenant: str | None = None
        user_id: str | None = None
        purpose: str | None = None


@dataclass(slots=True)
class APICredCredentialStore:
    resolver: APICredCredentialResolver
    user_id: str
    provider: str
    model: str
    quality: str = DEFAULT_QUALITY
    purpose: str = "docode"
    _restricted_provider: str = field(default="", init=False, repr=False)
    _credentials: dict[tuple[str, str], ProviderCredential] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.provider = normalize_openai_compatible_provider(self.provider) if self.provider else ""
        self.quality = normalize_quality(self.quality)
        self._restricted_provider = self.provider

    async def get_api_key(self, provider: str, context: AIRuntimeContext) -> str | None:
        credential = await self._credential_for(provider, context)
        return credential.api_key if credential is not None else None

    async def get_base_url(self, provider: str, context: AIRuntimeContext) -> str | None:
        credential = await self._credential_for(provider, context)
        return credential.base_url if credential is not None else None

    async def list_models(self, context: AIRuntimeContext, provider: str | None = None) -> dict[str, list[str]]:
        _ = context
        catalog = await self.resolver.list_providers(user_id=self.user_id)
        if provider is None:
            return catalog
        return {provider: catalog.get(provider, [])} if provider in catalog else {}

    async def resolve_model(
        self,
        context: AIRuntimeContext,
        provider: str | None = None,
        model: str | None = None,
    ) -> ModelSpec:
        _ = context
        selected_provider, selected_model = await self._select_model(provider=provider, model=model)
        credential = await self._resolve(selected_provider, selected_model)
        self.provider = normalize_openai_compatible_provider(credential.provider)
        self.model = credential.model
        return ModelSpec(provider=self.provider, model=self.model)

    async def _credential_for(self, provider: str, context: AIRuntimeContext) -> ProviderCredential | None:
        _ = context
        requested_provider = normalize_openai_compatible_provider(provider)
        if self._restricted_provider and requested_provider != self._restricted_provider:
            return None
        selected_provider, selected_model = await self._select_model(provider=requested_provider, model=self.model or None)
        credential = await self._resolve(selected_provider, selected_model)
        resolved_provider = normalize_openai_compatible_provider(credential.provider)
        if self._restricted_provider:
            self.provider = resolved_provider
            self.model = credential.model
        if requested_provider != resolved_provider:
            return None
        return credential

    async def _select_model(self, *, provider: str | None, model: str | None) -> tuple[str, str]:
        requested_provider = normalize_openai_compatible_provider(provider or self.provider or "")
        requested_model = model or self.model or ""
        if requested_provider and requested_model:
            return requested_provider, requested_model

        catalog = await self.resolver.list_providers(user_id=self.user_id)
        options = catalog_options(catalog)
        if requested_provider:
            scoped = [option for option in options if normalize_openai_compatible_provider(option.provider) == requested_provider and option.model]
            if scoped:
                return requested_provider, best_option_for_quality(scoped, normalize_quality(self.quality), default_quality_config()).model
            return requested_provider, requested_model or DEFAULT_MODEL

        default = next((option for option in options if option.provider == DEFAULT_PROVIDER and option.model == DEFAULT_MODEL), None)
        if normalize_quality(self.quality) == DEFAULT_QUALITY and default is not None:
            return default.provider, default.model
        if options:
            chosen = best_option_for_quality([option for option in options if option.model], normalize_quality(self.quality), default_quality_config())
            return normalize_openai_compatible_provider(chosen.provider), chosen.model
        return DEFAULT_PROVIDER, DEFAULT_MODEL

    async def _resolve(self, provider: str, model: str) -> ProviderCredential:
        key = (normalize_openai_compatible_provider(provider), model)
        if key not in self._credentials:
            credential = await self.resolver.resolve(user_id=self.user_id, provider=key[0], model=key[1])
            resolved = ProviderCredential(
                provider=normalize_openai_compatible_provider(credential.provider),
                model=credential.model,
                api_key=credential.api_key,
                base_url=credential.base_url,
            )
            self._credentials[key] = resolved
            self._credentials[(resolved.provider, resolved.model)] = resolved
        return self._credentials[key]


@dataclass(slots=True)
class APICredUsageSink:
    resolver: APICredCredentialResolver

    async def record(self, usage: UsageRecord) -> None:
        await self.resolver.report_usage(
            user_id=usage.user_id or "",
            provider=usage.provider or "",
            model=usage.model or "",
            tokens=usage.tokens,
            cost=usage.cost,
        )


def usage_record_from_snapshot(
    *,
    user_id: str,
    provider: str,
    model: str,
    usage: dict[str, object],
    purpose: str = "docode",
) -> UsageRecord:
    return UsageRecord(
        user_id=user_id,
        provider=provider,
        model=model,
        purpose=purpose,
        tokens=int(usage.get("total_tokens") or 0),
        cost=float(usage.get("cost") or 0.0),
    )


def normalize_openai_compatible_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"apicred", "openai-compatible", "openai_compatible"}:
        return "openai"
    return normalized


def catalog_options(catalog: dict[str, list[str]]) -> list[ModelOption]:
    return [
        ModelOption(provider=normalize_openai_compatible_provider(provider), model=model)
        for provider, models in sorted(catalog.items())
        for model in sorted(set(models))
        if model
    ]


@dataclass(frozen=True, slots=True)
class _DefaultQualityConfig:
    default_provider: str = DEFAULT_PROVIDER
    default_model: str = DEFAULT_MODEL


def default_quality_config() -> _DefaultQualityConfig:
    return _DefaultQualityConfig()
