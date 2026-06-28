from __future__ import annotations

from dataclasses import dataclass, field

from docode.llm.credentials import APICredCredentialResolver, ProviderCredential

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
    purpose: str = "docode"
    _credential: ProviderCredential | None = field(default=None, init=False, repr=False)

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
        self.provider = normalize_openai_compatible_provider(provider or self.provider)
        if model:
            self.model = model
        credential = await self._resolve()
        self.provider = normalize_openai_compatible_provider(credential.provider)
        self.model = credential.model
        return ModelSpec(provider=self.provider, model=self.model)

    async def _credential_for(self, provider: str, context: AIRuntimeContext) -> ProviderCredential | None:
        _ = context
        if normalize_openai_compatible_provider(provider) != self.provider:
            return None
        credential = await self._resolve()
        self.provider = normalize_openai_compatible_provider(credential.provider)
        self.model = credential.model
        if normalize_openai_compatible_provider(provider) != self.provider:
            return None
        return credential

    async def _resolve(self) -> ProviderCredential:
        if self._credential is None:
            credential = await self.resolver.resolve(user_id=self.user_id, provider=self.provider, model=self.model)
            self._credential = ProviderCredential(
                provider=normalize_openai_compatible_provider(credential.provider),
                model=credential.model,
                api_key=credential.api_key,
                base_url=credential.base_url,
            )
        return self._credential


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
