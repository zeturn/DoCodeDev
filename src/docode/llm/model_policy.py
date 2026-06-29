from __future__ import annotations

from dataclasses import dataclass

from docode.config import DocodeConfig
from docode.defaults import DEFAULT_QUALITY
from docode.llm.credentials import APICredCredentialResolver

QUALITY_TIERS = frozenset({"fast", "balanced", "strong"})


@dataclass(frozen=True, slots=True)
class ModelOption:
    provider: str
    model: str
    default: bool = False
    source: str = "apicred"


@dataclass(frozen=True, slots=True)
class ModelPolicyResult:
    provider: str
    model: str
    allowed: bool
    reason: str = ""
    quality: str = DEFAULT_QUALITY


class DocodeModelPolicy:
    def __init__(self, config: DocodeConfig, resolver: APICredCredentialResolver) -> None:
        self.config = config
        self.resolver = resolver

    async def list_options(self, *, user_id: str | None = None) -> list[ModelOption]:
        catalog = await self.resolver.list_providers(user_id=user_id)
        return self.options_from_catalog(catalog)

    async def list_models(self, context, provider: str | None = None) -> dict[str, list[str]]:
        user_id = getattr(context, "user_id", None)
        catalog = await self.resolver.list_providers(user_id=user_id)
        if provider is None:
            return catalog
        return {provider: catalog.get(provider, [])} if provider in catalog else {}

    async def resolve_model(
        self,
        context,
        provider: str | None = None,
        model: str | None = None,
        quality: str | None = None,
    ):
        result = await self.resolve(provider=provider, model=model, quality=quality, user_id=getattr(context, "user_id", None))
        if not result.allowed:
            raise ValueError(result.reason)
        try:
            from weav_ai_runtime import ModelSpec
        except Exception:
            return result
        return ModelSpec(provider=result.provider, model=result.model)

    def options_from_catalog(self, catalog: dict[str, list[str]]) -> list[ModelOption]:
        options: list[ModelOption] = []
        for provider, models in sorted(catalog.items()):
            if models:
                for model in sorted(set(models)):
                    options.append(
                        ModelOption(provider=provider, model=model, default=is_default(provider, model, self.config), source="apicred")
                    )
            else:
                options.append(
                    ModelOption(provider=provider, model="", default=provider == self.config.default_provider, source="apicred")
                )

        configured_default = ModelOption(
            provider=self.config.default_provider,
            model=self.config.default_model,
            default=True,
            source="config",
        )
        options = add_option_if_missing(options, configured_default)
        options = add_option_if_missing(options, ModelOption("scripted", "scripted", source="local"))
        return options

    async def resolve(
        self,
        *,
        provider: str | None,
        model: str | None,
        quality: str | None = None,
        user_id: str | None = None,
    ) -> ModelPolicyResult:
        requested_quality = normalize_quality(quality)
        requested_provider = provider or self.config.default_provider
        requested_model = model
        if requested_provider in {"scripted", "dev"} or requested_model == "scripted":
            return ModelPolicyResult(provider="scripted", model="scripted", allowed=True, quality=requested_quality)
        if requested_model is None:
            requested_provider, requested_model = await self.resolve_default_model(
                provider=provider,
                quality=requested_quality,
                user_id=user_id,
            )

        if requested_provider in {"scripted", "dev"} or requested_model == "scripted":
            return ModelPolicyResult(provider="scripted", model="scripted", allowed=True, quality=requested_quality)

        options = await self.list_options(user_id=user_id)
        if any(option.provider == requested_provider and option.model == requested_model for option in options):
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True, quality=requested_quality)

        provider_options = [option for option in options if option.provider == requested_provider]
        if provider_options and dated_snapshot_base_model(requested_model) in {option.model for option in provider_options}:
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True, quality=requested_quality)
        if provider_options and all(option.source == "config" for option in provider_options):
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True, quality=requested_quality)
        if provider_options and any(option.model == "" for option in provider_options):
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True, quality=requested_quality)
        if provider_options:
            return ModelPolicyResult(
                provider=requested_provider,
                model=requested_model,
                allowed=False,
                reason=f"model_not_available_for_provider:{requested_provider}:{requested_model}",
                quality=requested_quality,
            )
        return ModelPolicyResult(
            provider=requested_provider,
            model=requested_model,
            allowed=False,
            reason=f"provider_not_available:{requested_provider}",
            quality=requested_quality,
        )

    async def resolve_default_model(self, *, provider: str | None, quality: str, user_id: str | None = None) -> tuple[str, str]:
        options = [option for option in await self.list_options(user_id=user_id) if option.model and option.source != "local"]
        if provider is not None:
            scoped = [option for option in options if option.provider == provider]
            if scoped:
                return provider, best_model_for_quality(scoped, quality, self.config)
            return provider, self.config.default_model

        available_default = next(
            (option for option in options if option.provider == self.config.default_provider and option.model == self.config.default_model),
            None,
        )
        if quality == "balanced" and available_default is not None:
            return available_default.provider, available_default.model
        if not options:
            return self.config.default_provider, self.config.default_model
        chosen = best_option_for_quality(options, quality, self.config)
        return chosen.provider, chosen.model


def is_default(provider: str, model: str, config: DocodeConfig) -> bool:
    return provider == config.default_provider and model == config.default_model


def add_option_if_missing(options: list[ModelOption], option: ModelOption) -> list[ModelOption]:
    if any(existing.provider == option.provider and existing.model == option.model for existing in options):
        return options
    return [*options, option]


def normalize_quality(value: str | None) -> str:
    quality = (value or DEFAULT_QUALITY).strip().lower()
    if quality not in QUALITY_TIERS:
        raise ValueError("quality must be fast, balanced, or strong")
    return quality


def dated_snapshot_base_model(model: str | None) -> str:
    if model is None:
        return ""
    parts = model.rsplit("-", 3)
    if len(parts) != 4:
        return model
    year, month, day = parts[1:]
    if len(year) == 4 and len(month) == 2 and len(day) == 2 and year.isdigit() and month.isdigit() and day.isdigit():
        return parts[0]
    return model


def best_option_for_quality(options: list[ModelOption], quality: str, config: DocodeConfig) -> ModelOption:
    return min(options, key=lambda option: model_quality_rank(option, quality, config))


def best_model_for_quality(options: list[ModelOption], quality: str, config: DocodeConfig) -> str:
    return best_option_for_quality(options, quality, config).model


def model_quality_rank(option: ModelOption, quality: str, config: DocodeConfig) -> tuple[int, int, str, str]:
    model = option.model.lower()
    if option.provider == config.default_provider and option.model == config.default_model:
        default_rank = 0
    else:
        default_rank = 1
    if quality == "balanced":
        return (default_rank, 0, option.provider, option.model)
    if quality == "fast":
        return (fast_model_score(model), default_rank, option.provider, option.model)
    return (strong_model_score(model), default_rank, option.provider, option.model)


def fast_model_score(model: str) -> int:
    if any(marker in model for marker in ("mini", "small", "lite", "flash", "haiku", "fast", "turbo")):
        return 0
    if any(marker in model for marker in ("sonnet", "4o", "gpt-5", "pro")):
        return 2
    return 1


def strong_model_score(model: str) -> int:
    if any(marker in model for marker in ("opus", "sonnet", "pro", "max", "gpt-5")):
        return 0
    if any(marker in model for marker in ("gpt-4.1", "gpt-4o")):
        return 1
    if any(marker in model for marker in ("mini", "small", "lite", "flash", "haiku")):
        return 3
    return 2
