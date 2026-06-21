from __future__ import annotations

from dataclasses import dataclass

from docode.config import DocodeConfig
from docode.llm.credentials import APICredCredentialResolver


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


class DocodeModelPolicy:
    def __init__(self, config: DocodeConfig, resolver: APICredCredentialResolver) -> None:
        self.config = config
        self.resolver = resolver

    async def list_options(self, *, user_id: str | None = None) -> list[ModelOption]:
        catalog = await self.resolver.list_providers(user_id=user_id)
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

    async def resolve(self, *, provider: str | None, model: str | None, user_id: str | None = None) -> ModelPolicyResult:
        requested_provider = provider or self.config.default_provider
        requested_model = model or self.config.default_model
        if requested_provider in {"scripted", "dev"} or requested_model == "scripted":
            return ModelPolicyResult(provider="scripted", model="scripted", allowed=True)

        options = await self.list_options(user_id=user_id)
        if any(option.provider == requested_provider and option.model == requested_model for option in options):
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True)

        provider_options = [option for option in options if option.provider == requested_provider]
        if provider_options and any(option.model == "" for option in provider_options):
            return ModelPolicyResult(provider=requested_provider, model=requested_model, allowed=True)
        if provider_options:
            return ModelPolicyResult(
                provider=requested_provider,
                model=requested_model,
                allowed=False,
                reason=f"model_not_available_for_provider:{requested_provider}:{requested_model}",
            )
        return ModelPolicyResult(
            provider=requested_provider,
            model=requested_model,
            allowed=False,
            reason=f"provider_not_available:{requested_provider}",
        )


def is_default(provider: str, model: str, config: DocodeConfig) -> bool:
    return provider == config.default_provider and model == config.default_model


def add_option_if_missing(options: list[ModelOption], option: ModelOption) -> list[ModelOption]:
    if any(existing.provider == option.provider and existing.model == option.model for existing in options):
        return options
    return [*options, option]
