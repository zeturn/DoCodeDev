from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from docode.config import DocodeConfig
from docode.llm.credentials import parse_provider_catalog
from docode.llm.model_policy import DocodeModelPolicy


class FakeCatalogResolver:
    def __init__(self, catalog: dict[str, list[str]]) -> None:
        self.catalog = catalog
        self.user_ids: list[str | None] = []

    async def list_providers(self, *, user_id: str | None = None) -> dict[str, list[str]]:
        self.user_ids.append(user_id)
        return self.catalog


class ModelPolicyTests(IsolatedAsyncioTestCase):
    async def test_list_options_includes_apicred_catalog_default_and_scripted(self) -> None:
        resolver = FakeCatalogResolver({"anthropic": ["claude-sonnet-4-5"], "openai": ["gpt-4o"]})
        policy = DocodeModelPolicy(DocodeConfig(default_provider="openai", default_model="gpt-4o"), resolver)

        options = await policy.list_options(user_id="user-1")

        self.assertIn(("anthropic", "claude-sonnet-4-5", "apicred"), [(option.provider, option.model, option.source) for option in options])
        self.assertIn(("scripted", "scripted", "local"), [(option.provider, option.model, option.source) for option in options])
        self.assertTrue(next(option for option in options if option.provider == "openai" and option.model == "gpt-4o").default)
        self.assertEqual(resolver.user_ids, ["user-1"])

    async def test_resolve_rejects_unavailable_provider_or_model(self) -> None:
        policy = DocodeModelPolicy(DocodeConfig(default_provider="openai", default_model="gpt-4o"), FakeCatalogResolver({"openai": ["gpt-4o"]}))

        allowed = await policy.resolve(provider="openai", model="gpt-4o", user_id="user-1")
        wrong_model = await policy.resolve(provider="openai", model="gpt-missing", user_id="user-1")
        wrong_provider = await policy.resolve(provider="anthropic", model="claude", user_id="user-1")

        self.assertTrue(allowed.allowed)
        self.assertFalse(wrong_model.allowed)
        self.assertEqual(wrong_model.reason, "model_not_available_for_provider:openai:gpt-missing")
        self.assertFalse(wrong_provider.allowed)
        self.assertEqual(wrong_provider.reason, "provider_not_available:anthropic")

    async def test_provider_without_model_catalog_allows_requested_model(self) -> None:
        policy = DocodeModelPolicy(DocodeConfig(), FakeCatalogResolver({"local": []}))

        result = await policy.resolve(provider="local", model="codellm", user_id="user-1")

        self.assertTrue(result.allowed)
        self.assertEqual(result.provider, "local")
        self.assertEqual(result.model, "codellm")

    async def test_defaults_and_scripted_work_without_apicred_catalog(self) -> None:
        policy = DocodeModelPolicy(DocodeConfig(default_provider="openai", default_model="gpt-4o"), FakeCatalogResolver({}))

        default_result = await policy.resolve(provider=None, model=None, user_id="user-1")
        scripted_result = await policy.resolve(provider="dev", model=None, user_id="user-1")

        self.assertTrue(default_result.allowed)
        self.assertEqual((default_result.provider, default_result.model), ("openai", "gpt-4o"))
        self.assertTrue(scripted_result.allowed)
        self.assertEqual((scripted_result.provider, scripted_result.model), ("scripted", "scripted"))


class ProviderCatalogParsingTests(TestCase):
    def test_parse_provider_catalog_accepts_runtime_and_models_shapes(self) -> None:
        runtime_catalog = parse_provider_catalog(
            {
                "providers": [
                    {"id": "anthropic", "models": [{"id": "claude-sonnet-4-5"}]},
                    {"provider": "openai", "model_ids": ["gpt-4o"]},
                ]
            }
        )
        openai_catalog = parse_provider_catalog({"models": [{"id": "gpt-4o-mini"}]})

        self.assertEqual(runtime_catalog, {"anthropic": ["claude-sonnet-4-5"], "openai": ["gpt-4o"]})
        self.assertEqual(openai_catalog, {"openai": ["gpt-4o-mini"]})
