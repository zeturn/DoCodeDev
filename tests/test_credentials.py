from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

import httpx

from docode.llm.credentials import APICredCredentialResolver, RuntimeAuthorization
from docode.llm.weav_apicred_store import APICredCredentialStore, APICredUsageSink, AIRuntimeContext, UsageRecord


class FailingUsageResolver(APICredCredentialResolver):
    async def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append({"method": "POST", "path": path, "payload": dict(payload)})
        raise RuntimeError("usage report unavailable")


class ResponseResolver(APICredCredentialResolver):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__("https://apicred.invalid/v1", "service-token")
        self.response = response

    async def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append({"method": "POST", "path": path, "payload": dict(payload)})
        return dict(self.response)


class MissingRuntimeResolver(APICredCredentialResolver):
    async def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append({"method": "POST", "path": path, "payload": dict(payload)})
        request = httpx.Request("POST", f"{self.base_url}{path}")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)


class CredentialResolverTests(IsolatedAsyncioTestCase):
    async def test_authorize_fails_closed_for_provider_backed_jobs_when_apicred_is_unavailable(self) -> None:
        resolver = FailingUsageResolver("https://apicred.invalid/v1", "service-token")

        with self.assertRaises(RuntimeError) as raised:
            await resolver.authorize(user_id="user-1", provider="openai", model="gpt-4o", job_id="job_1", max_iterations=5)

        self.assertIn("apicred_authorize_unavailable:usage report unavailable", str(raised.exception))
        self.assertEqual(resolver.calls[0]["path"], "/runtime/authorize")

    async def test_authorize_keeps_local_scripted_runtime_available_for_smoke_jobs(self) -> None:
        resolver = FailingUsageResolver("https://apicred.invalid/v1", "service-token")

        authorization = await resolver.authorize(user_id="user-1", provider="scripted", model="scripted", job_id="job_1", max_iterations=5)

        self.assertTrue(authorization.allowed)
        self.assertEqual(authorization.reason, "local_scripted_runtime")
        self.assertEqual(resolver.calls[0]["path"], "/runtime/authorize")

    async def test_authorize_sends_runtime_limits_and_policy(self) -> None:
        resolver = ResponseResolver({"allowed": True, "budget_tokens": 100, "budget_cost": 0.25})

        authorization = await resolver.authorize(
            user_id="user-1",
            provider="openai",
            model="gpt-4o",
            job_id="job_1",
            max_iterations=5,
            max_runtime_seconds=900,
            max_tool_calls=20,
            max_llm_tokens=1000,
            max_llm_cost=0.5,
            sandbox_network_mode="no_internet",
            artifact_mode="pr",
        )

        payload = resolver.calls[0]["payload"]
        self.assertTrue(authorization.allowed)
        self.assertEqual(authorization.budget_tokens, 100)
        self.assertEqual(authorization.budget_cost, 0.25)
        self.assertEqual(payload["purpose"], "docode")
        self.assertEqual(payload["max_iterations"], 5)
        self.assertEqual(payload["max_runtime_seconds"], 900)
        self.assertEqual(payload["max_tool_calls"], 20)
        self.assertEqual(payload["max_llm_tokens"], 1000)
        self.assertEqual(payload["max_llm_cost"], 0.5)
        self.assertEqual(payload["sandbox_network_mode"], "no_internet")
        self.assertEqual(payload["artifact_mode"], "pr")

    async def test_resolve_fails_closed_when_apicred_credentials_are_unavailable(self) -> None:
        resolver = FailingUsageResolver("https://apicred.invalid/v1", "service-token")

        with self.assertRaises(RuntimeError) as raised:
            await resolver.resolve(user_id="user-1", provider="openai", model="gpt-4o")

        self.assertIn("apicred_credentials_resolve_unavailable:usage report unavailable", str(raised.exception))
        self.assertEqual(resolver.calls[0]["path"], "/runtime/credentials/resolve")

    async def test_resolve_does_not_use_apicred_service_token_as_provider_key(self) -> None:
        resolver = ResponseResolver({"provider": "openai", "model": "gpt-4o"})

        credential = await resolver.resolve(user_id="user-1", provider="openai", model="gpt-4o")

        self.assertIsNone(credential.api_key)
        self.assertNotIn("service-token", repr(credential))
        self.assertEqual(resolver.calls[0]["path"], "/runtime/credentials/resolve")

    async def test_resolve_accepts_runtime_token_from_apicred_response(self) -> None:
        resolver = ResponseResolver({"provider": "openai", "model": "gpt-4o", "token": "provider-runtime-token"})

        credential = await resolver.resolve(user_id="user-1", provider="openai", model="gpt-4o")

        self.assertEqual(credential.api_key, "provider-runtime-token")
        self.assertNotIn("provider-runtime-token", repr(credential))

    async def test_proxy_mode_skips_runtime_credit_and_uses_apicred_token(self) -> None:
        resolver = APICredCredentialResolver("https://apicred.example/v1", "apicred-api-token", mode="proxy")

        authorization = await resolver.authorize(user_id="user-1", provider="openai", model="gpt-5.4", job_id="job_1", max_iterations=5)
        credential = await resolver.resolve(user_id="user-1", provider="openai", model="gpt-5.4")
        await resolver.report_usage(user_id="user-1", provider="openai", model="gpt-5.4", tokens=12, cost=0.0)

        self.assertTrue(authorization.allowed)
        self.assertEqual(authorization.reason, "apicred_proxy_chat_completions")
        self.assertEqual(credential.api_key, "apicred-api-token")
        self.assertEqual(credential.base_url, "https://apicred.example/v1")
        self.assertEqual(credential.model, "gpt-5.4")
        self.assertEqual(resolver.calls[0]["method"], "SKIP")
        self.assertEqual(resolver.calls[0]["reason"], "apicred_proxy_chat_completions_bills_usage")

    async def test_auto_mode_falls_back_to_proxy_when_runtime_endpoint_is_missing(self) -> None:
        resolver = MissingRuntimeResolver("https://apicred.example/v1", "apicred-api-token", mode="auto")

        authorization = await resolver.authorize(user_id="user-1", provider="openai", model="gpt-5.4", job_id="job_1", max_iterations=5)
        credential = await resolver.resolve(user_id="user-1", provider="openai", model="gpt-5.4")

        self.assertTrue(authorization.allowed)
        self.assertTrue(resolver.proxy_active)
        self.assertEqual(credential.api_key, "apicred-api-token")
        self.assertEqual([call["path"] for call in resolver.calls], ["/runtime/authorize"])

    async def test_proxy_mode_requires_apicred_token(self) -> None:
        resolver = APICredCredentialResolver("https://apicred.example/v1", "", mode="proxy")

        with self.assertRaises(RuntimeError) as raised:
            await resolver.resolve(user_id="user-1", provider="openai", model="gpt-5.4")

        self.assertEqual(str(raised.exception), "apicred_proxy_token_required")

    async def test_runtime_authorization_repr_hides_raw_payload(self) -> None:
        authorization = RuntimeAuthorization(
            allowed=True,
            reason="ok",
            budget_tokens=100,
            raw={"signed_runtime_token": "secret-runtime-token"},
        )

        self.assertNotIn("secret-runtime-token", repr(authorization))

    async def test_report_usage_surfaces_failure_for_runner_audit(self) -> None:
        resolver = FailingUsageResolver("https://apicred.invalid/v1", "service-token")

        with self.assertRaises(RuntimeError) as raised:
            await resolver.report_usage(user_id="user-1", provider="openai", model="gpt-4o", tokens=12, cost=0.03)

        self.assertEqual(str(raised.exception), "usage report unavailable")
        self.assertEqual(resolver.calls[0]["path"], "/runtime/usage/report")
        self.assertEqual(resolver.calls[0]["payload"]["purpose"], "docode")

    async def test_apicred_credential_store_caches_async_resolution(self) -> None:
        resolver = ResponseResolver(
            {
                "provider": "openai",
                "model": "gpt-runtime",
                "api_key": "provider-key",
                "base_url": "https://llm.example/v1",
            }
        )
        store = APICredCredentialStore(resolver=resolver, user_id="user-1", provider="apicred", model="gpt-requested")
        context = AIRuntimeContext(user_id="user-1", purpose="docode")

        model = await store.resolve_model(context, provider="apicred", model="gpt-requested")
        api_key = await store.get_api_key("openai", context)
        base_url = await store.get_base_url("openai", context)

        self.assertEqual(model.provider, "openai")
        self.assertEqual(model.model, "gpt-runtime")
        self.assertEqual(api_key, "provider-key")
        self.assertEqual(base_url, "https://llm.example/v1")
        self.assertEqual([call["path"] for call in resolver.calls], ["/runtime/credentials/resolve"])

    async def test_apicred_usage_sink_records_usage_record(self) -> None:
        resolver = ResponseResolver({})
        sink = APICredUsageSink(resolver)

        await sink.record(UsageRecord(user_id="user-1", provider="openai", model="gpt-4o", tokens=12, cost=0.03, purpose="docode"))

        self.assertEqual(resolver.calls[0]["path"], "/runtime/usage/report")
        self.assertEqual(resolver.calls[0]["payload"]["user_id"], "user-1")
        self.assertEqual(resolver.calls[0]["payload"]["tokens"], 12)


if __name__ == "__main__":
    import unittest

    unittest.main()
