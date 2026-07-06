from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderCredential:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeAuthorization:
    allowed: bool
    reason: str = ""
    budget_tokens: int | None = None
    budget_cost: float | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)


class APICredCredentialResolver:
    """Resolve provider credentials through APICred without persisting keys."""

    def __init__(
        self,
        base_url: str,
        access_token: str = "",
        mode: str = "auto",
        local_credentials: dict[str, ProviderCredential] | None = None,
        *,
        retry_attempts: int = 5,
        retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.mode = normalize_apicred_mode(mode)
        self.proxy_active = self.mode == "proxy"
        self.calls: list[dict[str, Any]] = []
        self.local_credentials = dict(local_credentials or {})
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delays = retry_delays

    def use_access_token(self, access_token: str | None) -> None:
        if access_token:
            self.access_token = access_token

    async def authorize(
        self,
        *,
        user_id: str,
        provider: str,
        model: str,
        job_id: str,
        max_iterations: int,
        max_runtime_seconds: int | None = None,
        max_tool_calls: int | None = None,
        max_llm_tokens: int | None = None,
        max_llm_cost: float | None = None,
        sandbox_network_mode: str | None = None,
        artifact_mode: str | None = None,
    ) -> RuntimeAuthorization:
        local = self._local_credential(provider)
        if local is not None:
            return RuntimeAuthorization(allowed=True, reason=f"local_direct_credential:{provider}", raw={"provider": provider, "model": model})
        payload: dict[str, object] = {
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "job_id": job_id,
            "purpose": "docode",
            "max_iterations": max_iterations,
        }
        optional_policy = {
            "max_runtime_seconds": max_runtime_seconds,
            "max_tool_calls": max_tool_calls,
            "max_llm_tokens": max_llm_tokens,
            "max_llm_cost": max_llm_cost,
            "sandbox_network_mode": sandbox_network_mode,
            "artifact_mode": artifact_mode,
        }
        payload.update({key: value for key, value in optional_policy.items() if value is not None})
        if self.proxy_active:
            return self._proxy_authorization()

        try:
            data = await self._post("/runtime/authorize", payload)
        except Exception as exc:
            if is_local_scripted_runtime(provider, model):
                return RuntimeAuthorization(allowed=True, reason="local_scripted_runtime", raw={"error": str(exc)})
            if self.mode == "auto" and is_missing_runtime_endpoint(exc):
                self.proxy_active = True
                return self._proxy_authorization(raw={"runtime_error": str(exc)})
            raise RuntimeError(f"apicred_authorize_unavailable:{exc}") from exc

        allowed = bool(data.get("allowed", data.get("authorized", True)))
        return RuntimeAuthorization(
            allowed=allowed,
            reason=str(data.get("reason", "")),
            budget_tokens=int(data["budget_tokens"]) if data.get("budget_tokens") is not None else None,
            budget_cost=float(data["budget_cost"]) if data.get("budget_cost") is not None else None,
            raw=data,
        )

    async def resolve(self, *, user_id: str, provider: str, model: str) -> ProviderCredential:
        local = self._local_credential(provider, model=model)
        if local is not None:
            return local
        if self.proxy_active:
            return self._proxy_credential(provider=provider, model=model)

        payload = {
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "purpose": "docode",
        }
        try:
            data = await self._post("/runtime/credentials/resolve", payload)
        except Exception as exc:
            if self.mode == "auto" and is_missing_runtime_endpoint(exc):
                self.proxy_active = True
                return self._proxy_credential(provider=provider, model=model)
            raise RuntimeError(f"apicred_credentials_resolve_unavailable:{exc}") from exc
        return ProviderCredential(
            provider=str(data.get("provider", provider)),
            model=str(data.get("model", model)),
            api_key=data.get("api_key") or data.get("token") or None,
            base_url=data.get("base_url"),
        )

    def _local_credential(self, provider: str, model: str | None = None) -> ProviderCredential | None:
        credential = self.local_credentials.get(provider)
        if credential is None:
            return None
        resolved_model = model or credential.model
        return ProviderCredential(provider=credential.provider, model=resolved_model, api_key=credential.api_key, base_url=credential.base_url)

    async def list_providers(self, *, user_id: str | None = None) -> dict[str, list[str]]:
        if self.proxy_active:
            try:
                data = await self._get("/models", {"user_id": user_id, "purpose": "docode"})
            except Exception:
                return {}
            return parse_provider_catalog(data)
        try:
            data = await self._get("/runtime/providers", {"user_id": user_id, "purpose": "docode"})
        except Exception:
            try:
                data = await self._get("/models", {"user_id": user_id, "purpose": "docode"})
            except Exception:
                return {}
        return parse_provider_catalog(data)

    async def report_usage(self, *, user_id: str, provider: str, model: str, tokens: int = 0, cost: float = 0.0) -> None:
        if self.proxy_active:
            self.calls.append(
                {
                    "method": "SKIP",
                    "path": "/runtime/usage/report",
                    "reason": "apicred_proxy_chat_completions_bills_usage",
                    "payload": {"user_id": user_id, "provider": provider, "model": model, "tokens": tokens, "cost": cost},
                }
            )
            return
        await self._post(
            "/runtime/usage/report",
            {
                "user_id": user_id,
                "provider": provider,
                "model": model,
                "tokens": tokens,
                "cost": cost,
                "purpose": "docode",
            },
        )

    async def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        import httpx

        self.calls.append({"method": "POST", "path": path, "payload": dict(payload)})
        headers = {"Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return await self._request_json("POST", path, json=payload, headers=headers)

    def _proxy_authorization(self, raw: dict[str, Any] | None = None) -> RuntimeAuthorization:
        return RuntimeAuthorization(allowed=True, reason="apicred_proxy_chat_completions", raw=raw)

    def _proxy_credential(self, *, provider: str, model: str) -> ProviderCredential:
        return ProviderCredential(provider=provider, model=model, api_key=self.access_token, base_url=self.base_url)

    async def _get(self, path: str, params: dict[str, object | None] | None = None) -> dict[str, object]:
        import httpx

        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        self.calls.append({"method": "GET", "path": path, "params": dict(clean_params)})
        headers = {"Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return await self._request_json("GET", path, params=clean_params, headers=headers)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        json=json,
                        params=params,
                        headers=headers,
                    )
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {"data": data}
            except Exception as exc:
                last_exc = exc
                if not is_retryable_apicred_exception(exc) or attempt >= self.retry_attempts:
                    raise
                delay = retry_delay_with_jitter(self.retry_delays, attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc


def parse_provider_catalog(data: dict[str, object]) -> dict[str, list[str]]:
    providers = data.get("providers")
    if isinstance(providers, dict):
        return {str(provider): normalize_models(models) for provider, models in providers.items()}
    if isinstance(providers, list):
        catalog: dict[str, list[str]] = {}
        for item in providers:
            if isinstance(item, str):
                catalog.setdefault(item, [])
            elif isinstance(item, dict):
                provider = item.get("provider") or item.get("name") or item.get("id")
                if provider:
                    catalog[str(provider)] = normalize_models(item.get("models") or item.get("model_ids") or [])
        return catalog

    models = data.get("models") or data.get("data")
    normalized = normalize_openai_models(models)
    return {"openai": normalized} if normalized else {}


def normalize_models(value: object) -> list[str]:
    if isinstance(value, dict):
        value = value.get("models") or value.get("data") or []
    if isinstance(value, list):
        return [model for model in (model_id(item) for item in value) if model]
    if isinstance(value, str):
        return [value]
    return []


def normalize_openai_models(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [model for model in (model_id(item) for item in value) if model]


def model_id(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("id") or value.get("name") or value.get("model")
        return str(candidate) if candidate else None
    return None


def is_local_scripted_runtime(provider: str, model: str) -> bool:
    return provider in {"scripted", "dev"} or model == "scripted"


def normalize_apicred_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    return mode if mode in {"auto", "runtime", "proxy"} else "auto"


def is_missing_runtime_endpoint(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return status_code in {404, 405}


def is_retryable_apicred_exception(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        fragment in text
        for fragment in (
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "server disconnected",
            "temporary failure",
            "bad gateway",
            "service unavailable",
            "too many requests",
        )
    )


def retry_delay_with_jitter(retry_delays: tuple[float, ...], attempt: int) -> float:
    if not retry_delays:
        return 0.0
    base = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
    if base <= 0:
        return 0.0
    return base + random.uniform(0.0, min(1.0, base * 0.2))
