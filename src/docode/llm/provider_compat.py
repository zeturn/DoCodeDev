from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderErrorInfo:
    category: str
    retryable: bool
    status_code: int | None = None
    detail: str = ""


class ProviderUnavailableError(RuntimeError):
    def __init__(self, info: ProviderErrorInfo, *, attempts: int, cause: Exception) -> None:
        self.info = info
        self.category = info.category
        self.retryable = info.retryable
        self.status_code = info.status_code
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"{info.category}: {info.detail}")


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    tool_calls: list[Any] = field(default_factory=list)
    raw: Any = None


class LocalLLMRouter:
    """Small LLMRouter-compatible fallback for local development and tests."""

    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}

    def register(self, provider: str, client: Any) -> None:
        self.providers[provider] = client

    def get(self, provider: str) -> Any | None:
        return self.providers.get(provider)


class OpenAICompatibleChatClient:
    def __init__(self, api_key: str | None, base_url: str | None = None, timeout_seconds: float = 120.0) -> None:
        self.api_key = api_key or ""
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def achat(self, *, messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}


OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "apicred", "deepseek", "qwen", "zhipu", "openrouter"}


def build_provider_client(provider: str, api_key: str | None, base_url: str | None) -> Any:
    if base_url and provider.lower() in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleChatClient(api_key=api_key, base_url=base_url)
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    try:
        from weav_ai_providers import build_provider

        return build_provider(provider, **kwargs)
    except Exception:
        if provider.lower() in OPENAI_COMPATIBLE_PROVIDERS:
            return OpenAICompatibleChatClient(api_key=api_key, base_url=base_url)
        raise


async def call_provider(
    client: Any,
    prompt: str,
    model: str,
    *,
    max_attempts: int = 5,
    retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> ProviderCallResult:
    attempts = max(1, max_attempts)
    last_info: ProviderErrorInfo | None = None
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await call_provider_once(client, prompt, model)
        except Exception as exc:
            info = classify_provider_exception(exc)
            if info is None:
                raise
            last_info = info
            last_exc = exc
            if not info.retryable or attempt >= attempts:
                raise ProviderUnavailableError(info, attempts=attempt, cause=exc) from exc
            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)] if retry_delays else 0.0
            if delay > 0:
                delay += retry_jitter(delay)
            if delay > 0:
                await asyncio.sleep(delay)
    assert last_info is not None and last_exc is not None
    raise ProviderUnavailableError(last_info, attempts=attempts, cause=last_exc) from last_exc


async def call_provider_once(client: Any, prompt: str, model: str) -> ProviderCallResult:
    try:
        from weav_ai_runtime import call_llm_provider
    except Exception:
        return await call_provider_legacy(client, prompt, model)
    try:
        result = await call_llm_provider(client, prompt=prompt, model=model, purpose="docode")
    except (AttributeError, TypeError):
        result = await call_provider_legacy(client, prompt, model)
    if isinstance(result, ProviderCallResult):
        return result
    return provider_result_from_runtime_result(result)


def classify_provider_exception(exc: Exception) -> ProviderErrorInfo | None:
    if isinstance(exc, ProviderUnavailableError):
        return exc.info
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    response_text = response_text_or_empty(response)
    detail = str(exc)
    combined = f"{detail}\n{response_text}".lower()
    if status_code in {401, 403} or "401 unauthorized" in combined or "403 forbidden" in combined:
        return ProviderErrorInfo("provider_auth_failed", False, status_code, detail_with_body(detail, response_text))
    if status_code == 404 or "model_not_found" in combined or "model_not_available" in combined or "model not available" in combined:
        return ProviderErrorInfo("model_catalog_mismatch", False, status_code, detail_with_body(detail, response_text))
    if status_code == 429 or "rate limit" in combined or "too many requests" in combined:
        return ProviderErrorInfo("provider_rate_limited", True, status_code, detail_with_body(detail, response_text))
    if "no_upstream_capacity" in combined:
        return ProviderErrorInfo("provider_upstream_unavailable", True, status_code, detail_with_body(detail, response_text))
    if status_code in {408, 409, 425, 500, 502, 503, 504}:
        return ProviderErrorInfo("provider_upstream_unavailable", True, status_code, detail_with_body(detail, response_text))
    if "timeout" in combined or "timed out" in combined:
        return ProviderErrorInfo("provider_timeout", True, status_code, detail_with_body(detail, response_text))
    if any(fragment in combined for fragment in ("connection refused", "connection reset", "server disconnected", "bad gateway", "service unavailable", "gateway timeout")):
        return ProviderErrorInfo("provider_network_error", True, status_code, detail_with_body(detail, response_text))
    return None


def response_text_or_empty(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", "")
    if isinstance(text, str):
        return text
    try:
        return str(text)
    except Exception:
        return ""


def detail_with_body(detail: str, body: str) -> str:
    body = body.strip()
    if not body or body in detail:
        return detail
    return f"{detail}; response={body[:1000]}"


async def call_provider_legacy(client: Any, prompt: str, model: str) -> ProviderCallResult:
    config = provider_completion_config(model)
    if hasattr(client, "acomplete"):
        try:
            response = await client.acomplete(prompt=prompt, model=model)
        except TypeError:
            response = await client.acomplete(prompt, config)
        return provider_call_result(response)
    if hasattr(client, "complete"):
        try:
            response = client.complete(prompt=prompt, model=model)
        except TypeError:
            response = client.complete(prompt, config)
        if hasattr(response, "__await__"):
            response = await response
        return provider_call_result(response)
    if hasattr(client, "achat"):
        try:
            response = await client.achat(messages=[{"role": "user", "content": prompt}], model=model)
        except TypeError:
            response = await client.achat([{"role": "user", "content": prompt}], config)
        return provider_call_result(response)
    if hasattr(client, "chat"):
        try:
            response = client.chat(messages=[{"role": "user", "content": prompt}], model=model)
        except TypeError:
            response = client.chat([{"role": "user", "content": prompt}], config)
        if hasattr(response, "__await__"):
            response = await response
        return provider_call_result(response)
    raise RuntimeError("provider client does not expose a supported chat/completion method")


async def call_provider_text(client: Any, prompt: str, model: str) -> str:
    return (await call_provider(client, prompt, model)).text


def provider_completion_config(model: str) -> Any:
    try:
        from weav_provider_router.base import CompletionConfig
    except Exception:
        return SimpleNamespace(model=model, temperature=0.0)
    return CompletionConfig(model=model, temperature=0.0)


def provider_call_result(response: Any) -> ProviderCallResult:
    runtime_result = normalize_provider_response(response)
    if runtime_result is not None:
        return provider_result_from_runtime_result(runtime_result)

    usage = extract_usage(response)
    text = extract_text(response)
    if text is None:
        text = str(response)
    return ProviderCallResult(
        text=text,
        prompt_tokens=int_or_none(usage.get("prompt_tokens")),
        completion_tokens=int_or_none(usage.get("completion_tokens")),
        total_tokens=int_or_none(usage.get("total_tokens")),
        cost=float_or_none(usage.get("cost")),
        raw=response,
    )


def normalize_provider_response(response: Any) -> Any | None:
    try:
        from weav_ai_runtime import normalize_llm_call_result
    except Exception:
        return None
    return normalize_llm_call_result(response, purpose="docode")


def provider_result_from_runtime_result(response: Any) -> ProviderCallResult:
    usage = getattr(response, "usage", None)
    return ProviderCallResult(
        text=str(getattr(response, "text")),
        prompt_tokens=int_or_none(get_field(usage, "prompt_tokens")),
        completion_tokens=int_or_none(get_field(usage, "completion_tokens")),
        total_tokens=int_or_none(get_field(usage, "tokens") or get_field(usage, "total_tokens")),
        cost=float_or_none(get_field(usage, "cost")),
        tool_calls=list(getattr(response, "tool_calls", []) or []),
        raw=getattr(response, "raw", response),
    )


def create_llm_router() -> Any:
    router_cls = import_llm_router()
    if router_cls is None:
        return LocalLLMRouter()
    try:
        return router_cls()
    except Exception:
        return LocalLLMRouter()


def import_llm_router() -> Any | None:
    try:
        from weav_ai_core import LLMRouter

        return LLMRouter
    except Exception:
        pass
    try:
        from weav_ai_core.llm import LLMRouter

        return LLMRouter
    except Exception:
        return None


def register_provider(router: Any, provider: str, client: Any) -> None:
    if hasattr(router, "register"):
        try:
            router.register(provider, client)
            return
        except TypeError:
            router.register(client)
            return
    if hasattr(router, "providers"):
        router.providers[provider] = client
        return
    raise RuntimeError("LLM router does not expose a supported register method")


def extract_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        direct = first_present(value, "output_text", "text", "content")
        if direct is not None:
            return content_to_text(direct)
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            return extract_text(choices[0])
        message = value.get("message")
        if message is not None:
            return extract_text(message)
        data = value.get("data")
        if data is not None:
            return extract_text(data)
        output = value.get("output")
        if output is not None:
            return content_to_text(output)
        return None

    for attr in ("output_text", "text", "content"):
        if hasattr(value, attr):
            return content_to_text(getattr(value, attr))
    if hasattr(value, "choices"):
        choices = getattr(value, "choices")
        if isinstance(choices, list) and choices:
            return extract_text(choices[0])
    if hasattr(value, "message"):
        return extract_text(getattr(value, "message"))
    return None


def content_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = extract_text(item)
            if text is not None:
                parts.append(text)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        direct = first_present(value, "text", "content", "value")
        return content_to_text(direct) if direct is not None else None
    return str(value)


def extract_usage(value: Any) -> dict[str, object]:
    usage = get_field(value, "usage") or get_field(value, "usage_metadata") or {}
    result: dict[str, object] = {}
    for target, candidates in {
        "prompt_tokens": ("prompt_tokens", "input_tokens", "prompt", "input"),
        "completion_tokens": ("completion_tokens", "output_tokens", "completion", "output"),
        "total_tokens": ("total_tokens", "tokens", "total"),
        "cost": ("cost", "total_cost", "amount"),
    }.items():
        result[target] = first_present(usage, *candidates)
    for target in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        if result.get(target) is None:
            result[target] = get_field(value, target)
    return result


def first_present(value: Any, *keys: str) -> object | None:
    for key in keys:
        candidate = get_field(value, key)
        if candidate is not None:
            return candidate
    return None


def get_field(value: Any, key: str) -> object | None:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, key):
        return getattr(value, key)
    return None


def int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def retry_jitter(delay: float) -> float:
    if delay <= 0:
        return 0.0
    return random.uniform(0.0, min(1.0, delay * 0.2))
