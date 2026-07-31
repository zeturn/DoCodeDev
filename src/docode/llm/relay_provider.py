from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .decision import AgentDecision, format_decision_prompt, parse_decision
from .provider_compat import LocalLLMRouter, ProviderErrorInfo, ProviderUnavailableError
from .usage import LLMUsageMeter


RELAY_SCHEMA_VERSION = "1.0"

# Isolation levels a responder may claim for a returned decision. The harness
# validates these; "contaminated" is debugging-only and must never count toward
# a scored run.
RELAY_ISOLATION_LEVELS = {
    "relay_stateless_isolated",
    "relay_unverified",
    "relay_contaminated",
}


class RelayError(Exception):
    """Base class for relay provider errors."""


class RelayFatal(RelayError):
    """Configuration or security failure: fail closed, never retry."""


class RelayProviderUnavailable(RelayError, ProviderUnavailableError):
    """Transient relay/responder failure; classified as retryable."""

    def __init__(self, category: str = "provider_network_error", message: str = "", retryable: bool = True) -> None:
        info = ProviderErrorInfo(category=category, retryable=retryable, detail=message)
        # Call ProviderUnavailableError.__init__ explicitly: with the dual
        # inheritance (RelayError has no __init__), super() would otherwise
        # resolve to object.__init__ and reject the arguments.
        ProviderUnavailableError.__init__(self, info, attempts=1, cause=self)


class RelayBlocked(RelayError, ProviderUnavailableError):
    """The external responder determined the task cannot be completed.

    Phase 8 wires this into a proper 'blocked' completion path; until then it
    fails the job fast (non-retryable) rather than looping.
    """

    def __init__(self, request_id: str | None = None, content: Any = None) -> None:
        info = ProviderErrorInfo(category="relay_blocked", retryable=False, detail=f"responder reported blocked: {request_id}")
        ProviderUnavailableError.__init__(self, info, attempts=1, cause=self)
        self.request_id = request_id
        self.content = content


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def serialize_tools(tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if hasattr(tool, "name"):
            out.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema(),
                }
            )
        elif isinstance(tool, dict):
            out.append(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "input_schema": tool.get("input_schema") or tool.get("parameters"),
                }
            )
        else:
            raise RelayFatal(f"unsupported tool representation: {type(tool)!r}")
    return out


def compute_request_hash(request: dict[str, Any]) -> str:
    core = {
        "schema_version": request["schema_version"],
        "relay_session_id": request["relay_session_id"],
        "request_id": request["request_id"],
        "turn_index": request["turn_index"],
        "model": request.get("model"),
        "system": request["system"],
        "messages": request["messages"],
        "tools": request["tools"],
        "context": request["context"],
        "prompt": request["prompt"],
    }
    digest = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def write_atomic_json(path: str | os.PathLike[str], data: Any) -> None:
    """Write JSON atomically: temp file -> fsync -> rename (fail-closed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    blob = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def make_relay_response(
    request: dict[str, Any],
    *,
    content: Any,
    finish_reason: str = "stop",
    responder: str = "mock",
    isolation_level: str = "relay_stateless_isolated",
    finish_reason_detail: str | None = None,
) -> dict[str, Any]:
    """Build a response packet with a correct echoed request_hash.

    Used by responders (Codex adapter) and by tests so they never have to
    hand-compute the request_hash.
    """
    body = {"content": content, "finish_reason": finish_reason}
    response_hash = "sha256:" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return {
        "schema_version": request["schema_version"],
        "relay_session_id": request["relay_session_id"],
        "request_id": request["request_id"],
        "responder": responder,
        "isolation_level": isolation_level,
        "content": content,
        "finish_reason": finish_reason,
        "finish_reason_detail": finish_reason_detail,
        "created_at": _now_iso(),
        "request_hash": request["request_hash"],
        "response_hash": response_hash,
    }


@dataclass
class RelayConfig:
    base_dir: str
    requests_dir: str
    responses_dir: str
    session_id: str
    poll_interval: float = 0.5
    timeout_seconds: float = 600.0


def resolve_relay_config(env: dict[str, str] | None = None, *, session_id: str | None = None) -> RelayConfig:
    env = env if env is not None else dict(os.environ)
    base = env.get("DOCODE_RELAY_DIR")
    if not base:
        raise RelayFatal("DOCODE_RELAY_DIR is not set; external relay provider cannot start")
    base = os.path.abspath(base)
    requests_dir = env.get("DOCODE_RELAY_REQUESTS_DIR", os.path.join(base, "requests"))
    responses_dir = env.get("DOCODE_RELAY_RESPONSES_DIR", os.path.join(base, "responses"))
    sid = session_id or env.get("DOCODE_RELAY_SESSION_ID") or "relay-session"
    poll_interval = float(env.get("DOCODE_RELAY_POLL_INTERVAL", "0.5"))
    timeout = float(env.get("DOCODE_RELAY_TIMEOUT", "600"))
    return RelayConfig(
        base_dir=base,
        requests_dir=requests_dir,
        responses_dir=responses_dir,
        session_id=sid,
        poll_interval=poll_interval,
        timeout_seconds=timeout,
    )


class ExternalRelayProvider:
    """A :class:`DecisionLLM` that relays decisions over a shared filesystem.

    The provider is transport-agnostic: it writes a self-contained request
    packet (system/messages/tools/context + a rendered prompt) to
    ``requests_dir`` and waits for a response packet in ``responses_dir``
    produced by a stateless external responder (e.g. a Codex adapter). No model
    API key or provider credential is touched by the runtime itself.

    For tests/in-process use, pass ``responder`` to short-circuit the filesystem
    round-trip with a callable ``request -> response dict``.
    """

    def __init__(
        self,
        *,
        requests_dir: str,
        responses_dir: str,
        session_id: str,
        model: str = "external_relay",
        usage_meter: LLMUsageMeter | None = None,
        poll_interval: float = 0.5,
        timeout_seconds: float = 600.0,
        responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.requests_dir = Path(requests_dir)
        self.responses_dir = Path(responses_dir)
        self.session_id = session_id
        self.model = model
        self.usage_meter = usage_meter or LLMUsageMeter()
        self.poll_interval = float(poll_interval)
        self.timeout_seconds = float(timeout_seconds)
        self.responder = responder
        self._turn = 0
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    async def decide(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[Any],
        context: str,
    ) -> AgentDecision:
        self._turn += 1
        turn_index = self._turn
        request_id = uuid.uuid4().hex
        prompt = format_decision_prompt(system, messages, tools, context)
        request: dict[str, Any] = {
            "schema_version": RELAY_SCHEMA_VERSION,
            "relay_session_id": self.session_id,
            "request_id": request_id,
            "turn_index": turn_index,
            "created_at": _now_iso(),
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": serialize_tools(tools),
            "context": context,
            "prompt": prompt,
        }
        request["request_hash"] = compute_request_hash(request)

        if self.responder is not None:
            response = self.responder(request)
        else:
            write_atomic_json(self.requests_dir / f"{request_id}.json", request)
            response = await self._poll_response(request_id)

        return self._process_response(request, response)

    async def _poll_response(self, request_id: str) -> dict[str, Any]:
        path = self.responses_dir / f"{request_id}.json"
        deadline = time.monotonic() + max(0.0, self.timeout_seconds)
        while True:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    raise RelayFatal(f"relay_response_unreadable:{exc}") from exc
            if time.monotonic() >= deadline:
                raise RelayProviderUnavailable(
                    category="provider_network_error",
                    message=f"relay_response_timeout:{request_id}",
                    retryable=True,
                )
            await asyncio.sleep(self.poll_interval)

    def _process_response(self, request: dict[str, Any], response: Any) -> AgentDecision:
        if not isinstance(response, dict):
            raise RelayFatal("relay_responder_returned_non_object")
        if response.get("request_hash") != request.get("request_hash"):
            # Fail closed on forged/stale/duplicate responses.
            raise RelayFatal("relay_response_request_hash_mismatch")
        content = response.get("content")
        finish = str(response.get("finish_reason") or "stop")
        if finish == "blocked":
            raise RelayBlocked(request_id=request.get("request_id"), content=content)
        if finish == "error":
            raise RelayProviderUnavailable(
                category="provider_response_error",
                message=str(content),
                retryable=True,
            )
        if content is None:
            raise RelayProviderUnavailable(
                category="provider_response_empty",
                message="empty content",
                retryable=True,
            )
        if self.usage_meter is not None:
            self.usage_meter.record_text_call(prompt=request.get("prompt", ""), response=str(content))
        try:
            return parse_decision(str(content))
        except Exception as exc:  # noqa: BLE001
            raise RelayProviderUnavailable(
                category="provider_response_malformed",
                message=f"parse_decision_failed:{exc}",
                retryable=True,
            ) from exc


__all__ = [
    "RELAY_SCHEMA_VERSION",
    "RELAY_ISOLATION_LEVELS",
    "RelayError",
    "RelayFatal",
    "RelayProviderUnavailable",
    "RelayBlocked",
    "ExternalRelayProvider",
    "RelayConfig",
    "resolve_relay_config",
    "make_relay_response",
    "write_atomic_json",
    "compute_request_hash",
    "LocalLLMRouter",
]
