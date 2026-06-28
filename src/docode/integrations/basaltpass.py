from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class CachedExchange:
    subject_token: str
    access_token: str
    expires_at: float
    resource: str
    scope: str


class BasaltPassTokenExchangeClient:
    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self._cache: dict[tuple[str, str, str], CachedExchange] = {}

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.client_id and self.client_secret)

    async def exchange(self, *, subject_token: str | None, resource: str, scope: str) -> str | None:
        if not self.configured or not subject_token:
            return None
        key = (subject_token, resource, scope)
        cached = self._cache.get(key)
        if cached and time.time() < cached.expires_at:
            return cached.access_token

        form = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "resource": resource,
            "scope": scope,
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/oauth/token",
                data=form,
                auth=(self.client_id, self.client_secret),
                headers={"Accept": "application/json"},
            )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        token = str(payload.get("access_token") or "")
        if not token:
            raise RuntimeError("BasaltPass token exchange returned no access_token")
        expires_in = int(payload.get("expires_in") or 300)
        self._cache[key] = CachedExchange(
            subject_token=subject_token,
            access_token=token,
            expires_at=time.time() + max(expires_in - 30, 30),
            resource=resource,
            scope=str(payload.get("scope") or scope),
        )
        return token
