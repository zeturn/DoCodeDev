from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Protocol

try:
    from fastapi import Header, HTTPException
except ModuleNotFoundError:
    def Header(default: object = None, *, alias: str | None = None) -> object:
        _ = alias
        return default

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

from docode.config import DocodeConfig
from docode.storage.models import CodingJob
from docode.storage.repository import JobRepository


@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: str
    tenant: str | None = None
    auth_source: str = "local"
    apicred_access_token: str | None = None


@dataclass(frozen=True, slots=True)
class AuthVerification:
    allowed: bool
    user_id: str | None = None
    tenant: str | None = None
    reason: str = ""
    raw: dict[str, Any] | None = None


class SessionVerifier(Protocol):
    async def verify(
        self,
        *,
        authorization: str | None,
        forwarded_user_id: str | None,
        forwarded_tenant: str | None,
    ) -> AuthVerification: ...


class APICredSessionVerifier:
    def __init__(self, base_url: str, service_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token = service_token
        self.calls: list[dict[str, Any]] = []

    async def verify(
        self,
        *,
        authorization: str | None,
        forwarded_user_id: str | None,
        forwarded_tenant: str | None,
    ) -> AuthVerification:
        payload = {
            "purpose": "docode",
            "authorization": redact_authorization(authorization),
            "forwarded_user_id": forwarded_user_id,
            "forwarded_tenant": forwarded_tenant,
        }
        token = bearer_token(authorization)
        if token:
            payload["access_token"] = token
        try:
            data = await self._post("/runtime/auth/verify", payload)
        except Exception:
            data = await self._post("/auth/verify", payload)
        allowed = bool(data.get("allowed", data.get("authorized", data.get("active", False))))
        return AuthVerification(
            allowed=allowed,
            user_id=str(data.get("user_id") or data.get("sub") or forwarded_user_id or "") or None,
            tenant=str(data.get("tenant") or data.get("tenant_id") or forwarded_tenant or "") or None,
            reason=str(data.get("reason") or ""),
            raw=data,
        )

    async def _post(self, path: str, payload: dict[str, object | None]) -> dict[str, Any]:
        import httpx

        clean_payload = {key: value for key, value in payload.items() if value is not None}
        trace_payload = dict(clean_payload)
        if "access_token" in trace_payload:
            trace_payload["access_token"] = redact_authorization(str(trace_payload["access_token"]))
        self.calls.append({"method": "POST", "path": path, "payload": trace_payload})
        headers = {"Accept": "application/json"}
        if self.service_token:
            headers["Authorization"] = f"Bearer {self.service_token}"
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url}{path}", json=clean_payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}


class BasaltPassSessionVerifier:
    def __init__(self, base_url: str, client_id: str, client_secret: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.calls: list[dict[str, Any]] = []

    async def verify(
        self,
        *,
        authorization: str | None,
        forwarded_user_id: str | None,
        forwarded_tenant: str | None,
    ) -> AuthVerification:
        token = bearer_token(authorization)
        if not token:
            return AuthVerification(allowed=False, reason="token_missing")
        data = await self._introspect(token)
        allowed = bool(data.get("active"))
        client_id = str(data.get("client_id") or "")
        audience = str(data.get("aud") or "")
        if allowed and self.client_id and self.client_id not in {client_id, audience}:
            return AuthVerification(allowed=False, reason="token_invalid")
        return AuthVerification(
            allowed=allowed,
            user_id=str(data.get("sub") or forwarded_user_id or "") or None,
            tenant=str(data.get("tenant_id") or data.get("tenant") or forwarded_tenant or "") or None,
            reason="" if allowed else "token_invalid",
            raw=data,
        )

    async def _introspect(self, token: str) -> dict[str, Any]:
        import httpx

        self.calls.append({"method": "POST", "path": "/api/v1/oauth/introspect", "token": redact_authorization(token)})
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/oauth/introspect",
                data={"token": token},
                auth=(self.client_id, self.client_secret),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}


class UserContextDependency:
    def __init__(self, *, auth_required: bool, verifier: SessionVerifier | None = None) -> None:
        self.auth_required = auth_required
        self.verifier = verifier

    async def __call__(
        self,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
        x_basalt_user_id: Annotated[str | None, Header(alias="X-Basalt-User-ID")] = None,
        x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    ) -> UserContext:
        forwarded_user_id = x_basalt_user_id or x_user_id
        token = bearer_token(authorization)
        if self.verifier is not None and (authorization or self.auth_required):
            try:
                verified = await self.verifier.verify(
                    authorization=authorization,
                    forwarded_user_id=forwarded_user_id,
                    forwarded_tenant=x_tenant_id,
                )
            except Exception as exc:
                if self.auth_required:
                    raise HTTPException(status_code=503, detail="auth verification unavailable") from exc
            else:
                if not verified.allowed:
                    raise HTTPException(status_code=401, detail=verified.reason or "unauthorized")
                if verified.user_id:
                    return UserContext(
                        user_id=verified.user_id,
                        tenant=verified.tenant,
                        auth_source="apicred",
                        apicred_access_token=token,
                    )
                if self.auth_required:
                    raise HTTPException(status_code=401, detail="auth verification missing user")

        if self.auth_required and not forwarded_user_id:
            raise HTTPException(status_code=401, detail="missing authenticated user")
        return UserContext(
            user_id=forwarded_user_id or "local",
            tenant=x_tenant_id,
            auth_source="forwarded" if forwarded_user_id else "local",
            apicred_access_token=token,
        )


async def get_user_context(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    x_basalt_user_id: Annotated[str | None, Header(alias="X-Basalt-User-ID")] = None,
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> UserContext:
    """Read user identity from an upstream auth gateway.

    BasaltPass/APICred should validate the session or service token before
    traffic reaches DoCode and forward a stable user id. Local development can
    still run with a bearer token only, which maps to the `local` user.
    """

    return await UserContextDependency(auth_required=False)(
        authorization=authorization,
        x_user_id=x_user_id,
        x_basalt_user_id=x_basalt_user_id,
        x_tenant_id=x_tenant_id,
    )


def make_user_context_dependency(config: DocodeConfig, verifier: SessionVerifier | None = None):
    if verifier is None and config.basaltpass_enabled:
        verifier = BasaltPassSessionVerifier(config.basaltpass_base_url, config.basaltpass_client_id, config.basaltpass_client_secret)
    dependency = UserContextDependency(
        auth_required=config.auth_required,
        verifier=verifier or APICredSessionVerifier(config.apicred_base_url, config.apicred_token),
    )

    async def _dependency(
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
        x_basalt_user_id: Annotated[str | None, Header(alias="X-Basalt-User-ID")] = None,
        x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    ) -> UserContext:
        return await dependency(
            authorization=authorization,
            x_user_id=x_user_id,
            x_basalt_user_id=x_basalt_user_id,
            x_tenant_id=x_tenant_id,
        )

    return _dependency


async def require_owned_job(repository: JobRepository, job_id: str, user: UserContext) -> CodingJob:
    job = await repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if user.user_id == "local":
        return job
    if job.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return authorization.strip()


def redact_authorization(authorization: str | None) -> str | None:
    token = bearer_token(authorization)
    if not token:
        return None
    return f"bearer:{len(token)}"
