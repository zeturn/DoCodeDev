from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from docode.api.auth import APICredSessionVerifier, AuthVerification, HTTPException, UserContextDependency, bearer_token, redact_authorization


class FakeVerifier:
    def __init__(self, result: AuthVerification | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, object | None]] = []

    async def verify(self, *, authorization: str | None, forwarded_user_id: str | None, forwarded_tenant: str | None) -> AuthVerification:
        self.calls.append(
            {
                "authorization": authorization,
                "forwarded_user_id": forwarded_user_id,
                "forwarded_tenant": forwarded_tenant,
            }
        )
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeAPICredSessionVerifier(APICredSessionVerifier):
    def __init__(self) -> None:
        super().__init__("https://apicred.invalid/v1", "service-secret")
        self.posts: list[tuple[str, dict[str, object | None]]] = []

    async def _post(self, path: str, payload: dict[str, object | None]):
        self.posts.append((path, payload))
        if path == "/runtime/auth/verify":
            raise RuntimeError("missing")
        return {"allowed": True, "user_id": "verified-user", "tenant_id": "tenant-1"}


class AuthDependencyTests(IsolatedAsyncioTestCase):
    async def test_required_auth_uses_verified_identity_over_forwarded_header(self) -> None:
        verifier = FakeVerifier(AuthVerification(allowed=True, user_id="verified-user", tenant="tenant-1"))
        dependency = UserContextDependency(auth_required=True, verifier=verifier)

        user = await dependency(
            authorization="Bearer user-token",
            x_user_id="spoofed-user",
            x_basalt_user_id=None,
            x_tenant_id="forwarded-tenant",
        )

        self.assertEqual(user.user_id, "verified-user")
        self.assertEqual(user.tenant, "tenant-1")
        self.assertEqual(user.auth_source, "apicred")
        self.assertEqual(verifier.calls[0]["forwarded_user_id"], "spoofed-user")

    async def test_required_auth_rejects_missing_identity(self) -> None:
        dependency = UserContextDependency(auth_required=True, verifier=None)

        with self.assertRaises(HTTPException) as raised:
            await dependency(authorization=None, x_user_id=None, x_basalt_user_id=None, x_tenant_id=None)

        self.assertEqual(raised.exception.status_code, 401)

    async def test_denied_verification_rejects_request(self) -> None:
        dependency = UserContextDependency(auth_required=True, verifier=FakeVerifier(AuthVerification(allowed=False, reason="expired")))

        with self.assertRaises(HTTPException) as raised:
            await dependency(authorization="Bearer expired", x_user_id=None, x_basalt_user_id=None, x_tenant_id=None)

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail, "expired")

    async def test_optional_auth_falls_back_to_forwarded_identity_when_verifier_unavailable(self) -> None:
        dependency = UserContextDependency(auth_required=False, verifier=FakeVerifier(RuntimeError("apicred unavailable")))

        user = await dependency(
            authorization="Bearer token",
            x_user_id=None,
            x_basalt_user_id="basalt-user",
            x_tenant_id="tenant-1",
        )

        self.assertEqual(user.user_id, "basalt-user")
        self.assertEqual(user.auth_source, "forwarded")

    async def test_apicred_verifier_falls_back_to_auth_verify_endpoint(self) -> None:
        verifier = FakeAPICredSessionVerifier()

        result = await verifier.verify(
            authorization="Bearer user-token",
            forwarded_user_id="forwarded-user",
            forwarded_tenant="tenant-1",
        )

        self.assertTrue(result.allowed)
        self.assertEqual(result.user_id, "verified-user")
        self.assertEqual([path for path, _payload in verifier.posts], ["/runtime/auth/verify", "/auth/verify"])


class AuthHelperTests(TestCase):
    def test_bearer_token_and_redaction(self) -> None:
        self.assertEqual(bearer_token("Bearer secret-token"), "secret-token")
        self.assertEqual(bearer_token("raw-token"), "raw-token")
        self.assertEqual(redact_authorization("Bearer secret-token"), "bearer:12")
