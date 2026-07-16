from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from docode.dobox.client import (
    DoBoxClient,
    DoBoxTransportError,
    build_transport_error,
    classify_transport_error,
    operation_label,
    project_path,
    raise_for_status,
    session_payload_id,
)


class RecordingDoBoxClient(DoBoxClient):
    def __init__(self) -> None:
        super().__init__("http://dobox.example", "token")
        self.requests: list[dict[str, Any]] = []

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        self.requests.append({"method": method, "path": path, "json": json or {}, "timeout": timeout})
        if path.endswith("/exec"):
            return {"output": "ok", "exit_code": 0, "truncated": True}
        if path.endswith("/files/read"):
            return {"content": "partial", "path": "/workspace/large.txt", "file_name": "large.txt", "bytes": 7, "truncated": True}
        if "/git/status" in path:
            return {"status": " M README.md\n", "exit_code": 0}
        if "/git/diff" in path:
            return {"diff": "diff --git a/README.md b/README.md\n+change\n", "exit_code": 0, "truncated": True}
        if "/logs" in path:
            return {"logs": "recent logs\n"}
        return {}


class DoBoxClientTests(IsolatedAsyncioTestCase):
    async def test_run_command_uses_project_exec_endpoint_and_session_payload(self) -> None:
        client = RecordingDoBoxClient()

        result = await client.run_command("project-1", "go test ./...", agent_session_id="42")

        self.assertEqual(result.output, "ok")
        self.assertTrue(result.truncated)
        self.assertEqual(client.requests[0]["method"], "POST")
        self.assertEqual(client.requests[0]["path"], "/api/projects/project-1/exec")
        self.assertEqual(client.requests[0]["json"]["agent_session_id"], 42)
        self.assertEqual(client.requests[0]["json"]["command"], "go test ./...")

    async def test_get_helpers_encode_agent_session_query_params(self) -> None:
        client = RecordingDoBoxClient()

        status = await client.git_status("project-1", agent_session_id="42")
        diff = await client.git_diff("project-1", agent_session_id="42")
        logs = await client.logs("project-1", "tail with spaces", agent_session_id="42")

        self.assertEqual(status.output, " M README.md\n")
        self.assertIn("+change", diff)
        self.assertEqual(logs, "recent logs\n")
        self.assertEqual(client.requests[0]["path"], "/api/projects/project-1/git/status?agent_session_id=42")
        self.assertEqual(client.requests[1]["path"], "/api/projects/project-1/git/diff?agent_session_id=42")
        self.assertEqual(client.requests[2]["path"], "/api/projects/project-1/logs?tail=tail+with+spaces&agent_session_id=42")

    async def test_git_diff_result_preserves_truncation(self) -> None:
        client = RecordingDoBoxClient()

        result = await client.git_diff_result("project-1", agent_session_id="42")

        self.assertIn("+change", result.output)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.truncated)
        self.assertEqual(client.requests[0]["path"], "/api/projects/project-1/git/diff?agent_session_id=42")

    async def test_read_file_preserves_truncation_metadata(self) -> None:
        client = RecordingDoBoxClient()

        result = await client.read_file("project-1", "large.txt", agent_session_id="42")

        self.assertEqual(result.content, "partial")
        self.assertEqual(result.path, "/workspace/large.txt")
        self.assertEqual(result.file_name, "large.txt")
        self.assertEqual(result.bytes_read, 7)
        self.assertTrue(result.truncated)
        self.assertEqual(client.requests[0]["path"], "/api/projects/project-1/files/read")
        self.assertEqual(client.requests[0]["json"]["path"], "large.txt")
        self.assertEqual(client.requests[0]["json"]["agent_session_id"], 42)
        self.assertGreater(client.requests[0]["json"]["max_bytes"], 0)
        self.assertGreater(client.requests[0]["json"]["max_lines"], 0)

    def test_session_payload_id_and_project_path_helpers(self) -> None:
        self.assertEqual(session_payload_id("7"), 7)
        self.assertIsNone(session_payload_id("session-7"))
        self.assertEqual(project_path("p1", "/artifacts/archive/", agent_session_id="7"), "/api/projects/p1/artifacts/archive?agent_session_id=7")

    def test_http_error_includes_response_body(self) -> None:
        class Response:
            is_error = True
            status_code = 500
            text = '{"error":"sandbox image missing"}'

        with self.assertRaises(RuntimeError) as raised:
            raise_for_status(Response(), "POST", "/api/projects")

        self.assertIn("POST /api/projects failed with HTTP 500", str(raised.exception))
        self.assertIn("sandbox image missing", str(raised.exception))


class TransportErrorContextTests(IsolatedAsyncioTestCase):
    def test_classify_transport_error_kinds(self) -> None:
        self.assertEqual(classify_transport_error(httpx.RemoteProtocolError("x")), "protocol")
        self.assertEqual(classify_transport_error(httpx.ConnectError("x")), "connect")
        self.assertEqual(classify_transport_error(httpx.WriteError("x")), "write")
        self.assertEqual(classify_transport_error(httpx.ReadError("x")), "read")

    def test_operation_label_maps_known_endpoints(self) -> None:
        self.assertEqual(
            operation_label("GET", "/api/projects/1/artifacts/archive?agent_session_id=2"),
            "archive_workspace",
        )
        self.assertEqual(operation_label("POST", "/api/projects/1/files/write"), "write_file")
        self.assertEqual(operation_label("POST", "/api/projects/1/exec"), "run_command")
        self.assertEqual(operation_label("POST", "/api/projects"), "create_project")

    def test_build_transport_error_has_context_without_secrets(self) -> None:
        exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
        err = build_transport_error(exc, method="GET", path="/api/projects/1/artifacts/archive")

        self.assertIsInstance(err, DoBoxTransportError)
        self.assertEqual(err.operation, "archive_workspace")
        self.assertEqual(err.kind, "protocol")
        self.assertEqual(err.method, "GET")
        self.assertEqual(err.path, "/api/projects/1/artifacts/archive")
        self.assertEqual(err.exception_type, "RemoteProtocolError")
        message = str(err)
        self.assertIn("archive_workspace", message)
        self.assertIn("protocol", message)
        self.assertIn("Server disconnected", message)
        self.assertNotIn("token", message.lower())
        self.assertNotIn("authorization", message.lower())

    async def test_request_wraps_transport_error_with_context(self) -> None:
        client = DoBoxClient("http://dobox.example", "super-secret-token")

        class BoomClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "BoomClient":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

            async def request(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

        with patch("httpx.AsyncClient", BoomClient):
            with self.assertRaises(DoBoxTransportError) as raised:
                await client._request("POST", "/api/projects/1/files/write", json={"content": "sensitive-file-body"})

        err = raised.exception
        self.assertEqual(err.operation, "write_file")
        self.assertEqual(err.kind, "protocol")
        self.assertEqual(err.method, "POST")
        message = str(err)
        self.assertNotIn("super-secret-token", message)
        self.assertNotIn("sensitive-file-body", message)

    async def test_archive_workspace_wraps_transport_error(self) -> None:
        client = DoBoxClient("http://dobox.example", "super-secret-token")

        class BoomClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "BoomClient":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

            async def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

        with patch("httpx.AsyncClient", BoomClient):
            with self.assertRaises(DoBoxTransportError) as raised:
                await client.archive_workspace("1", agent_session_id="2")

        err = raised.exception
        self.assertEqual(err.operation, "archive_workspace")
        self.assertEqual(err.kind, "protocol")
        self.assertNotIn("super-secret-token", str(err))
