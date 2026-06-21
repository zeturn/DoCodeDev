from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase

from docode.dobox.client import DoBoxClient, project_path, raise_for_status, session_payload_id


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
        self.assertEqual(client.requests[0]["json"], {"path": "large.txt", "agent_session_id": 42})

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
