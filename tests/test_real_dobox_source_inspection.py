from __future__ import annotations

import asyncio
import json
import os
from unittest import IsolatedAsyncioTestCase, skipUnless

from docode.config import load_config
from docode.dobox.client import DoBoxClient
from docode.dobox.tools import DoBoxTools
from docode.runtime.smoke import ensure_dobox_smoke_token
from docode.storage.models import new_id


FIXTURE_SERVER = '''from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/feed?cursor=next")
            self.end_headers()
            return
        body = json.dumps({"path": self.path, "items": [{"id": 1}, {"id": 2}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", 18765), Handler).serve_forever()
'''


@skipUnless(os.getenv("DOCODE_REAL_DOBOX_SMOKE") == "1", "set DOCODE_REAL_DOBOX_SMOKE=1 to run the real DoBox smoke")
class RealDoBoxSourceInspectionTests(IsolatedAsyncioTestCase):
    async def test_inspect_source_reaches_local_fixture_inside_project_sandbox(self) -> None:
        config = load_config()
        token, token_check = await ensure_dobox_smoke_token(config)
        self.assertEqual(token_check.status, "passed", token_check.detail)
        self.assertIsNotNone(token)
        client = DoBoxClient(config.dobox_base_url, token or "")
        project = await client.create_project(
            name=f"docode-source-inspection-{new_id('smoke')}",
            network_mode="project",
        )
        session = await client.create_agent_session(project.project_id, "source-inspection-smoke")
        server_task = None
        try:
            await client.write_file(project.project_id, "source_server.py", FIXTURE_SERVER, agent_session_id=session.session_id)
            server_task = asyncio.create_task(
                client.run_command(
                    project.project_id,
                    ["python3", "source_server.py"],
                    timeout_sec=20,
                    agent_session_id=session.session_id,
                )
            )
            await asyncio.sleep(0.75)
            result = await DoBoxTools(client, project.project_id, agent_session_id=session.session_id).inspect_source(
                "http://127.0.0.1:18765/redirect",
                mode="json",
                max_bytes=4_096,
                timeout=5,
            )
            payload = json.loads(result.output)

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(payload["status_code"], 200)
            self.assertTrue(payload["final_url"].endswith("/feed?cursor=next"))
            self.assertEqual(json.loads(payload["body"])["path"], "/feed?cursor=next")
            self.assertEqual(result.metadata["execution_scope"], "sandbox")
        finally:
            if server_task is not None:
                server_task.cancel()
                try:
                    await server_task
                except BaseException:
                    pass
            await client.delete_project(project.project_id)
