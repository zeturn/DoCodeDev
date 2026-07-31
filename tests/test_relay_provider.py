from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
import unittest.mock as mock
from pathlib import Path

from docode.llm.decision import AgentDecision
from docode.llm.relay_provider import (
    ExternalRelayProvider,
    RelayBlocked,
    RelayFatal,
    RelayProviderUnavailable,
    RelayConfig,
    compute_request_hash,
    make_relay_response,
    resolve_relay_config,
    serialize_tools,
    write_atomic_json,
)
from docode.llm.relay_responder import RelayResponder

TOOL_CALL_JSON = (
    '{"type":"tool_call","tool_name":"read_file","args":{"path":"a.txt"},"reason":"inspect"}'
)
FINAL_JSON = (
    '{"type":"final_candidate","summary":"done","verification":"","no_test_reason":null,'
    '"remaining_risks":[]}'
)


class FakeTool:
    def __init__(self, name: str, description: str, schema: dict) -> None:
        self.name = name
        self.description = description
        self._schema = schema

    def input_schema(self) -> dict:
        return self._schema


class RelayProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="relay-test-")
        self.req = Path(self.tmp) / "requests"
        self.resp = Path(self.tmp) / "responses"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _base_provider(self, **kwargs) -> ExternalRelayProvider:
        return ExternalRelayProvider(
            requests_dir=str(self.req),
            responses_dir=str(self.resp),
            session_id="sess-1",
            poll_interval=0.02,
            timeout_seconds=5,
            **kwargs,
        )

    def _fake_responder(self, content, finish_reason="stop", tamper_hash=False):
        def responder(request):
            response = make_relay_response(request, content=content, finish_reason=finish_reason)
            if tamper_hash:
                response["request_hash"] = "sha256:deadbeef"
            return response

        return responder

    async def test_in_process_tool_call(self) -> None:
        provider = self._base_provider(responder=self._fake_responder(TOOL_CALL_JSON))
        decision = await provider.decide(system="s", messages=[], tools=[], context="c")
        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.type, "tool_call")
        self.assertEqual(decision.tool_name, "read_file")

    async def test_in_process_final_candidate(self) -> None:
        provider = self._base_provider(responder=self._fake_responder(FINAL_JSON))
        decision = await provider.decide(system="s", messages=[], tools=[], context="c")
        self.assertEqual(decision.type, "final_candidate")
        self.assertEqual(decision.summary, "done")

    async def test_malformed_content_raises_retryable(self) -> None:
        provider = self._base_provider(responder=self._fake_responder("not-json"))
        with self.assertRaises(RelayProviderUnavailable):
            await provider.decide(system="s", messages=[], tools=[], context="c")

    async def test_responder_error_finish_raises_retryable(self) -> None:
        provider = self._base_provider(responder=self._fake_responder("boom", finish_reason="error"))
        with self.assertRaises(RelayProviderUnavailable):
            await provider.decide(system="s", messages=[], tools=[], context="c")

    async def test_blocked_finish_raises_relay_blocked(self) -> None:
        provider = self._base_provider(responder=self._fake_responder(None, finish_reason="blocked"))
        with self.assertRaises(RelayBlocked):
            await provider.decide(system="s", messages=[], tools=[], context="c")

    async def test_request_hash_mismatch_fails_closed(self) -> None:
        provider = self._base_provider(responder=self._fake_responder(FINAL_JSON, tamper_hash=True))
        with self.assertRaises(RelayFatal):
            await provider.decide(system="s", messages=[], tools=[], context="c")

    async def test_filesystem_roundtrip_via_thread(self) -> None:
        provider = self._base_provider()  # filesystem polling, no in-process responder

        def background():
            for _ in range(500):
                pending = [
                    p
                    for p in self.req.glob("*.json")
                    if not p.name.endswith((".processing.json", ".tmp"))
                ]
                if pending:
                    request = json.loads(pending[0].read_text(encoding="utf-8"))
                    write_atomic_json(
                        self.resp / f"{request['request_id']}.json",
                        make_relay_response(request, content=TOOL_CALL_JSON, finish_reason="stop"),
                    )
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=background, daemon=True)
        thread.start()
        decision = await provider.decide(system="s", messages=[], tools=[], context="c")
        thread.join(timeout=5)
        self.assertEqual(decision.type, "tool_call")
        # The request file must have been written by the provider.
        self.assertTrue(any(self.req.glob("*.json")))

    async def test_build_relay_runtime_wires_provider(self) -> None:
        import os

        from docode.llm.relay_provider import ExternalRelayProvider
        from docode.llm.runtime_builder import build_docode_runtime
        from docode.storage.models import CodingJob

        os.environ["DOCODE_RELAY_DIR"] = self.tmp
        job = CodingJob(
            id="job-relay",
            user_id="u",
            instruction="do",
            provider="external_relay",
            model="external_relay",
        )

        class FakeResolver:
            proxy_active = False

        runtime = await build_docode_runtime(job, FakeResolver())
        self.assertEqual(runtime.provider, "external_relay")
        self.assertIsInstance(runtime.llm, ExternalRelayProvider)
        # No APICred client: verifier_judge/reviewer must stay disabled.
        self.assertIsNone(runtime.provider_client)

    def test_serialize_tools_duck_typed_and_dict(self) -> None:
        tools = [
            FakeTool("read_file", "reads", {"type": "object"}),
            {"name": "write_file", "description": "writes", "input_schema": {"type": "object"}},
        ]
        serialized = serialize_tools(tools)
        self.assertEqual(serialized[0]["name"], "read_file")
        self.assertEqual(serialized[1]["name"], "write_file")
        self.assertEqual(serialized[1]["input_schema"], {"type": "object"})

    def test_serialize_tools_rejects_unknown(self) -> None:
        with self.assertRaises(RelayFatal):
            serialize_tools([object()])

    def test_request_hash_stable_and_content_sensitive(self) -> None:
        base = {
            "schema_version": "1.0",
            "relay_session_id": "s",
            "request_id": "r",
            "turn_index": 1,
            "model": "external_relay",
            "system": "sys",
            "messages": [],
            "tools": [],
            "context": "ctx",
            "prompt": "prompt",
        }
        h1 = compute_request_hash(base)
        self.assertEqual(h1, compute_request_hash(dict(base)))
        mutated = dict(base, context="ctx2")
        self.assertNotEqual(h1, compute_request_hash(mutated))


class RelayConfigTests(unittest.TestCase):
    def test_missing_relay_dir_raises(self) -> None:
        with self.assertRaises(RelayFatal):
            resolve_relay_config({})

    def test_explicit_dir_resolves_subdirs(self) -> None:
        import os

        base = os.path.abspath("/tmp/relay")
        cfg = resolve_relay_config({"DOCODE_RELAY_DIR": base}, session_id="job-9")
        self.assertIsInstance(cfg, RelayConfig)
        self.assertEqual(cfg.session_id, "job-9")
        self.assertEqual(cfg.requests_dir, os.path.join(base, "requests"))
        self.assertEqual(cfg.responses_dir, os.path.join(base, "responses"))


class RelayResponderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="relay-responder-")
        self.req = Path(self.tmp) / "requests"
        self.resp = Path(self.tmp) / "responses"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sample_request(self) -> dict:
        request = {
            "schema_version": "1.0",
            "relay_session_id": "s",
            "request_id": "r1",
            "turn_index": 1,
            "created_at": "now",
            "model": "external_relay",
            "system": "sys",
            "messages": [],
            "tools": [],
            "context": "ctx",
            "prompt": "prompt",
        }
        request["request_hash"] = compute_request_hash(request)
        return request

    def test_mock_backend_writes_final_candidate(self) -> None:
        responder = RelayResponder(str(self.req), str(self.resp), backend="mock", once=True)
        write_atomic_json(self.req / "r1.json", self._sample_request())
        self.assertTrue(responder.process_one())
        response_path = self.resp / "r1.json"
        self.assertTrue(response_path.exists())
        response = json.loads(response_path.read_text(encoding="utf-8"))
        self.assertEqual(response["finish_reason"], "stop")
        self.assertEqual(response["isolation_level"], "relay_stateless_isolated")
        decision = json.loads(response["content"])
        self.assertEqual(decision["type"], "final_candidate")

    def test_no_pending_request_returns_false(self) -> None:
        responder = RelayResponder(str(self.req), str(self.resp), backend="mock", once=True)
        self.assertFalse(responder.process_one())

    def test_codex_backend_builds_command(self) -> None:
        with mock.patch("docode.llm.relay_responder.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=FINAL_JSON, stderr="")
            responder = RelayResponder(
                str(self.req),
                str(self.resp),
                backend="codex",
                model="gpt-x",
                codex_extra_args=["--foo"],
            )
            out = responder._codex_content(self._sample_request())
            self.assertEqual(out.strip(), FINAL_JSON)
            args = run.call_args[0][0]
            self.assertEqual(args[0], "codex")
            self.assertIn("exec", args)
            self.assertIn("-m", args)
            self.assertIn("gpt-x", args)
            self.assertIn("--foo", args)

    def test_codex_backend_failure_returns_error_response(self) -> None:
        with mock.patch("docode.llm.relay_responder.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=1, stdout="", stderr="nope")
            responder = RelayResponder(str(self.req), str(self.resp), backend="codex")
            response = responder._answer(self._sample_request())
            self.assertEqual(response["finish_reason"], "error")


if __name__ == "__main__":
    unittest.main()
