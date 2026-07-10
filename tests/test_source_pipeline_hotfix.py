from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock

import docode
from docode.agent import loop
from docode.agent.state import AgentState
from docode.dobox.tools import ToolDefinition
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import CommandResult
from docode.storage.models import CodingJob


class SourcePipelineHotfixImportTests(TestCase):
    def test_hotfix_is_installed(self) -> None:
        self.assertTrue(getattr(loop, "_source_pipeline_hotfix_v1_applied", False), getattr(docode, "__runtime_hotfix_error__", None))
        self.assertTrue(getattr(docode, "__runtime_hotfix_applied__", False), getattr(docode, "__runtime_hotfix_error__", None))
        self.assertIsNone(getattr(docode, "__runtime_hotfix_error__", None))

    def test_same_origin_derived_source_is_allowed(self) -> None:
        state = SimpleNamespace(
            job=SimpleNamespace(
                instruction=(
                    "Build a cursor collector from http://127.0.0.1:8765/orbit/measurements?cursor=. "
                    "Verification commands:\npython tool.py"
                )
            ),
            messages=[
                {
                    "role": "tool",
                    "tool": "inspect_source",
                    "exit_code": 0,
                    "metadata": {
                        "requested_url": "http://127.0.0.1:8765/orbit/measurements?cursor=",
                        "final_url": "http://127.0.0.1:8765/orbit/measurements?cursor=",
                    },
                }
            ],
        )

        blocked = loop.crawler_external_source_tool_block(
            state,
            "inspect_source",
            {"url": "http://127.0.0.1:8765/orbit/measurements?cursor=next-2"},
        )

        self.assertEqual(blocked, "")

    def test_cross_origin_and_invalid_port_sources_are_blocked(self) -> None:
        state = SimpleNamespace(
            job=SimpleNamespace(instruction="Build a collector from http://127.0.0.1:8765/feed"),
            messages=[],
        )

        cross_origin = loop.crawler_external_source_tool_block(
            state, "inspect_source", {"url": "http://127.0.0.1:9999/other"}
        )
        invalid_port = loop.crawler_external_source_tool_block(
            state, "inspect_source", {"url": "http://127.0.0.1:not-a-port/other"}
        )

        self.assertIn("same source origin", cross_origin)
        self.assertIn("same source origin", invalid_port)

    def test_xml_namespace_in_written_code_is_not_treated_as_source_drift(self) -> None:
        state = SimpleNamespace(
            job=SimpleNamespace(instruction="Build an RSS collector from http://127.0.0.1:8765/feed.xml"),
            messages=[],
        )

        blocked = loop.crawler_external_source_tool_block(
            state,
            "write_file",
            {
                "path": "feed_reader.py",
                "content": "DC = '{http://purl.org/dc/elements/1.1/}creator'\n",
            },
        )

        self.assertEqual(blocked, "")


class InspectSourceCacheTests(IsolatedAsyncioTestCase):
    async def test_duplicate_source_inspection_uses_cache(self) -> None:
        payload = {
            "requested_url": "http://127.0.0.1:8765/feed",
            "final_url": "http://127.0.0.1:8765/feed",
            "status_code": 200,
            "content_type": "application/json",
            "mode": "raw",
            "body": json.dumps({"items": [{"id": 1}], "next_cursor": "next-2"}),
            "original_bytes": 42,
            "returned_bytes": 42,
            "truncated": False,
            "body_encoding": "utf-8",
        }
        client = SimpleNamespace(run_command=AsyncMock(return_value=CommandResult(output=json.dumps(payload), exit_code=0)))
        tools = DoBoxTools(client, "project-1", agent_session_id="session-1")

        first = await tools.inspect_source("http://127.0.0.1:8765/feed", mode="raw")
        second = await tools.inspect_source("http://127.0.0.1:8765/feed", mode="raw")

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(client.run_command.await_count, 1)
        self.assertFalse(first.metadata.get("cached"))
        self.assertTrue(second.metadata.get("cached"))
        self.assertFalse(second.metadata.get("network_request_performed"))
        self.assertIn("pagination_fields", first.metadata.get("structure_summary", {}))
        self.assertNotIn('"body"', second.output)

    async def test_different_same_origin_url_performs_another_request(self) -> None:
        def response(*args: object, **kwargs: object) -> CommandResult:
            config = json.loads(str(args[1][-1]))
            payload = {
                "requested_url": config["url"],
                "final_url": config["url"],
                "status_code": 200,
                "content_type": "application/json",
                "mode": "raw",
                "body": "{}",
                "original_bytes": 2,
                "returned_bytes": 2,
                "truncated": False,
            }
            return CommandResult(output=json.dumps(payload), exit_code=0)

        client = SimpleNamespace(run_command=AsyncMock(side_effect=response))
        tools = DoBoxTools(client, "project-1", agent_session_id="session-1")

        await tools.inspect_source("http://127.0.0.1:8765/feed")
        await tools.inspect_source("http://127.0.0.1:8765/feed?cursor=next")

        self.assertEqual(client.run_command.await_count, 2)

    async def test_summary_caps_and_redacts_pagination_values(self) -> None:
        payload = {
            "requested_url": "http://127.0.0.1:8765/feed",
            "final_url": "http://127.0.0.1:8765/feed",
            "status_code": 200,
            "content_type": "application/json",
            "mode": "raw",
            "body": json.dumps({"next_cursor": "x" * 500, "next_access_token": "do-not-copy"}),
            "returned_bytes": 550,
            "truncated": False,
        }
        client = SimpleNamespace(run_command=AsyncMock(return_value=CommandResult(output=json.dumps(payload), exit_code=0)))
        tools = DoBoxTools(client, "project-1")

        result = await tools.inspect_source("http://127.0.0.1:8765/feed")
        pagination = result.metadata["structure_summary"]["pagination_fields"]

        self.assertLessEqual(len(pagination["next_cursor"]), 214)
        self.assertEqual(pagination["next_access_token"], "[redacted]")


class VerificationOrderHotfixTests(TestCase):
    def test_controller_restarts_explicit_plan_after_repair_edit(self) -> None:
        producer = "python build_output.py"
        validator = "python validate_output.py"
        job = CodingJob(
            id="job-hotfix-order",
            user_id="test-user",
            instruction="Repair the collector. Verification commands:\n1. python build_output.py\n2. python validate_output.py",
            provider="test",
            model="test",
        )
        state = AgentState(job=job)
        state.task_contract = SimpleNamespace(must_run_commands=[producer, validator], must_modify_files=[])
        state.inspection = SimpleNamespace()
        state.repair_mode = "targeted_repair"
        state.active_repair_started_at = 0
        state.active_repair_action = {"target_files": ["collector.py"], "rerun_commands": [validator]}
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "edit_file",
                    "exit_code": 0,
                    "metadata": {"path": "collector.py"},
                }
            ]
        )
        state.latest_git_status = SimpleNamespace(output=" M collector.py\n")
        snapshot = SimpleNamespace(phase=loop.WorkflowPhase.TEST_REQUIRED, diff_exists=True)

        command = loop.controller_owned_required_command(state, snapshot)

        self.assertEqual(command, producer)

    def test_targeted_repair_can_inspect_same_origin_cursor_while_tests_are_missing(self) -> None:
        job = CodingJob(
            id="job-hotfix-cursor",
            user_id="test-user",
            instruction="Repair the crawler from http://127.0.0.1:8765/feed. Verification commands:\npython validate.py",
            provider="test",
            model="test",
        )
        state = AgentState(job=job)
        state.inspection = SimpleNamespace()
        state.task_contract = SimpleNamespace(must_run_commands=["python validate.py"], must_modify_files=[])
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {"target_files": ["crawler.py"], "rerun_commands": ["python validate.py"]}
        state.latest_git_status = SimpleNamespace(output=" M crawler.py\n")
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "inspect_source",
                    "exit_code": 0,
                    "output": json.dumps(
                        {
                            "requested_url": "http://127.0.0.1:8765/feed",
                            "final_url": "http://127.0.0.1:8765/feed",
                            "status_code": 200,
                            "execution_scope": "sandbox",
                            "mode": "raw",
                            "body": "page one",
                        }
                    ),
                    "metadata": {"requested_url": "http://127.0.0.1:8765/feed", "execution_scope": "sandbox"},
                },
                {"role": "tool", "tool": "edit_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
            ]
        )
        snapshot = loop.workflow_snapshot(state, state.latest_git_status.output)
        definition = ToolDefinition("inspect_source", "", {}, AsyncMock())

        selected = loop.allowed_tool_definitions_for_state([definition], state)

        self.assertEqual([item.name for item in selected], ["inspect_source"])
        self.assertEqual(loop.repair_mode_tool_block(state, "inspect_source"), "")
        self.assertEqual(loop.required_test_tool_block(state, snapshot, "inspect_source", {}), "")

    def test_non_crawler_controller_behavior_is_unchanged(self) -> None:
        command = "python - <<'PY'\nprint('ok')\nPY"
        job = CodingJob(
            id="job-hotfix-noncrawler",
            user_id="test-user",
            instruction="Update a utility. Verification commands:\n" + command,
            provider="test",
            model="test",
        )
        state = AgentState(job=job)
        state.inspection = SimpleNamespace()
        state.task_contract = SimpleNamespace(must_run_commands=[command], must_modify_files=[])
        state.latest_git_status = SimpleNamespace(output=" M utility.py\n")
        state.messages.append({"role": "tool", "tool": "edit_file", "exit_code": 0, "metadata": {"path": "utility.py"}})
        snapshot = loop.workflow_snapshot(state, state.latest_git_status.output)

        self.assertEqual(loop.controller_owned_required_command(state, snapshot), command)

    def test_mixed_xml_namespaces_summary_does_not_crash(self) -> None:
        from docode._runtime_hotfix_source_pipeline import _source_structure_summary

        summary = _source_structure_summary(
            '<rss xmlns="urn:x" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>x</dc:title></rss>',
            "application/xml",
        )

        self.assertEqual(summary["kind"], "xml")
        self.assertEqual(summary["namespace_prefixes"], ["dc", "default"])
