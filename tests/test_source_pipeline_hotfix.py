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
from docode.dobox.types import CommandResult, ToolResult
from docode.storage.models import CodingJob


def crawler_state(*, url: str = "http://127.0.0.1:8765/feed") -> AgentState:
    job = CodingJob(
        id="job-source-progress",
        user_id="test-user",
        instruction=f"Build a crawler from {url}",
        provider="test",
        model="test",
    )
    state = AgentState(job=job)
    state.inspection = SimpleNamespace()
    state.task_contract = SimpleNamespace(must_run_commands=[], must_modify_files=[])
    state.latest_git_status = ToolResult(tool="git_status", output="", exit_code=0)
    state.messages.append(source_result_message(url, controller_owned=True))
    return state


def source_result_message(url: str, *, controller_owned: bool = False, cached: bool = False) -> dict[str, object]:
    payload = {
        "requested_url": url,
        "final_url": url,
        "status_code": 200,
        "execution_scope": "sandbox",
        "mode": "raw",
        "body": '{"items":[],"next_cursor":"next"}',
    }
    return {
        "role": "tool",
        "tool": "inspect_source",
        "exit_code": 0,
        "output": json.dumps(payload),
        "metadata": {
            "requested_url": url,
            "final_url": url,
            "status_code": 200,
            "execution_scope": "sandbox",
            "controller_owned": controller_owned,
            "cached": cached,
        },
    }


class SourcePipelineCoreTests(TestCase):
    def test_package_import_has_no_runtime_patch_state(self) -> None:
        self.assertEqual(docode.__version__, "0.2.0")
        self.assertFalse(hasattr(docode, "__runtime_hotfix_applied__"))
        self.assertFalse(hasattr(loop, "_source_pipeline_hotfix_v1_applied"))

    def test_controller_source_evidence_is_successful_before_edit(self) -> None:
        state = crawler_state()

        evidence = loop.successful_source_inspection(state.messages, state.job.instruction)

        self.assertIsNotNone(evidence)
        self.assertTrue(evidence.controller_owned)

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

    def test_duplicate_url_in_another_mode_is_rejected_before_tool_execution(self) -> None:
        state = crawler_state(url="HTTP://Example.Test:80/feed?cursor=one#section")

        blocked = loop.crawler_external_source_tool_block(
            state,
            "inspect_source",
            {"url": "http://example.test/feed?cursor=one", "mode": "text"},
        )

        self.assertIn("duplicate_source_inspection", blocked)
        self.assertIn("Read or edit", blocked)

    def test_duplicate_feedback_hides_network_tools_but_keeps_read_and_edit_tools(self) -> None:
        state = crawler_state()
        state.add_feedback(
            "duplicate_source_inspection: This source is already available in Source Inspection memory. Read or edit now."
        )
        definitions = [
            ToolDefinition(name, "", {}, AsyncMock())
            for name in (
                "inspect_source",
                "fetch_url",
                "web_search",
                "run_command",
                "read_file",
                "list_files",
                "write_file",
                "edit_file",
                "git_status",
            )
        ]

        selected = {item.name for item in loop.allowed_tool_definitions_for_state(definitions, state)}

        self.assertEqual(state.consecutive_failures, 1)
        self.assertNotIn("inspect_source", selected)
        self.assertNotIn("fetch_url", selected)
        self.assertNotIn("web_search", selected)
        self.assertNotIn("run_command", selected)
        self.assertTrue({"read_file", "list_files", "write_file", "edit_file", "git_status"} <= selected)
        self.assertIsNotNone(loop.successful_source_inspection(state.messages, state.job.instruction))

    def test_two_no_edit_decisions_force_source_to_edit_transition(self) -> None:
        state = crawler_state()
        state.messages.extend(
            [
                {"role": "tool", "tool": "read_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
                {"role": "tool", "tool": "list_files", "exit_code": 0, "metadata": {"path": "."}},
            ]
        )
        definitions = [
            ToolDefinition(name, "", {}, AsyncMock())
            for name in ("inspect_source", "read_file", "write_file", "run_command")
        ]

        selected = {item.name for item in loop.allowed_tool_definitions_for_state(definitions, state)}

        self.assertEqual(selected, {"read_file", "write_file"})

    def test_two_optional_source_urls_force_edit_but_successful_edit_resets_restriction(self) -> None:
        state = crawler_state(url="http://127.0.0.1:8765/feed?cursor=")
        state.task_contract = SimpleNamespace(must_run_commands=["python validate.py"], must_modify_files=[])
        state.messages.extend(
            [
                source_result_message("http://127.0.0.1:8765/feed?cursor=two"),
                source_result_message("http://127.0.0.1:8765/feed?cursor=three"),
            ]
        )
        definitions = [
            ToolDefinition(name, "", {}, AsyncMock())
            for name in ("inspect_source", "read_file", "write_file")
        ]

        before_edit = {item.name for item in loop.allowed_tool_definitions_for_state(definitions, state)}
        state.messages.append({"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}})
        state.latest_git_status = ToolResult(tool="git_status", output=" M crawler.py\n", exit_code=0)
        after_edit = {item.name for item in loop.allowed_tool_definitions_for_state(definitions, state)}

        self.assertNotIn("inspect_source", before_edit)
        self.assertIn("write_file", before_edit)
        self.assertIn("inspect_source", after_edit)

    def test_distinct_same_origin_next_page_is_allowed_within_budget(self) -> None:
        state = crawler_state(url="https://example.test/articles?page=1")

        blocked = loop.crawler_external_source_tool_block(
            state, "inspect_source", {"url": "https://example.test/articles?page=2", "mode": "raw"}
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

    async def test_different_mode_uses_same_canonical_url_response(self) -> None:
        payload = {
            "requested_url": "HTTP://Example.Test:80/feed#fragment",
            "final_url": "http://example.test/feed",
            "status_code": 200,
            "content_type": "application/json",
            "mode": "raw",
            "body": '{"items":[1,2]}',
            "original_bytes": 15,
            "returned_bytes": 15,
            "truncated": False,
        }
        client = SimpleNamespace(run_command=AsyncMock(return_value=CommandResult(output=json.dumps(payload), exit_code=0)))
        tools = DoBoxTools(client, "project-1")

        raw = await tools.inspect_source("HTTP://Example.Test:80/feed#fragment", mode="raw")
        as_json = await tools.inspect_source("http://example.test/feed", mode="json")

        self.assertTrue(raw.ok)
        self.assertTrue(as_json.ok)
        self.assertEqual(client.run_command.await_count, 1)
        self.assertEqual(json.loads(as_json.output)["body"], '{"items":[1,2]}')
        self.assertTrue(as_json.metadata["cached"])
        self.assertFalse(as_json.metadata["network_request_performed"])

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
        from docode.dobox.source_cache import source_structure_summary

        summary = source_structure_summary(
            '<rss xmlns="urn:x" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>x</dc:title></rss>',
            "application/xml",
        )

        self.assertEqual(summary["kind"], "xml")
        self.assertEqual(summary["namespace_prefixes"], ["dc", "default"])


class EvidenceBackedRepairTests(IsolatedAsyncioTestCase):
    def repair_state(self) -> AgentState:
        state = crawler_state(url="http://127.0.0.1:8765/feed")
        state.task_contract = SimpleNamespace(
            must_run_commands=["python collector.py source out.json", "python validate.py"],
            must_modify_files=["collector.py"],
        )
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "write_file",
                    "exit_code": 0,
                    "output": "+ collection = payload.get('items', [])",
                    "metadata": {"path": "collector.py"},
                },
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 0,
                    "output": "wrote 0 records",
                    "metadata": {"command": "python collector.py source out.json"},
                },
            ]
        )
        return state

    def test_zero_record_validator_failure_creates_source_backed_parser_mismatch(self) -> None:
        state = self.repair_state()
        result = ToolResult(
            tool="run_command",
            output="Traceback\nAssertionError: 0",
            exit_code=1,
            metadata={"command": "python validate.py"},
        )

        action = loop.parser_source_mismatch_repair(state, result)

        self.assertIsNotNone(action)
        self.assertEqual(action.failure_class, "parser_source_mismatch")
        self.assertEqual(action.producer_semantic_result, "zero_records")
        self.assertIn("source/parser diagnosis", action.instruction)
        self.assertIn("next_cursor", action.instruction)
        self.assertEqual(action.rerun_commands[0], "python collector.py source out.json")

    async def test_third_identical_parser_mismatch_becomes_non_convergent(self) -> None:
        state = self.repair_state()
        result = ToolResult(
            tool="run_command",
            output="AssertionError: 0",
            exit_code=1,
            metadata={"command": "python validate.py"},
        )
        action = loop.parser_source_mismatch_repair(state, result)
        agent = object.__new__(loop.CodingAgentLoop)
        agent.repository = SimpleNamespace(add_step=AsyncMock())

        await agent.activate_targeted_repair(state, action, result)
        await agent.activate_targeted_repair(state, action, result)
        await agent.activate_targeted_repair(state, action, result)

        self.assertEqual(state.terminal_repair_reason, "repeated_zero_record_parser_failure")
        self.assertEqual(state.failure_signatures[action.signature], 3)

    def test_non_empty_producer_resets_parser_mismatch_convergence(self) -> None:
        state = self.repair_state()
        state.failure_signatures["parser_source_mismatch:collector.py:zero_records"] = 2
        result = ToolResult(
            tool="run_command",
            output="wrote 4 records",
            exit_code=0,
            metadata={"command": "python collector.py source out.json"},
        )

        loop.reset_parser_mismatch_convergence(state, result)

        self.assertEqual(state.failure_signatures, {})

    async def test_invalid_source_tool_feedback_can_be_recorded_without_outer_iteration_increment(self) -> None:
        state = crawler_state()
        agent = object.__new__(loop.CodingAgentLoop)
        agent.repository = SimpleNamespace(add_step=AsyncMock())
        agent.usage_meter = None

        await agent.record_unavailable_tool_requested(
            state,
            requested_tool="inspect_source",
            requested_args={"url": "http://127.0.0.1:8765/feed"},
            available_tools=["read_file", "write_file"],
            workflow_state={"phase": "EDIT_REQUIRED"},
            increment_iteration=False,
            reason="source_inspection_complete_edit_required",
        )

        self.assertEqual(state.iteration, 0)
        self.assertIn("must edit", state.messages[-1]["content"])
