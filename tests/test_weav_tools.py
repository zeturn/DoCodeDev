from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from docode.agent.tools import CompositeAgentTools
from docode.agent.weav_tools import LocalToolRegistry, build_agent_tool_registry, weav_output_to_docode_result
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import CommandResult
from docode.web.tools import WebTools, WebToolsConfig


class FakeDoBoxClient:
    async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
        _ = project_id, command, cwd, timeout_sec, output_limit, agent_session_id
        return CommandResult("ok\n", 0)

    async def read_file(self, project_id, path, agent_session_id=None):
        _ = project_id, path, agent_session_id
        return "content"

    async def write_file(self, project_id, path, content, agent_session_id=None):
        _ = project_id, path, content, agent_session_id

    async def list_files(self, project_id, path=".", agent_session_id=None):
        _ = project_id, path, agent_session_id
        return CommandResult("README.md\n", 0)

    async def search(self, project_id, query, path=".", agent_session_id=None):
        _ = project_id, query, path, agent_session_id
        return CommandResult("", 1)

    async def git_status(self, project_id, agent_session_id=None):
        _ = project_id, agent_session_id
        return CommandResult("", 0)

    async def git_diff(self, project_id, agent_session_id=None):
        _ = project_id, agent_session_id
        return ""

    async def git_commit(self, project_id, message, agent_session_id=None):
        _ = project_id, message, agent_session_id
        return CommandResult("[main abc] commit\n", 0)

    async def preview(self, project_id, port, agent_session_id=None):
        _ = project_id, agent_session_id
        return {"url": f"https://preview.example/{port}"}

    async def logs(self, project_id, tail="200", agent_session_id=None):
        _ = project_id, tail, agent_session_id
        return ""


class WeavToolAdapterTests(IsolatedAsyncioTestCase):
    async def test_build_agent_tool_registry_includes_composite_tools(self) -> None:
        dobox_tools = DoBoxTools(FakeDoBoxClient(), "project-1")
        web_tools = WebTools(WebToolsConfig(openai_api_key=""))
        composite = CompositeAgentTools(dobox_tools, web_tools)

        registry = build_agent_tool_registry(composite, LocalToolRegistry())

        self.assertIsNotNone(registry.get("run_command"))
        self.assertIsNotNone(registry.get("inspect_source"))
        self.assertIsNotNone(registry.get("edit_file"))
        self.assertIsNotNone(registry.get("apply_patch"))
        self.assertIsNotNone(registry.get("fetch_url"))
        self.assertIsNotNone(registry.get("web_search"))

        output = await registry.call("run_command", {"command": "true"})
        self.assertTrue(output["ok"])
        self.assertEqual(output["content"], "ok\n")


class WeavToolOutputAdapterTests(TestCase):
    def test_weav_output_to_docode_result_accepts_dict_shape(self) -> None:
        result = weav_output_to_docode_result(
            {
                "tool": "run_command",
                "ok": False,
                "exit_code": 2,
                "content": "failed",
                "truncated": True,
                "metadata": {"reason": "bad_input"},
            }
        )

        self.assertEqual(result.tool, "run_command")
        self.assertEqual(result.exit_code, 2)
        self.assertEqual(result.output, "failed")
        self.assertTrue(result.truncated)
        self.assertEqual(result.metadata, {"reason": "bad_input"})
