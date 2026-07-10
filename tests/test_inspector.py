from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from docode.agent.inspector import ProjectInspector
from docode.dobox.types import ToolResult


class InspectorTools:
    async def list_files(self, path: str = ".") -> ToolResult:
        return ToolResult(
            tool="list_files",
            output=(
                "total 16\n"
                "-rw-r--r--  1 app app  22 Jan  1 00:00 README.md\n"
                "-rw-r--r--  1 app app  18 Jan  1 00:00 package.json\n"
            ),
        )

    async def read_file(self, path: str) -> ToolResult:
        return ToolResult(tool="read_file", output=f"content for {path}")

    async def detect_test_command(self) -> str:
        return "npm test"

    async def detect_build_command(self) -> str:
        return "npm run build"

    async def detect_lint_command(self) -> str | None:
        return None


class ProjectInspectorTests(IsolatedAsyncioTestCase):
    async def test_reads_important_files_from_ls_la_listing(self) -> None:
        inspection = await ProjectInspector().inspect("update settings", InspectorTools())

        self.assertEqual(set(inspection.important_files), {"README.md", "package.json"})
        self.assertEqual(inspection.detected_commands["test"], "npm test")
        self.assertIn("`npm run build` exits successfully.", inspection.acceptance_criteria)

    async def test_explicit_command_is_preserved_separately_from_detected_tests(self) -> None:
        command = "python3 checks/check_contract.py --mode exact"
        inspection = await ProjectInspector().inspect(
            f"Update source.py.\nVerification commands:\n1. {command}",
            InspectorTools(),
        )

        self.assertEqual(inspection.explicit_commands, [command])
        self.assertEqual(inspection.detected_commands["test"], "npm test")
        self.assertIn(f"The exact required command `{command}` exits successfully after the latest edit.", inspection.acceptance_criteria)
