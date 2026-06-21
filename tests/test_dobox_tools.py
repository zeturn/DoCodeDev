from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from docode.dobox.tools import DoBoxTools, LocalToolRegistry, build_dobox_tool_registry, register_dobox_tools
from docode.dobox.types import CommandResult, FileResult


class FakeDoBoxClient:
    def __init__(self, *, package_scripts: set[str] | None = None, truncated_command: bool = False) -> None:
        self.package_scripts = package_scripts or set()
        self.truncated_command = truncated_command
        self.agent_session_ids: list[str | None] = []

    async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        command_text = " ".join(command) if isinstance(command, list) else str(command)
        if "p.scripts && p.scripts['test']" in command_text:
            return CommandResult("yes", 0 if "test" in self.package_scripts else 1)
        if "p.scripts && p.scripts['build']" in command_text:
            return CommandResult("yes", 0 if "build" in self.package_scripts else 1)
        if "p.scripts && p.scripts['lint']" in command_text:
            return CommandResult("yes", 0 if "lint" in self.package_scripts else 1)
        if "test -f go.mod" in command_text:
            return CommandResult("yes", 0)
        if "test -f" in command_text:
            return CommandResult("", 1)
        return CommandResult(f"{project_id}:{cwd}:{command}", 0, truncated=self.truncated_command)

    async def git_diff(self, project_id, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return "diff --git a/a b/a\n+change\n"

    async def git_status(self, project_id, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return CommandResult(" M a\n", 0)

    async def git_commit(self, project_id, message, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return CommandResult(f"[main abc123] {message}\n", 0)

    async def read_file(self, project_id, path, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return "content"

    async def write_file(self, project_id, path, content, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return None

    async def list_files(self, project_id, path=".", agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return CommandResult("a\nb\n", 0)

    async def search(self, project_id, query, path=".", agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return CommandResult("a:1:match\n", 0)

    async def preview(self, project_id, port, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return {"preview_url": f"https://preview.example/{project_id}/{port}"}

    async def logs(self, project_id, tail="200", agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        return f"{project_id}:last {tail} lines\n"


class DoBoxToolsTests(IsolatedAsyncioTestCase):
    async def test_detects_go_test_command(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "p1")
        self.assertEqual(await tools.detect_test_command(), "go test ./...")

    async def test_detects_package_build_and_lint_commands(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(package_scripts={"build", "lint"}), "p1")
        self.assertEqual(await tools.detect_build_command(), "npm run build")
        self.assertEqual(await tools.detect_lint_command(), "npm run lint")

    async def test_run_build_and_lint_return_not_detected_when_absent(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "p1")
        lint = await tools.run_lint()
        self.assertEqual(lint.exit_code, 0)
        self.assertEqual(lint.metadata, {"detected": False})

    async def test_tool_results_never_expose_container_id(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "project-123")
        result = await tools.run_command("go test ./...", cwd="/workspace")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("project-123", result.output)
        self.assertNotIn("container", result.metadata or {})

    async def test_rejects_tool_paths_outside_workspace_before_calling_dobox(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        command = await tools.run_command("pwd", cwd="/etc")
        read = await tools.read_file("../secret.txt")
        write = await tools.write_file("/var/tmp/secret.txt", "secret")
        listing = await tools.list_files("/workspace/../etc")
        search = await tools.search("secret", path="../../")

        self.assertEqual(command.exit_code, 2)
        self.assertEqual(read.exit_code, 2)
        self.assertEqual(write.exit_code, 2)
        self.assertEqual(listing.exit_code, 2)
        self.assertEqual(search.exit_code, 2)
        self.assertTrue(all(result.metadata["rejected"] for result in [command, read, write, listing, search]))
        self.assertEqual(client.agent_session_ids, [])

    async def test_allows_relative_and_workspace_paths(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        command = await tools.run_command("pwd", cwd="/workspace/src")
        read = await tools.read_file("src/../README.md")
        listing = await tools.list_files(".")

        self.assertEqual(command.exit_code, 0)
        self.assertEqual(read.exit_code, 0)
        self.assertEqual(listing.exit_code, 0)
        self.assertEqual(client.agent_session_ids, [None, None, None])

    async def test_run_command_preserves_server_truncation_flag(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(truncated_command=True), "project-123")
        result = await tools.run_command("yes")

        self.assertTrue(result.truncated)

    async def test_detected_verification_tools_preserve_truncation_flag(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(package_scripts={"build", "lint"}, truncated_command=True), "project-123")

        tests = await tools.run_tests()
        build = await tools.run_build()
        lint = await tools.run_lint()

        self.assertTrue(tests.truncated)
        self.assertTrue(build.truncated)
        self.assertTrue(lint.truncated)
        self.assertEqual(tests.metadata, {"command": "go test ./...", "detected": True})
        self.assertEqual(build.metadata, {"command": "npm run build", "detected": True})
        self.assertEqual(lint.metadata, {"command": "npm run lint", "detected": True})

    async def test_read_file_preserves_server_truncation_flag(self) -> None:
        class TruncatedReadClient(FakeDoBoxClient):
            async def read_file(self, project_id, path, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return FileResult(content="partial", path="/workspace/large.txt", file_name="large.txt", bytes_read=7, truncated=True)

        tools = DoBoxTools(TruncatedReadClient(), "project-123")

        result = await tools.read_file("large.txt")

        self.assertEqual(result.output, "partial")
        self.assertTrue(result.truncated)
        self.assertEqual(
            result.metadata,
            {"path": "large.txt", "resolved_path": "/workspace/large.txt", "file_name": "large.txt", "bytes": 7},
        )

    async def test_tool_calls_include_agent_session_id(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123", agent_session_id="session-7")

        await tools.run_command("go test ./...")
        await tools.read_file("README.md")
        await tools.git_diff()
        await tools.logs(20)

        self.assertEqual(client.agent_session_ids, ["session-7", "session-7", "session-7", "session-7"])

    async def test_git_commit_tool(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "project-123")
        result = await tools.git_commit("ship changes")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ship changes", result.output)

    async def test_git_diff_preserves_server_truncation_flag(self) -> None:
        class TruncatedDiffClient(FakeDoBoxClient):
            async def git_diff_result(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return CommandResult("diff --git a/a b/a\n+change\n", 0, truncated=True)

        tools = DoBoxTools(TruncatedDiffClient(), "project-123")

        result = await tools.git_diff()

        self.assertIn("+change", result.output)
        self.assertTrue(result.truncated)

    async def test_preview_and_logs_tools_are_project_level(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "project-123")

        preview = await tools.preview(3000)
        logs = await tools.logs(50)

        self.assertEqual(preview.output, "https://preview.example/project-123/3000")
        self.assertEqual(preview.metadata, {"port": 3000})
        self.assertIn("last 50 lines", logs.output)
        self.assertEqual(logs.metadata, {"tail": 50})
        self.assertNotIn("container_id", preview.metadata)
        self.assertNotIn("container_id", logs.metadata)

    async def test_preview_descriptor_fallback_is_concise(self) -> None:
        class DescriptorPreviewClient(FakeDoBoxClient):
            async def preview(self, project_id, port, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return {
                    "project_id": 1,
                    "sandbox_id": 2,
                    "port": port,
                    "status": "preview_descriptor",
                    "message": "Preview proxy routing is not exposed by this endpoint.",
                }

        tools = DoBoxTools(DescriptorPreviewClient(), "project-123")

        preview = await tools.preview(3000)

        self.assertEqual(preview.output, "preview port 3000: Preview proxy routing is not exposed by this endpoint.")

    async def test_builds_weav_compatible_tool_registry(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "project-123")
        registry = build_dobox_tool_registry(tools, LocalToolRegistry())

        spec = registry.get("run_command")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.input_schema["properties"]["command"]["type"], "string")
        self.assertEqual(spec.input_schema["required"], ["command"])
        self.assertIsNotNone(registry.get("preview"))
        self.assertIsNotNone(registry.get("logs"))

        output = await registry.call("run_command", {"command": "go test ./...", "cwd": "/workspace"})
        self.assertTrue(output["ok"])
        self.assertEqual(output["exit_code"], 0)
        self.assertNotIn("container_id", output["metadata"])

        preview = await registry.call("preview", {"port": 3000})
        self.assertEqual(preview["content"], "https://preview.example/project-123/3000")

    async def test_registers_with_name_handler_registry_shape(self) -> None:
        class NameHandlerRegistry:
            def __init__(self) -> None:
                self.handlers = {}

            def register(self, name, handler):
                self.handlers[name] = handler

        tools = DoBoxTools(FakeDoBoxClient(), "project-123")
        registry = NameHandlerRegistry()
        register_dobox_tools(registry, tools)

        output = await registry.handlers["read_file"]({"path": "README.md"})
        self.assertEqual(output["content"], "content")
        self.assertTrue(output["ok"])
