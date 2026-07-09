from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from docode.dobox.tools import DoBoxTools, LocalToolRegistry, build_dobox_tool_registry, register_dobox_tools
from docode.dobox.types import CommandResult, FileResult


class FakeDoBoxClient:
    def __init__(self, *, package_scripts: set[str] | None = None, truncated_command: bool = False) -> None:
        self.package_scripts = package_scripts or set()
        self.truncated_command = truncated_command
        self.agent_session_ids: list[str | None] = []
        self.files: dict[str, str] = {"README.md": "hello\nworld\n"}
        self.written_files: list[tuple[str, str]] = []

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
        if "git apply --check" in command_text:
            return CommandResult(" README.md | 1 +\n", 0)
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
        return self.files.get(path, "content")

    async def write_file(self, project_id, path, content, agent_session_id=None):
        self.agent_session_ids.append(agent_session_id)
        self.files[path] = content
        self.written_files.append((path, content))
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

    async def test_git_diff_result_exception_uses_command_fallback(self) -> None:
        class RaisingDiffClient(FakeDoBoxClient):
            def __init__(self) -> None:
                super().__init__()
                self.commands: list[str] = []

            async def git_diff_result(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                raise TimeoutError("diff timed out")

            async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                command_text = " ".join(command) if isinstance(command, list) else str(command)
                self.commands.append(command_text)
                if "git --no-pager diff" in command_text:
                    return CommandResult(
                        "diff --git a/app.py b/app.py\n+change\n"
                        "diff --git a/__pycache__/app.cpython-314.pyc b/__pycache__/app.cpython-314.pyc\n+cache\n",
                        0,
                    )
                return await super().run_command(project_id, command, cwd, timeout_sec, output_limit, agent_session_id)

        tools = DoBoxTools(RaisingDiffClient(), "project-123")

        result = await tools.git_diff()

        self.assertEqual(result.exit_code, 0)
        self.assertIn("diff --git a/app.py b/app.py", result.output)
        self.assertNotIn("__pycache__", result.output)
        self.assertTrue(result.metadata["runtime_command_fallback"])
        self.assertEqual(result.metadata["endpoint_error_type"], "TimeoutError")

    async def test_git_diff_result_and_command_fallback_exception_returns_runtime_safe_fallback(self) -> None:
        class RaisingDiffAndCommandClient(FakeDoBoxClient):
            async def git_diff_result(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                raise TimeoutError("diff timed out")

            async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                raise RuntimeError("exec unavailable")

        tools = DoBoxTools(RaisingDiffAndCommandClient(), "project-123")

        result = await tools.git_diff()

        self.assertEqual(result.exit_code, 124)
        self.assertIn("git_diff unavailable: RuntimeError: exec unavailable", result.output)
        self.assertEqual(result.metadata, {"runtime_safe_fallback": True, "error_type": "RuntimeError"})

    async def test_git_status_exception_uses_command_fallback_and_strips_ansi(self) -> None:
        class RaisingStatusClient(FakeDoBoxClient):
            async def git_status(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                raise RuntimeError("status unavailable")

            async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                command_text = " ".join(command) if isinstance(command, list) else str(command)
                if "git status --porcelain" in command_text:
                    return CommandResult("\x1b[31m M app.py\x1b[0m\n M __pycache__/app.cpython-314.pyc\n", 0)
                return await super().run_command(project_id, command, cwd, timeout_sec, output_limit, agent_session_id)

        tools = DoBoxTools(RaisingStatusClient(), "project-123")

        result = await tools.git_status()

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output, " M app.py\n")
        self.assertNotIn("\x1b[", result.output)
        self.assertNotIn("__pycache__", result.output)
        self.assertTrue(result.metadata["runtime_command_fallback"])
        self.assertEqual(result.metadata["endpoint_error_type"], "RuntimeError")

    async def test_git_status_endpoint_output_strips_ansi_and_pycache_noise(self) -> None:
        class NoisyStatusClient(FakeDoBoxClient):
            async def git_status(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return CommandResult("\x1b[32m M app.py\x1b[0m\n?? tests/__pycache__/test_app.cpython-314.pyc\n", 0)

        tools = DoBoxTools(NoisyStatusClient(), "project-123")

        result = await tools.git_status()

        self.assertEqual(result.output, " M app.py\n")
        self.assertNotIn("\x1b[", result.output)
        self.assertNotIn(".pyc", result.output)

    async def test_git_helpers_preserve_normal_command_failures(self) -> None:
        class FailingGitClient(FakeDoBoxClient):
            async def git_status(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return CommandResult("fatal: not a git repository", 128)

            async def git_diff_result(self, project_id, agent_session_id=None):
                self.agent_session_ids.append(agent_session_id)
                return CommandResult("fatal: bad revision", 128)

        tools = DoBoxTools(FailingGitClient(), "project-123")

        status = await tools.git_status()
        diff = await tools.git_diff()

        self.assertEqual(status.exit_code, 128)
        self.assertEqual(diff.exit_code, 128)
        self.assertNotEqual((status.metadata or {}).get("runtime_safe_fallback"), True)
        self.assertNotEqual((diff.metadata or {}).get("runtime_safe_fallback"), True)

    async def test_edit_file_replaces_exact_text_and_returns_diff(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        result = await tools.edit_file("README.md", "hello\n", "hello there\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("-hello", result.output)
        self.assertIn("+hello there", result.output)
        self.assertEqual(client.files["README.md"], "hello there\nworld\n")

    async def test_edit_file_rejects_noop_replacement(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        result = await tools.edit_file("README.md", "hello\n", "hello\n")

        self.assertEqual(result.exit_code, 2)
        self.assertIn("would not change", result.output)
        self.assertEqual(client.files["README.md"], "hello\nworld\n")

    async def test_replace_in_file_alias_uses_find_replace_arguments(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        result = await tools.replace_in_file("README.md", "hello\n", "hello Ada\n")

        self.assertEqual(result.tool, "replace_in_file")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("+hello Ada", result.output)
        self.assertEqual(client.files["README.md"], "hello Ada\nworld\n")

    async def test_edit_file_rejects_missing_or_ambiguous_match(self) -> None:
        client = FakeDoBoxClient()
        client.files["README.md"] = "alpha\nbeta\nalpha\n"
        tools = DoBoxTools(client, "project-123")

        missing = await tools.edit_file("README.md", "gamma", "delta")
        ambiguous = await tools.edit_file("README.md", "alpha", "delta")

        self.assertEqual(missing.exit_code, 1)
        self.assertIn("did not match exactly", missing.output)
        self.assertEqual(ambiguous.exit_code, 1)
        self.assertIn("matched 2 times", ambiguous.output)

    async def test_apply_patch_writes_temp_patch_and_runs_git_apply(self) -> None:
        client = FakeDoBoxClient()
        tools = DoBoxTools(client, "project-123")

        result = await tools.apply_patch("diff --git a/README.md b/README.md\n")

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(client.written_files[-1][0], ".docode_apply_patch.diff")
        self.assertEqual(result.metadata, {"patch_bytes": 35})

    async def test_apply_patch_failure_cleans_patch_file(self) -> None:
        class FailingPatchClient(FakeDoBoxClient):
            def __init__(self) -> None:
                super().__init__()
                self.commands: list[str] = []

            async def run_command(self, project_id, command, cwd="/workspace", timeout_sec=120, output_limit=1_000_000, agent_session_id=None):
                self.commands.append(str(command))
                if "git apply --check" in str(command):
                    return CommandResult("patch failed", 1)
                return CommandResult("removed", 0)

        client = FailingPatchClient()
        tools = DoBoxTools(client, "project-123")

        result = await tools.apply_patch("diff --git a/README.md b/README.md\n")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("patch failed", result.output)
        self.assertTrue(any("rm -f .docode_apply_patch.diff" in command for command in client.commands))

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
        self.assertIsNotNone(registry.get("edit_file"))
        self.assertIsNotNone(registry.get("replace_in_file"))
        self.assertIsNotNone(registry.get("apply_patch"))
        self.assertIsNotNone(registry.get("read_file_range"))
        self.assertIsNotNone(registry.get("read_symbol"))
        self.assertIsNotNone(registry.get("preview"))
        self.assertIsNotNone(registry.get("logs"))

        output = await registry.call("run_command", {"command": "go test ./...", "cwd": "/workspace"})
        self.assertTrue(output["ok"])
        self.assertEqual(output["exit_code"], 0)
        self.assertNotIn("container_id", output["metadata"])

        preview = await registry.call("preview", {"port": 3000})
        self.assertEqual(preview["content"], "https://preview.example/project-123/3000")

    async def test_read_file_range_returns_numbered_excerpt(self) -> None:
        tools = DoBoxTools(FakeDoBoxClient(), "project-123")

        result = await tools.read_file_range("README.md", start_line=2, end_line=2)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.output, "2: world")
        self.assertEqual(result.metadata["path"], "README.md")
        self.assertEqual(result.metadata["start_line"], 2)

    async def test_read_symbol_returns_python_definition(self) -> None:
        client = FakeDoBoxClient()
        client.files["crawler.py"] = "x = 1\n\nclass Parser:\n    def parse(self):\n        return []\n\n"
        tools = DoBoxTools(client, "project-123")

        result = await tools.read_symbol("crawler.py", "Parser", context_lines=0)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("3: class Parser:", result.output)
        self.assertEqual(result.metadata["symbol"], "Parser")

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
        self.assertEqual(output["content"], "hello\nworld\n")
        self.assertTrue(output["ok"])
