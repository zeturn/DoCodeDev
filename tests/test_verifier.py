from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from docode.agent.verifier import CodingVerifier
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class VerifierTools:
    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M a\n", exit_code=0)

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/a b/a\n+change\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", exit_code=0, metadata={"detected": True, "command": "pytest"})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="build failed", exit_code=1, metadata={"detected": True, "command": "npm run build"})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", exit_code=0, metadata={"detected": False})


class PassingVerifierTools(VerifierTools):
    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="ok", exit_code=0, metadata={"detected": True, "command": "npm run build"})


class TruncatedDiffVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/a b/a\n+change\n", truncated=True)


class FailingStatusVerifierTools(PassingVerifierTools):
    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output="fatal: not a git repository", exit_code=128)


class NonGitArtifactVerifierTools(FailingStatusVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="usage: git diff --no-index <path> <path>", exit_code=129)

    async def list_files(self, path: str = ".") -> ToolResult:
        self.path = path
        return ToolResult(tool="list_files", output="DOCODE_RESULT.md\n", exit_code=0, metadata={"path": path})


class UntrackedArtifactVerifierTools(PassingVerifierTools):
    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output="?? crawler.py\n?? tests/test_crawler.py\n", exit_code=0)

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="", exit_code=0)

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})

    async def list_files(self, path: str = ".") -> ToolResult:
        self.path = path
        return ToolResult(tool="list_files", output="crawler.py\ntests/\ntests/test_crawler.py\n", exit_code=0, metadata={"path": path})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="ok", exit_code=0)


class RaisingVerifierTools(PassingVerifierTools):
    async def run_tests(self) -> ToolResult:
        raise RuntimeError("pytest crashed")


class NoDetectedPythonVerifierTools(PassingVerifierTools):
    def __init__(self, *, smoke_exit_code: int = 0) -> None:
        self.commands: list[str] = []
        self.smoke_exit_code = smoke_exit_code

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/crawler.py b/crawler.py\n+print('run')\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", exit_code=0, metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.commands.append(command)
        _ = cwd
        return ToolResult(tool="run_command", output="smoke output", exit_code=self.smoke_exit_code, metadata={"command": command})


class VetoingJudge:
    async def judge(self, *, instruction, diff, tests, build, lint):
        from docode.agent.verifier import VerifierJudgement

        _ = instruction, diff, tests, build, lint
        return VerifierJudgement(
            passed=False,
            confidence=0.72,
            reason="The diff changes a file but does not implement the requested behavior.",
            required_fixes=["implement the requested behavior"],
        )


class BrokenJudge:
    async def judge(self, *, instruction, diff, tests, build, lint):
        _ = instruction, diff, tests, build, lint
        raise ValueError("not json")


class VerifierTests(IsolatedAsyncioTestCase):
    async def test_failing_build_blocks_success(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            VerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIn("fix failing build command", result.required_fixes)
        self.assertEqual(result.git_status, " M a\n")
        self.assertEqual(result.status_result.output, " M a\n")
        self.assertEqual(result.build_result.output, "build failed")

    async def test_verifier_judge_can_veto_passing_commands(self) -> None:
        result = await CodingVerifier(judge=VetoingJudge()).verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="add settings page"),
            PassingVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.llm_judgement)
        self.assertEqual(result.llm_judgement.reason, "The diff changes a file but does not implement the requested behavior.")
        self.assertIn("implement the requested behavior", result.required_fixes)

    async def test_truncated_git_diff_blocks_success(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            TruncatedDiffVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIn("reduce or split the change so the complete git diff can be exported", result.required_fixes)
        self.assertTrue(result.git_diff)

    async def test_failing_git_status_blocks_success(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            FailingStatusVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIn("fix failing git status command", result.required_fixes)
        self.assertEqual(result.status_result.exit_code, 128)
        self.assertIn("not a git repository", result.git_status)

    async def test_non_git_workspace_artifact_can_verify_without_patch(self) -> None:
        tools = NonGitArtifactVerifierTools()
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="create a result file"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.git_diff, "")
        self.assertIsNotNone(result.workspace_result)
        self.assertEqual(result.workspace_result.output, "DOCODE_RESULT.md\n")
        self.assertIn("explicit workspace artifacts exist", result.reason)

    async def test_untracked_files_in_empty_git_workspace_count_as_artifact_evidence(self) -> None:
        tools = UntrackedArtifactVerifierTools()
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIn("diff --git a/crawler.py b/crawler.py", result.git_diff)
        self.assertIsNotNone(result.workspace_result)
        self.assertIn("python3 -m py_compile crawler.py", tools.command)

    async def test_verifier_converts_tool_exception_to_failed_check(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            RaisingVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIn("fix failing verification command", result.required_fixes)
        self.assertIsNotNone(result.test_result)
        self.assertEqual(result.test_result.exit_code, 1)
        self.assertIn("pytest crashed", result.test_result.output)
        self.assertEqual(result.test_result.metadata["exception_type"], "RuntimeError")

    async def test_broken_verifier_judge_fails_structurally(self) -> None:
        result = await CodingVerifier(judge=BrokenJudge()).verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            PassingVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.confidence, 0.0)
        self.assertIsNotNone(result.llm_judgement)
        self.assertTrue(result.llm_judgement.reason.startswith("verifier_model_failed:"))

    async def test_python_crawler_smoke_failure_blocks_success(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=1)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据"),
            tools,
        )

        self.assertFalse(result.passed)
        self.assertIn("fix failing smoke verification command", result.required_fixes)
        self.assertIsNotNone(result.smoke_result)
        self.assertEqual(result.smoke_result.exit_code, 1)
        self.assertGreaterEqual(len(tools.commands), 1)
        self.assertIn("python3 -m py_compile crawler.py", tools.commands[-1])
        self.assertIn("python3 crawler.py", tools.commands[-1])

    async def test_python_smoke_success_allows_no_detected_standard_commands(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIsNotNone(result.smoke_result)
        self.assertEqual(result.smoke_result.exit_code, 0)
        self.assertIn("python3 crawler.py", result.smoke_result.metadata["command"])

    async def test_python_crawler_smoke_checks_for_csv_output(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据并保存 CSV"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIn("glob.glob", result.smoke_result.metadata["command"])
        self.assertIn("*.csv", result.smoke_result.metadata["command"])

    async def test_python_crawler_smoke_checks_for_json_output_when_requested(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="抓取至少 5 条记录并写入 data/output.json"),
            tools,
        )

        self.assertTrue(result.passed)
        command = result.smoke_result.metadata["command"]
        self.assertIn("data/output.json", command)
        self.assertIn("min_records=5", command)
        self.assertNotIn("*.csv", command)

    async def test_python_crawler_exit_zero_error_output_blocks_success(self) -> None:
        class ErrorOutputTools(NoDetectedPythonVerifierTools):
            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                return ToolResult(tool="run_command", output='Error fetching data: {"error":"Forbidden"}', exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据并保存 CSV"),
            ErrorOutputTools(smoke_exit_code=0),
        )

        self.assertFalse(result.passed)
        self.assertIn("fix failing smoke verification command", result.required_fixes)
        self.assertTrue(any("call web_search again" in fix for fix in result.required_fixes))
        self.assertEqual(result.smoke_result.exit_code, 1)

    async def test_python_crawler_404_output_requires_endpoint_reinspection(self) -> None:
        class NotFoundOutputTools(NoDetectedPythonVerifierTools):
            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                return ToolResult(
                    tool="run_command",
                    output="requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://api.example.test/latest",
                    exit_code=1,
                )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据并保存 CSV"),
            NotFoundOutputTools(smoke_exit_code=0),
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("re-inspect the source documentation with fetch_url" in fix for fix in result.required_fixes))
        self.assertTrue(any("do not keep retrying the same endpoint" in fix for fix in result.required_fixes))

    async def test_python_crawler_dependency_warning_not_found_does_not_fail_smoke(self) -> None:
        class WarningOutputTools(NoDetectedPythonVerifierTools):
            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                return ToolResult(
                    tool="run_command",
                    output="Saved exchange rates to exchange_rates.csv\nCSV outputs: exchange_rates.csv\nOptional dependency was not found",
                    exit_code=0,
                )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据并保存 CSV"),
            WarningOutputTools(smoke_exit_code=0),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.smoke_result.exit_code, 0)
