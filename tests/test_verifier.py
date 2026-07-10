from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.runtime.python_cmd import local_python_command_args
from docode.agent.verifier import (
    CodingVerifier,
    VerificationEvidence,
    build_verification_plan,
    crawler_output_artifact_verified,
    diff_contains_placeholder,
    json_output_check_script,
    requires_json_output_check,
    verification_evidence_from_steps,
)
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
        return ToolResult(tool="run_command", output="JSON outputs: data/output.json\nmin_records=1", exit_code=0)


class StatusOnlyChangeVerifierTools(PassingVerifierTools):
    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M calculator.py\n", exit_code=0)

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="", exit_code=0)

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="ok", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="ok", exit_code=0, metadata={"command": command})


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
        if "*.csv" in command:
            return ToolResult(tool="run_command", output="CSV outputs: output.csv", exit_code=self.smoke_exit_code, metadata={"command": command})
        if "JSON outputs:" in command:
            return ToolResult(tool="run_command", output="JSON outputs: data/output.json\nmin_records=5", exit_code=self.smoke_exit_code, metadata={"command": command})
        return ToolResult(tool="run_command", output="smoke output", exit_code=self.smoke_exit_code, metadata={"command": command})


class CrawlerPolicyVerifierTools(NoDetectedPythonVerifierTools):
    def __init__(self, diff: str, *, smoke_output: str = "JSON outputs: data/output.json\nmin_records=5") -> None:
        super().__init__(smoke_exit_code=0)
        self.diff = diff
        self.smoke_output = smoke_output

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output=self.diff)

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.commands.append(command)
        _ = cwd
        return ToolResult(tool="run_command", output=self.smoke_output, exit_code=0, metadata={"command": command})


class BugfixWithoutTestVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/src/app.py b/src/app.py\n+return retry()\n")

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="ok", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="ok", exit_code=0)


class DocsVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/README.md b/README.md\n+More usage docs.\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", exit_code=0, metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})


class StrictDocsVerifierTools(DocsVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(
            tool="git_diff",
            output=(
                "diff --git a/README.md b/README.md\n"
                "+## Installation\n"
                "+Run `pip install .`.\n"
                "+## Usage\n"
                "+Run the tool from the command line.\n"
            ),
        )

    async def run_tests(self) -> ToolResult:
        raise RuntimeError("docs-only verifier should not run tests")

    async def run_build(self) -> ToolResult:
        raise RuntimeError("docs-only verifier should not run build")

    async def run_lint(self) -> ToolResult:
        raise RuntimeError("docs-only verifier should not run lint")


class CliHintVerifierTools(PassingVerifierTools):
    def __init__(self) -> None:
        self.command = ""

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/cli.py b/cli.py\n+print(f'Hello, {args.name}!')\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", exit_code=0, metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="Hello, Ada!\n", exit_code=0, metadata={"command": command})


class ApiVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/config/api_endpoint.txt b/config/api_endpoint.txt\n+https://api.example.test/v1/items\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", exit_code=0, metadata={"detected": True, "command": "pytest"})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="ok", exit_code=0)


class SourceRepairVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(
            tool="git_diff",
            output=(
                "diff --git a/source_config.py b/source_config.py\n"
                "-SOURCE_URL = 'https://api.example.invalid/missing'\n"
                "+SOURCE_URL = 'https://jsonplaceholder.typicode.com/todos/1'\n"
            ),
        )

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command detected", exit_code=0, metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", exit_code=0, metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", exit_code=0, metadata={"detected": False})

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        self.command = command
        _ = cwd
        return ToolResult(tool="run_command", output="ok", exit_code=0, metadata={"command": command})


class ArtifactExportVerifierTools(PassingVerifierTools):
    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="diff --git a/module.py b/module.py\n-VALUE = 'old'\n+VALUE = 'new'\n")

    async def run_tests(self) -> ToolResult:
        raise RuntimeError("artifact export verifier should not run tests")

    async def run_build(self) -> ToolResult:
        raise RuntimeError("artifact export verifier should not run build")

    async def run_lint(self) -> ToolResult:
        raise RuntimeError("artifact export verifier should not run lint")

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        raise RuntimeError(f"artifact export verifier should not run smoke command: {command}")


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


class HangingJudge:
    async def judge(self, *, instruction, status=None, diff, tests, build, lint, smoke=None):
        _ = instruction, status, diff, tests, build, lint, smoke
        await asyncio.sleep(3600)


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
        self.assertTrue(any("python3 -m py_compile crawler.py" in command for command in result.smoke_result.metadata["commands"]))

    async def test_status_changes_count_as_change_evidence_when_git_diff_is_empty(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            StatusOnlyChangeVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertIn("diff --git a/calculator.py b/calculator.py", result.git_diff)
        self.assertNotIn("produce a non-empty git diff or explicit artifact", result.required_fixes)

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

    async def test_verifier_judge_timeout_falls_back_to_deterministic_checks(self) -> None:
        result = await CodingVerifier(judge=HangingJudge(), judge_timeout_seconds=0.01).verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="ship it"),
            PassingVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertIsNone(result.llm_judgement)
        self.assertGreater(result.confidence, 0.8)

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
        self.assertIn("python3 -m py_compile crawler.py", tools.commands)

    async def test_python_smoke_success_allows_no_detected_standard_commands(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python script"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIsNotNone(result.smoke_result)
        self.assertEqual(result.smoke_result.exit_code, 0)
        self.assertIn("python3 crawler.py", result.smoke_result.metadata["command"])

    async def test_python_smoke_missing_file_attaches_workspace_diagnostic(self) -> None:
        class MissingFileTools(NoDetectedPythonVerifierTools):
            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                if "find /workspace" in command:
                    return ToolResult(tool="run_command", output="/workspace\nREADME.md\n", exit_code=0)
                return ToolResult(tool="run_command", output="python3: can't open file '/workspace/crawler.py': [Errno 2] No such file or directory", exit_code=2)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="生成一个可运行的 Python 爬虫脚本抓取每日数据"),
            MissingFileTools(smoke_exit_code=1),
        )

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.smoke_result)
        diagnostic = result.smoke_result.metadata["workspace_diagnostic"]
        self.assertEqual(diagnostic["exit_code"], 0)
        self.assertIn("README.md", diagnostic["output"])

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

    async def test_crawler_rejects_undeclared_third_party_dependencies(self) -> None:
        diff = (
            "diff --git a/crawler.py b/crawler.py\n"
            "+import requests\n"
            "+from bs4 import BeautifulSoup\n"
            "+def main():\n"
            "+    print('JSON outputs: data/output.json')\n"
            "+if __name__ == '__main__':\n"
            "+    main()\n"
        )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Build a crawler that writes data/output.json with at least 5 records."),
            CrawlerPolicyVerifierTools(diff),
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("third-party Python dependency used but not declared" in fix for fix in result.required_fixes))

    async def test_crawler_allows_declared_third_party_dependencies(self) -> None:
        diff = (
            "diff --git a/crawler.py b/crawler.py\n"
            "+import requests\n"
            "+from bs4 import BeautifulSoup\n"
            "+def main():\n"
            "+    print('JSON outputs: data/output.json')\n"
            "+if __name__ == '__main__':\n"
            "+    main()\n"
            "diff --git a/requirements.txt b/requirements.txt\n"
            "+requests\n"
            "+beautifulsoup4\n"
        )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Build a crawler that writes data/output.json with at least 5 records."),
            CrawlerPolicyVerifierTools(diff),
        )

        self.assertTrue(result.passed)

    async def test_crawler_prefers_dry_run_when_entrypoint_supports_it(self) -> None:
        diff = (
            "diff --git a/crawler.py b/crawler.py\n"
            "+import argparse\n"
            "+def main():\n"
            "+    parser = argparse.ArgumentParser()\n"
            "+    parser.add_argument('--dry-run', action='store_true')\n"
            "+    print('JSON outputs: data/output.json')\n"
            "+if __name__ == '__main__':\n"
            "+    main()\n"
        )
        tools = CrawlerPolicyVerifierTools(diff)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Build a crawler that writes data/output.json with at least 5 records."),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertTrue(any(command == "python3 crawler.py --dry-run" for command in tools.commands))

    async def test_crawler_rejects_duplicate_implementation(self) -> None:
        diff = (
            "diff --git a/crawler.py b/crawler.py\n"
            "+def main():\n"
            "+    pass\n"
            "+if __name__ == '__main__':\n"
            "+    main()\n"
            "+def main():\n"
            "+    pass\n"
            "+if __name__ == '__main__':\n"
            "+    main()\n"
        )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Build a crawler that writes data/output.json with at least 5 records."),
            CrawlerPolicyVerifierTools(diff),
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("appears duplicated" in fix for fix in result.required_fixes))

    async def test_crawler_requires_output_artifact_evidence(self) -> None:
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('ok')\n"

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Build a crawler."),
            CrawlerPolicyVerifierTools(diff, smoke_output="ok"),
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("dry-run must write an output artifact" in fix for fix in result.required_fixes))

    async def test_local_fixture_crawler_does_not_require_external_source_evidence(self) -> None:
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('JSON outputs: out.json')\n"

        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction=(
                    "Implement crawler.py so it parses fixtures/products.html and writes product records to JSON.\n\n"
                    "Verification commands:\n"
                    "1. python -m unittest discover -s tests\n"
                    "2. python crawler.py fixtures/products.html --output out.json"
                ),
            ),
            CrawlerPolicyVerifierTools(diff, smoke_output="JSON outputs: out.json\nmin_records=2"),
        )

        self.assertTrue(result.passed)
        self.assertFalse(result.verification_plan.require_external_source_verified)
        self.assertNotIn(
            "verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run",
            result.required_fixes,
        )

    async def test_crawler_with_public_url_requires_external_source_evidence(self) -> None:
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('JSON outputs: out.json')\n"

        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="Build a crawler for https://example.test/products that writes product records to JSON.",
            ),
            CrawlerPolicyVerifierTools(diff, smoke_output="JSON outputs: out.json\nmin_records=2"),
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.verification_plan.require_external_source_verified)
        self.assertIn(
            "verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run",
            result.required_fixes,
        )

    async def test_public_url_crawler_accepts_successful_explicit_url_command_evidence(self) -> None:
        command = "python crawler.py --url https://example.test/products --output out.json --dry-run"
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('JSON outputs: out.json')\n"

        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction=(
                    "Build a crawler for https://example.test/products that writes product records to JSON.\n\n"
                    "Verification commands:\n"
                    f"1. {command}"
                ),
            ),
            CrawlerPolicyVerifierTools(diff, smoke_output="JSON outputs: out.json\nmin_records=2"),
            evidence=VerificationEvidence(
                successful_fetch_urls=[],
                successful_web_search_queries=[],
                relevant_fetch_urls=[],
                successful_commands=[command],
                successful_command_outputs=["wrote 12 records to out.json"],
            ),
        )

        self.assertTrue(result.passed)

    async def test_public_url_crawler_rejects_unrelated_or_print_only_url_command(self) -> None:
        required = "python crawler.py --url https://example.test/products --output out.json --dry-run"
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('JSON outputs: out.json')\n"
        instruction = (
            "Build a crawler for https://example.test/products that writes product records to JSON.\n\n"
            "Verification commands:\n"
            f"1. {required}"
        )
        for observed in ("python crawler.py --url https://other.test/items", "echo https://example.test/products"):
            with self.subTest(observed=observed):
                result = await CodingVerifier().verify(
                    CodingJob(id=new_id("job"), user_id="u1", instruction=instruction),
                    CrawlerPolicyVerifierTools(diff, smoke_output="JSON outputs: out.json\nmin_records=2"),
                    evidence=VerificationEvidence(
                        successful_fetch_urls=[],
                        successful_web_search_queries=[],
                        relevant_fetch_urls=[],
                        successful_commands=[observed],
                        successful_command_outputs=["ok"],
                    ),
                )

                self.assertFalse(result.passed)

    async def test_crawler_output_flag_alone_does_not_verify_artifact(self) -> None:
        diff = "diff --git a/crawler.py b/crawler.py\n+def main():\n+    print('done')\n"

        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction=(
                    "Implement crawler.py so it parses fixtures/products.html and writes product records to JSON.\n\n"
                    "Verification commands:\n"
                    "1. python crawler.py fixtures/products.html --output out.json"
                ),
            ),
            CrawlerPolicyVerifierTools(diff, smoke_output="done"),
        )

        self.assertFalse(result.passed)
        self.assertFalse(crawler_output_artifact_verified("python crawler.py fixtures/products.html --output out.json"))
        self.assertTrue(any("dry-run must write an output artifact" in fix for fix in result.required_fixes))

    def test_json_api_response_does_not_require_output_file(self) -> None:
        self.assertFalse(requires_json_output_check("Implement client.parse_items_response so it extracts item names from a JSON API response."))
        self.assertTrue(requires_json_output_check("Build a crawler that writes data/output.json with at least 2 records."))
        self.assertTrue(requires_json_output_check("Build a crawler that writes product records to JSON."))

    def test_apicred_literal_text_does_not_require_external_source_verification(self) -> None:
        plan = build_verification_plan("Create DOCODE_DEEPSEEK_RESULT.md containing exactly: DeepSeek via APICred works.")

        self.assertFalse(plan.require_external_source_verified)
        self.assertNotIn("api_contract_or_mock", plan.required_commands)

    def test_api_adapter_still_requires_external_source_verification(self) -> None:
        plan = build_verification_plan("add API adapter for external endpoint")

        self.assertTrue(plan.require_external_source_verified)
        self.assertIn("api_contract_or_mock", plan.required_commands)

    def test_fixtures_path_does_not_trigger_bugfix_test_requirement(self) -> None:
        plan = build_verification_plan("Build a crawler that parses fixtures/source.html and writes data/output.json.")

        self.assertFalse(plan.require_test_change)
        self.assertNotIn("related_test", plan.required_commands)

    def test_json_output_check_script_runs_as_python_c_argument(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "data"
            output_dir.mkdir()
            (output_dir / "output.json").write_text(json.dumps([{"name": "a"}, {"name": "b"}]), encoding="utf-8")

            result = subprocess.run(
                local_python_command_args("-c", json_output_check_script(2)),
                cwd=tmp,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data/output.json", result.stdout)

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

    async def test_truncated_smoke_output_does_not_fail_when_exit_code_is_zero(self) -> None:
        class TruncatedOutputTools(NoDetectedPythonVerifierTools):
            async def run_tests(self) -> ToolResult:
                return ToolResult(tool="run_tests", output="OK", exit_code=0, metadata={"detected": True, "command": "python3 -m unittest discover -s tests"})

            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                return ToolResult(
                    tool="run_command",
                    output=("line\n" * 200) + "not found <truncated>",
                    exit_code=0,
                    truncated=False,
                    metadata={"command": command},
                )

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Fix noisy.py so tests pass even when command output is very large."),
            TruncatedOutputTools(smoke_exit_code=0),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.smoke_result.exit_code, 0)
        self.assertIn("<truncated>", result.smoke_result.output)

    async def test_truncated_smoke_failure_is_allowed_when_standard_tests_passed(self) -> None:
        class TruncatedFailureTools(NoDetectedPythonVerifierTools):
            async def run_tests(self) -> ToolResult:
                return ToolResult(tool="run_tests", output="OK", exit_code=0, metadata={"detected": True, "command": "python3 -m unittest discover -s tests"})

            async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
                self.commands.append(command)
                _ = cwd
                if "py_compile" in command or "unittest" in command:
                    return ToolResult(tool="run_command", output="OK", exit_code=0, metadata={"command": command})
                return ToolResult(
                    tool="run_command",
                    output="line\n" * 2000,
                    exit_code=1,
                    metadata={"command": command},
                )

        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="Fix noisy.py so tests pass even when command output is very large.\nVerification commands:\n- python3 generate_output.py\n- python3 -m unittest discover -s tests",
            ),
            TruncatedFailureTools(smoke_exit_code=1),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.smoke_result.exit_code, 0)

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

    async def test_bugfix_plan_allows_existing_detected_tests_without_test_change(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="fix retry bug in payment adapter"),
            BugfixWithoutTestVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertIsNotNone(result.verification_plan)
        self.assertTrue(result.verification_plan.require_test_change)
        self.assertNotIn("add or update a related test for this bugfix, or record why no automated test is appropriate", result.required_fixes)

    async def test_bugfix_plan_allows_missing_test_with_recorded_reason(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="fix retry bug in payment adapter"),
            BugfixWithoutTestVerifierTools(),
            evidence=VerificationEvidence(
                successful_fetch_urls=[],
                successful_web_search_queries=[],
                no_test_reason="No automated test is appropriate because this is a configuration-only production hotfix; manual verification was performed.",
            ),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.required_fixes, [])

    async def test_docs_plan_does_not_require_test_change_or_placeholder_gate(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="update README docs"),
            DocsVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertIsNotNone(result.verification_plan)
        self.assertFalse(result.verification_plan.require_test_change)
        self.assertFalse(result.verification_plan.require_no_placeholder)

    async def test_docs_only_verifier_checks_readme_terms_without_tests(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Update README.md with installation and usage sections. Do not change code."),
            StrictDocsVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.verification_plan.docs_only)
        self.assertTrue(result.verification_plan.forbid_code_changes)
        self.assertEqual(result.test_result.metadata["skipped"], True)

    def test_javascript_bugfix_does_not_trigger_cli_script_entrypoint(self) -> None:
        plan = build_verification_plan("Fix the JavaScript sum bug and keep node tests passing.")

        self.assertFalse(plan.require_entrypoint_run)
        self.assertTrue(plan.require_test_change)

    def test_source_repair_plan_requires_external_evidence_not_related_test(self) -> None:
        plan = build_verification_plan(
            "Replace the broken data source URL in source_config.py with a documented working source and record the verification evidence."
        )

        self.assertTrue(plan.external_source_repair)
        self.assertTrue(plan.require_external_source_verified)
        self.assertFalse(plan.require_test_change)
        self.assertNotIn("related_test", plan.required_commands)

    def test_git_diff_sentence_is_not_a_smoke_command(self) -> None:
        plan = build_verification_plan(
            "Make a minimal code change.\n\n"
            "Evaluation hints:\n"
            "- Verification commands:\n"
            "- git diff is non-empty\n"
            "- Semantic checks:\n"
            "- artifact_mode=pr"
        )

        self.assertEqual(plan.smoke_commands, [])

    def test_numbered_verification_commands_become_smoke_commands(self) -> None:
        plan = build_verification_plan(
            "Implement crawler.py so it parses fixtures/products.html and writes product records to JSON.\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests\n"
            "2. python crawler.py fixtures/products.html --output out.json"
        )

        self.assertEqual(
            plan.smoke_commands,
            [
                "python -m unittest discover -s tests",
                "python crawler.py fixtures/products.html --output out.json",
            ],
        )
        self.assertFalse(plan.require_external_source_verified)
        self.assertTrue(plan.require_crawler_artifacts)

    def test_placeholder_check_allows_legitimate_pass_statement(self) -> None:
        self.assertFalse(diff_contains_placeholder("+except ValueError:\n+    pass\n"))
        self.assertTrue(diff_contains_placeholder("+    pass  # TODO implement later\n"))

    async def test_source_repair_accepts_fetch_evidence_without_test_change(self) -> None:
        result = await CodingVerifier(judge=VetoingJudge()).verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction=(
                    "Replace the broken data source URL in source_config.py with a documented working source and record "
                    "the verification evidence.\n\n"
                    "Evaluation hints:\n"
                    "- Verification commands:\n"
                    "- python3 -m py_compile source_config.py\n"
                    "- Semantic checks:\n"
                    "- fetch_url evidence required"
                ),
            ),
            SourceRepairVerifierTools(),
            evidence=VerificationEvidence(
                successful_fetch_urls=["https://jsonplaceholder.typicode.com/todos/1"],
                successful_web_search_queries=[],
                no_test_reason="No automated test framework detected; manual verification and py_compile were performed.",
            ),
        )

        self.assertTrue(result.passed)
        self.assertIsNone(result.llm_judgement)

    async def test_eval_verify_with_hint_becomes_smoke_command(self) -> None:
        plan = build_verification_plan(
            "Turn cli.py into a working command line tool that prints a greeting for --name.\n\n"
            "Evaluation hints:\n"
            "- verify with: python3 cli.py --name Ada"
        )

        self.assertEqual(plan.smoke_commands, ["python3 cli.py --name Ada"])

    async def test_cli_verify_with_hint_is_used_for_smoke(self) -> None:
        tools = CliHintVerifierTools()
        result = await CodingVerifier().verify(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction=(
                    "Turn cli.py into a working command line tool that prints a greeting for --name.\n\n"
                    "Evaluation hints:\n"
                    "- verify with: python3 cli.py --name Ada"
                ),
            ),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIn("python3 cli.py --name Ada", tools.command)

    async def test_external_source_requires_tool_evidence_not_diff_url(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="add API adapter for external endpoint"),
            ApiVerifierTools(),
        )

        self.assertFalse(result.passed)
        self.assertIn("verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run", result.required_fixes)

    async def test_external_source_accepts_successful_fetch_url_evidence(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="add API adapter for external endpoint"),
            ApiVerifierTools(),
            evidence=VerificationEvidence(
                successful_fetch_urls=["https://api.example.test/docs"],
                successful_web_search_queries=[],
                relevant_fetch_urls=["https://api.example.test/docs"],
            ),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.evidence.successful_fetch_urls, ["https://api.example.test/docs"])
        self.assertEqual(result.evidence.relevant_fetch_urls, ["https://api.example.test/docs"])

    async def test_external_source_accepts_successful_fetch_url_when_written_to_diff(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Replace the broken data source URL in source_config.py with a documented working source."),
            ApiVerifierTools(),
            evidence=VerificationEvidence(
                successful_fetch_urls=["https://api.example.test/v1/items"],
                successful_web_search_queries=[],
                relevant_fetch_urls=[],
            ),
        )

        self.assertTrue(result.passed)

    async def test_artifact_export_verifier_skips_standard_commands_and_smoke(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="Make a minimal code change and prepare the job for PR artifact export mode."),
            ArtifactExportVerifierTools(),
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.verification_plan.artifact_export)
        self.assertEqual(result.smoke_result.metadata["detected"], False)
        self.assertEqual(result.test_result.metadata["skipped"], True)

    def test_verification_evidence_from_steps_collects_successful_source_tools(self) -> None:
        steps = [
            {
                "type": "tool_result",
                "tool": "fetch_url",
                "exit_code": 0,
                "metadata": {"url": "https://example.test/docs", "goal": "api auth", "returned_bytes": 100, "status_code": 200},
                "output": json.dumps({"confidence": "medium", "relevant_sections": [{"heading": "Auth", "text": "token"}]}),
            },
            {"type": "tool_result", "tool": "web_search", "exit_code": 0, "metadata": {"query": "example api docs"}},
            {"type": "tool_result", "tool": "fetch_url", "exit_code": 1, "metadata": {"url": "https://bad.test"}},
        ]

        evidence = verification_evidence_from_steps(steps)

        self.assertEqual(evidence.successful_fetch_urls, ["https://example.test/docs"])
        self.assertEqual(evidence.relevant_fetch_urls, ["https://example.test/docs"])
        self.assertEqual(evidence.successful_web_search_queries, ["example api docs"])

    async def test_external_source_rejects_unrelated_fetch_url_evidence(self) -> None:
        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="add API adapter for external endpoint"),
            ApiVerifierTools(),
            evidence=VerificationEvidence(
                successful_fetch_urls=["https://example.test/home"],
                successful_web_search_queries=[],
                relevant_fetch_urls=[],
            ),
        )

        self.assertFalse(result.passed)
        self.assertIn("verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run", result.required_fixes)

    async def test_cli_plan_runs_python_entrypoint(self) -> None:
        tools = NoDetectedPythonVerifierTools(smoke_exit_code=0)

        result = await CodingVerifier().verify(
            CodingJob(id=new_id("job"), user_id="u1", instruction="add a CLI script"),
            tools,
        )

        self.assertTrue(result.passed)
        self.assertIsNotNone(result.verification_plan)
        self.assertTrue(result.verification_plan.require_entrypoint_run)
        self.assertIn("python3 crawler.py", result.smoke_result.metadata["command"])
