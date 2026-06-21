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


class RaisingVerifierTools(PassingVerifierTools):
    async def run_tests(self) -> ToolResult:
        raise RuntimeError("pytest crashed")


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
