from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob


@dataclass(frozen=True, slots=True)
class VerifierJudgement:
    passed: bool
    confidence: float
    reason: str
    required_fixes: list[str]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    confidence: float
    reason: str
    required_fixes: list[str]
    git_status: str = ""
    git_diff: str = ""
    status_result: ToolResult | None = None
    test_result: ToolResult | None = None
    build_result: ToolResult | None = None
    lint_result: ToolResult | None = None
    workspace_result: ToolResult | None = None
    llm_judgement: VerifierJudgement | None = None


class VerifierJudge(Protocol):
    async def judge(
        self,
        *,
        instruction: str,
        status: ToolResult,
        diff: str,
        tests: ToolResult,
        build: ToolResult,
        lint: ToolResult,
    ) -> VerifierJudgement: ...


class CodingVerifier:
    def __init__(self, judge: VerifierJudge | None = None) -> None:
        self.judge = judge

    async def verify(self, job: CodingJob, tools: DoBoxTools) -> VerificationResult:
        status_result = await safe_tool_call("git_status", tools.git_status)
        diff_result = await safe_tool_call("git_diff", tools.git_diff)
        test_result = await safe_tool_call("run_tests", tools.run_tests)
        build_result = await safe_tool_call("run_build", tools.run_build)
        lint_result = await safe_tool_call("run_lint", tools.run_lint)
        non_git_workspace = is_non_git_status(status_result)
        workspace_result: ToolResult | None = None
        has_explicit_artifact = False
        if non_git_workspace and not job.repo_url:
            workspace_result = await safe_optional_tool_call("list_files", tools, ".")
            has_explicit_artifact = workspace_result.exit_code == 0 and bool(workspace_result.output.strip()) and not workspace_result.truncated

        status_ok = status_result.exit_code == 0 or has_explicit_artifact
        status_complete = not status_result.truncated
        has_diff = diff_result.exit_code == 0 and bool(diff_result.output.strip())
        diff_complete = not diff_result.truncated
        has_change_evidence = has_diff or has_explicit_artifact
        tests_ok = test_result.exit_code == 0
        build_ok = build_result.exit_code == 0
        lint_ok = lint_result.exit_code == 0
        verified_diff = diff_result.output if has_diff else ""

        fixes: list[str] = []
        if not status_ok:
            fixes.append("fix failing git status command")
        if not status_complete:
            fixes.append("reduce or split the change so the complete git status can be inspected")
        if not has_change_evidence:
            fixes.append("produce a non-empty git diff or explicit artifact")
        if has_diff and not diff_complete:
            fixes.append("reduce or split the change so the complete git diff can be exported")
        if not tests_ok:
            fixes.append("fix failing verification command")
        if not build_ok:
            fixes.append("fix failing build command")
        if not lint_ok:
            fixes.append("fix failing lint command")

        judgement = await self._judge(job, status_result, verified_diff, test_result, build_result, lint_result)
        if judgement is not None and not judgement.passed:
            fixes.extend(fix for fix in judgement.required_fixes if fix not in fixes)

        command_checks_passed = status_ok and status_complete and has_change_evidence and diff_complete and tests_ok and build_ok and lint_ok
        llm_checks_passed = judgement is None or judgement.passed
        if command_checks_passed and llm_checks_passed:
            confidence = min(judgement.confidence, 0.95) if judgement is not None else 0.86
            reason = judgement.reason if judgement is not None else verification_success_reason(has_explicit_artifact)
            return VerificationResult(
                passed=True,
                confidence=confidence,
                reason=reason,
                required_fixes=[],
                git_status=status_result.output,
                git_diff=verified_diff,
                status_result=status_result,
                test_result=test_result,
                build_result=build_result,
                lint_result=lint_result,
                workspace_result=workspace_result,
                llm_judgement=judgement,
            )

        reason = f"Verification failed for instruction: {job.instruction}"
        if judgement is not None and judgement.reason:
            reason = f"{reason}; verifier model: {judgement.reason}"
        return VerificationResult(
            passed=False,
            confidence=min(judgement.confidence, 0.65) if judgement is not None else 0.35,
            reason=reason,
            required_fixes=fixes,
            git_status=status_result.output,
            git_diff=verified_diff,
            status_result=status_result,
            test_result=test_result,
            build_result=build_result,
            lint_result=lint_result,
            workspace_result=workspace_result,
            llm_judgement=judgement,
        )

    async def _judge(
        self,
        job: CodingJob,
        status_result: ToolResult,
        diff: str,
        test_result: ToolResult,
        build_result: ToolResult,
        lint_result: ToolResult,
    ) -> VerifierJudgement | None:
        if self.judge is None:
            return None
        try:
            try:
                return await self.judge.judge(
                    instruction=job.instruction,
                    status=status_result,
                    diff=diff,
                    tests=test_result,
                    build=build_result,
                    lint=lint_result,
                )
            except TypeError as exc:
                if "status" not in str(exc):
                    raise
                return await self.judge.judge(
                    instruction=job.instruction,
                    diff=diff,
                    tests=test_result,
                    build=build_result,
                    lint=lint_result,
                )
        except Exception as exc:
            return VerifierJudgement(
                passed=False,
                confidence=0.0,
                reason=f"verifier_model_failed:{exc}",
                required_fixes=["retry verification with a valid structured verifier judgement"],
            )


async def safe_tool_call(tool_name: str, call) -> ToolResult:
    try:
        return await call()
    except Exception as exc:
        detail = str(exc)
        return ToolResult(
            tool=tool_name,
            output=f"{tool_name} failed: {detail}",
            exit_code=1,
            metadata={"exception_type": type(exc).__name__, "error": detail},
        )


async def safe_optional_tool_call(tool_name: str, tools: DoBoxTools, *args) -> ToolResult:
    call = getattr(tools, tool_name, None)
    if call is None:
        return ToolResult(
            tool=tool_name,
            output=f"{tool_name} unavailable",
            exit_code=1,
            metadata={"exception_type": "AttributeError", "error": f"{tool_name} unavailable"},
        )
    return await safe_tool_call(tool_name, lambda: call(*args))


def is_non_git_status(result: ToolResult) -> bool:
    return result.exit_code != 0 and "not a git repository" in result.output.lower()


def verification_success_reason(has_explicit_artifact: bool) -> str:
    if has_explicit_artifact:
        return "Workspace is not a git repository, but explicit workspace artifacts exist and detected tests/build/lint passed or were not detected."
    return "Git status succeeded, diff is non-empty, and detected tests/build/lint passed or were not detected."
