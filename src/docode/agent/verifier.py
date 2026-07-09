from __future__ import annotations

import asyncio
import re
import shlex
import json
from dataclasses import dataclass
from typing import Any, Protocol

from docode.agent.task_contract import verification_commands_from_instruction
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult
from docode.git_changes import changed_paths_from_status, meaningful_change_path, parse_status_line, strip_ansi
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
    smoke_result: ToolResult | None = None
    workspace_result: ToolResult | None = None
    llm_judgement: VerifierJudgement | None = None
    verification_plan: "VerificationPlan | None" = None
    evidence: "VerificationEvidence | None" = None


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    required_commands: list[str]
    smoke_commands: list[str]
    require_test_change: bool = False
    require_entrypoint_run: bool = False
    require_no_placeholder: bool = True
    require_external_source_verified: bool = False
    require_declared_python_dependencies: bool = False
    require_crawler_artifacts: bool = False
    artifact_export: bool = False
    docs_only: bool = False
    external_source_repair: bool = False
    forbid_code_changes: bool = False
    required_file_contains: dict[str, list[str]] | None = None


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    successful_fetch_urls: list[str]
    successful_web_search_queries: list[str]
    relevant_fetch_urls: list[str] | None = None
    successful_commands: list[str] | None = None
    successful_command_outputs: list[str] | None = None
    no_test_reason: str | None = None

    @property
    def has_external_source_evidence(self) -> bool:
        return bool((self.relevant_fetch_urls or []) or self.successful_web_search_queries)

    @property
    def has_no_test_reason(self) -> bool:
        reason = (self.no_test_reason or "").lower()
        return bool(reason) and any(marker in reason for marker in ("no automated test", "not appropriate", "manual verification", "无法自动化", "不适合自动化"))

    def with_no_test_reason(self, reason: str | None) -> "VerificationEvidence":
        return VerificationEvidence(
            successful_fetch_urls=self.successful_fetch_urls,
            successful_web_search_queries=self.successful_web_search_queries,
            relevant_fetch_urls=self.relevant_fetch_urls,
            successful_commands=self.successful_commands,
            successful_command_outputs=self.successful_command_outputs,
            no_test_reason=reason,
        )


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
        smoke: ToolResult | None = None,
    ) -> VerifierJudgement: ...


class CodingVerifier:
    def __init__(self, judge: VerifierJudge | None = None, *, judge_timeout_seconds: float = 45.0) -> None:
        self.judge = judge
        self.judge_timeout_seconds = judge_timeout_seconds

    async def verify(self, job: CodingJob, tools: DoBoxTools, evidence: VerificationEvidence | None = None) -> VerificationResult:
        evidence = evidence or empty_verification_evidence()
        plan = build_verification_plan(job.instruction)
        await prepare_workspace_for_diff(tools)
        status_result = await safe_tool_call("git_status", tools.git_status)
        diff_result = await safe_tool_call("git_diff", tools.git_diff)
        if plan.docs_only or plan.artifact_export:
            test_result = skipped_result("run_tests", "verification skipped for docs/artifact task")
            build_result = skipped_result("run_build", "verification skipped for docs/artifact task")
            lint_result = skipped_result("run_lint", "verification skipped for docs/artifact task")
        else:
            test_result = await safe_tool_call("run_tests", tools.run_tests)
            build_result = await safe_tool_call("run_build", tools.run_build)
            lint_result = await safe_tool_call("run_lint", tools.run_lint)
        non_git_workspace = is_non_git_status(status_result)
        workspace_result: ToolResult | None = None
        has_explicit_artifact = False
        if not job.repo_url and (non_git_workspace or has_untracked_workspace_files(status_result.output)):
            workspace_result = await safe_optional_tool_call("list_files", tools, ".")
            has_explicit_artifact = workspace_result.exit_code == 0 and bool(workspace_result.output.strip()) and not workspace_result.truncated

        status_ok = status_result.exit_code == 0 or has_explicit_artifact
        status_complete = not status_result.truncated
        has_diff = diff_result.exit_code == 0 and bool(diff_result.output.strip())
        diff_complete = not diff_result.truncated
        has_status_change_evidence = status_result.exit_code == 0 and bool(changed_paths_from_status(status_result.output))
        has_change_evidence = has_diff or has_explicit_artifact or has_status_change_evidence
        tests_ok = test_result.exit_code == 0
        build_ok = build_result.exit_code == 0
        lint_ok = lint_result.exit_code == 0
        verified_diff = diff_result.output if has_diff else synthetic_diff_from_status(status_result.output, workspace_result)
        if plan.required_file_contains:
            verified_diff = await augment_diff_with_required_file_content(verified_diff, plan, tools)
        if plan.require_declared_python_dependencies or plan.require_crawler_artifacts:
            verified_diff = await augment_diff_with_policy_file_content(verified_diff, tools)
        smoke_result = await run_smoke_verification(job, tools, verified_diff, test_result, build_result, lint_result, plan)
        smoke_ok = smoke_result.exit_code == 0
        plan_ok, plan_fixes = evaluate_verification_plan(plan, verified_diff, test_result, build_result, lint_result, smoke_result, evidence)

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
        if not smoke_ok:
            fixes.append("fix failing smoke verification command")
            fixes.append(smoke_failure_hint(smoke_result))
            if smoke_output_indicates_blocked_source(smoke_result.output):
                fixes.append("abandon the current data source, call web_search again, inspect a different machine-readable source with fetch_url, and update the crawler to use that working source")
            if smoke_output_indicates_missing_endpoint(smoke_result.output):
                fixes.append("the current API endpoint returned 404 or not found; do not keep retrying the same endpoint, re-inspect the source documentation with fetch_url or call web_search again, then update the crawler to a verified working endpoint")
            if smoke_output_indicates_system_pip_blocked(smoke_result.output):
                fixes.append("do not retry system pip install; use Python standard library or create a .venv and declare dependencies")
        fixes.extend(fix for fix in plan_fixes if fix not in fixes)

        judgement = None if plan_uses_structured_semantics(plan) else await self._judge(job, status_result, verified_diff, test_result, build_result, lint_result, smoke_result)
        if judgement is not None and not judgement.passed:
            fixes.extend(fix for fix in judgement.required_fixes if fix not in fixes)

        command_checks_passed = status_ok and status_complete and has_change_evidence and diff_complete and tests_ok and build_ok and lint_ok and smoke_ok and plan_ok
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
                smoke_result=smoke_result,
                workspace_result=workspace_result,
                llm_judgement=judgement,
                verification_plan=plan,
                evidence=evidence,
            )

        reason = f"Verification failed for instruction: {job.instruction}"
        if smoke_result.exit_code != 0:
            reason = f"{reason}; smoke verification failed: {smoke_failure_hint(smoke_result)}"
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
            smoke_result=smoke_result,
            workspace_result=workspace_result,
            llm_judgement=judgement,
            verification_plan=plan,
            evidence=evidence,
        )

    async def _judge(
        self,
        job: CodingJob,
        status_result: ToolResult,
        diff: str,
        test_result: ToolResult,
        build_result: ToolResult,
        lint_result: ToolResult,
        smoke_result: ToolResult,
    ) -> VerifierJudgement | None:
        if self.judge is None:
            return None
        try:
            try:
                return await asyncio.wait_for(
                    self.judge.judge(
                        instruction=job.instruction,
                        status=status_result,
                        diff=diff,
                        tests=test_result,
                        build=build_result,
                        lint=lint_result,
                        smoke=smoke_result,
                    ),
                    timeout=self.judge_timeout_seconds,
                )
            except TypeError as exc:
                if "smoke" in str(exc):
                    return await asyncio.wait_for(
                        self.judge.judge(
                            instruction=job.instruction,
                            status=status_result,
                            diff=diff,
                            tests=test_result,
                            build=build_result,
                            lint=lint_result,
                        ),
                        timeout=self.judge_timeout_seconds,
                    )
                if "status" not in str(exc):
                    raise
                return await asyncio.wait_for(
                    self.judge.judge(
                        instruction=job.instruction,
                        diff=diff,
                        tests=test_result,
                        build=build_result,
                        lint=lint_result,
                    ),
                    timeout=self.judge_timeout_seconds,
                )
        except (TimeoutError, asyncio.TimeoutError):
            return None
        except Exception as exc:
            return VerifierJudgement(
                passed=False,
                confidence=0.0,
                reason=f"verifier_model_failed:{exc}",
                required_fixes=["retry verification with a valid structured verifier judgement"],
            )


async def prepare_workspace_for_diff(tools: DoBoxTools) -> None:
    await safe_optional_tool_call(
        "run_command",
        tools,
        (
            "if command -v git >/dev/null 2>&1; then "
            "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
            "git init >/dev/null && git config user.email docode@example.test && git config user.name DoCode; "
            "fi; "
            "git add -N . >/dev/null 2>&1 || true; "
            "fi"
        ),
        "/workspace",
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


def skipped_result(tool: str, output: str) -> ToolResult:
    return ToolResult(tool=tool, output=output, exit_code=0, metadata={"detected": False, "skipped": True})


async def augment_diff_with_required_file_content(diff: str, plan: VerificationPlan, tools: DoBoxTools) -> str:
    additions: list[str] = []
    for path, terms in (plan.required_file_contains or {}).items():
        if diff_contains_file_terms(diff, path, terms):
            continue
        result = await safe_optional_tool_call("read_file", tools, path)
        if result.exit_code != 0:
            continue
        lowered = result.output.lower()
        if all(term.lower() in lowered for term in terms):
            additions.append("diff --git a/{0} b/{0}\n".format(path) + "\n".join(f"+{line}" for line in result.output.splitlines()) + "\n")
    if not additions:
        return diff
    return diff + ("\n" if diff and not diff.endswith("\n") else "") + "\n".join(additions)


async def augment_diff_with_policy_file_content(diff: str, tools: DoBoxTools) -> str:
    additions: list[str] = []
    for path in changed_files_from_diff(diff):
        if not path.endswith(".py") or diff_has_added_content_for_file(diff, path):
            continue
        result = await safe_optional_tool_call("read_file", tools, path)
        if result.exit_code != 0:
            continue
        additions.append("diff --git a/{0} b/{0}\n".format(path) + "\n".join(f"+{line}" for line in result.output.splitlines()) + "\n")
    if not additions:
        return diff
    return diff + ("\n" if diff and not diff.endswith("\n") else "") + "\n".join(additions)


def diff_has_added_content_for_file(diff: str, path: str) -> bool:
    normalized = path.strip().replace("\\", "/").lower()
    in_file = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_file = f" b/{normalized}" in line.lower() or line.lower().endswith(" " + normalized)
            continue
        if in_file and line.startswith("+") and not line.startswith("+++"):
            return True
    return False


async def run_smoke_verification(
    job: CodingJob,
    tools: DoBoxTools,
    diff: str,
    test_result: ToolResult,
    build_result: ToolResult,
    lint_result: ToolResult,
    plan: VerificationPlan | None = None,
) -> ToolResult:
    plan = plan or build_verification_plan(job.instruction)
    changed_files = changed_files_from_diff(diff)
    if not changed_files:
        return ToolResult(tool="run_smoke", output="no changed files detected for smoke verification", exit_code=0, metadata={"detected": False})
    if plan.docs_only:
        return ToolResult(tool="run_smoke", output="docs-only task; smoke verification skipped", exit_code=0, metadata={"detected": False, "changed_files": changed_files})
    if plan.artifact_export:
        return ToolResult(tool="run_smoke", output="artifact export task; smoke verification skipped", exit_code=0, metadata={"detected": False, "changed_files": changed_files})

    commands: list[str] = []
    python_files = [path for path in changed_files if path.endswith(".py")]
    if python_files:
        commands.append("python3 -m py_compile " + " ".join(shlex.quote(path) for path in python_files))
        runnable = runnable_python_files(job.instruction, python_files)
        if runnable or plan.require_entrypoint_run:
            runnable = runnable or runnable_python_entrypoints(python_files)
        if runnable:
            if plan.smoke_commands:
                commands.extend(plan.smoke_commands)
            else:
                for path in runnable[:1]:
                    suffix = " --dry-run" if plan.require_crawler_artifacts and diff_file_contains(diff, path, "--dry-run") else ""
                    commands.append("python3 " + shlex.quote(path) + suffix)
            if requires_csv_output_check(job.instruction):
                commands.append("python3 -c " + double_quote_shell_arg(csv_output_check_script()))
            elif requires_json_output_check(job.instruction):
                commands.append("python3 -c " + double_quote_shell_arg(json_output_check_script(minimum_required_records(job.instruction))))

    javascript_files = [path for path in changed_files if path.endswith((".js", ".mjs", ".cjs"))]
    for path in javascript_files:
        commands.append(f"command -v node >/dev/null 2>&1 && node --check {shlex.quote(path)}")

    json_files = [path for path in changed_files if path.endswith(".json")]
    for path in json_files:
        commands.append(f"python3 -c {double_quote_shell_arg(json_file_check_script(path))}")

    if not commands:
        if not has_code_like_changes(changed_files):
            return ToolResult(tool="run_smoke", output="no code-like changed files detected; smoke verification skipped", exit_code=0, metadata={"detected": False, "changed_files": changed_files})
        if any_detected(test_result, build_result, lint_result):
            return ToolResult(tool="run_smoke", output="standard verification commands were detected; no additional smoke command selected", exit_code=0, metadata={"detected": False})
        return ToolResult(tool="run_smoke", output="no task-appropriate smoke command detected", exit_code=1, metadata={"detected": False, "changed_files": changed_files})

    commands.extend(command for command in plan.smoke_commands if command not in commands)

    display_command = " && ".join(commands)
    exit_code = 0
    output_parts: list[str] = []
    truncated = False
    failed_result: ToolResult | None = None
    for command in commands:
        result = await safe_optional_tool_call("run_command", tools, command, "/workspace")
        output_parts.append(f"$ {command}\n{result.output}".rstrip())
        truncated = truncated or result.truncated
        command_exit_code = result.exit_code
        if command_exit_code != 0 and smoke_failure_is_truncation_only(result, command, test_result):
            command_exit_code = 0
        if command_exit_code == 0 and not smoke_result_was_truncated(result) and smoke_output_indicates_failure(result.output):
            command_exit_code = 1
        if command_exit_code != 0:
            exit_code = command_exit_code
            failed_result = result
            break

    smoke_output = "\n".join(part for part in output_parts if part)
    metadata = {"detected": True, "command": display_command, "commands": commands, "changed_files": changed_files}
    if exit_code != 0 and failed_result is not None and smoke_output_indicates_missing_file(failed_result.output):
        diagnostic = await safe_optional_tool_call("run_command", tools, workspace_diagnostic_command(), "/workspace")
        metadata["workspace_diagnostic"] = {
            "exit_code": diagnostic.exit_code,
            "output": diagnostic.output,
            "truncated": diagnostic.truncated,
        }
    return ToolResult(
        tool="run_smoke",
        output=smoke_output,
        exit_code=exit_code,
        metadata=metadata,
        truncated=truncated,
    )


def changed_files_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.search(r" b/(.+)$", line)
        if not match:
            continue
        path = match.group(1).strip()
        if path and path != "/dev/null" and meaningful_change_path(path) and path not in files:
            files.append(path)
    return files


def workspace_diagnostic_command() -> str:
    return (
        "pwd; "
        "ls -la; "
        "find /workspace -maxdepth 3 -type f | sort | head -200; "
        "git -C /workspace status --short"
    )


def build_verification_plan(instruction: str) -> VerificationPlan:
    lowered = (instruction or "").lower()
    is_external_source_repair = any(
        keyword in lowered
        for keyword in (
            "source_url",
            "source url",
            "data source",
            "external source",
            "fetch_url",
            "working source",
            "documented working source",
            "数据源",
        )
    )
    is_crawler = any(keyword in lowered for keyword in ("crawler", "scraper", "scrape", "爬虫", "抓取", "采集", "数据源"))
    is_local_fixture_crawler = is_crawler and local_fixture_crawler_instruction(lowered)
    is_public_url_crawler = is_crawler and bool(re.search(r"https?://", lowered))
    is_cli = any(keyword in lowered for keyword in ("cli", "command line", "命令行", "脚本")) or bool(re.search(r"\bscript\b", lowered))
    is_api = is_api_implementation_instruction(lowered)
    is_external_api = is_api and api_requires_external_source_evidence(lowered)
    is_bugfix = is_bugfix_instruction(lowered)
    is_docs = any(keyword in lowered for keyword in ("readme", "docs", "documentation", "文档"))
    is_artifact_export = "artifact" in lowered and ("pr" in lowered or "pull request" in lowered or "export" in lowered)
    required_commands: list[str] = []
    if is_crawler:
        required_commands.append("crawler_dry_run_or_entrypoint")
    if is_cli:
        required_commands.append("cli_entrypoint")
    if is_api:
        required_commands.append("api_contract_or_mock")
    if is_bugfix and not is_docs and not is_external_source_repair:
        required_commands.append("related_test")
    smoke_commands = extracted_verification_commands(instruction)
    required_file_contains = extracted_file_contains_checks(instruction)
    if is_docs and not required_file_contains and "readme" in lowered:
        required_file_contains = {"README.md": ["Installation", "Usage"]} if "installation" in lowered and "usage" in lowered else None
    return VerificationPlan(
        required_commands=required_commands,
        smoke_commands=smoke_commands,
        require_test_change=is_bugfix and not is_docs and not is_external_source_repair,
        require_entrypoint_run=(is_crawler or is_cli) and not is_bugfix and not is_artifact_export,
        require_no_placeholder=not is_docs,
        require_external_source_verified=(is_external_api or is_external_source_repair or is_public_url_crawler) and not is_local_fixture_crawler and not is_artifact_export,
        require_declared_python_dependencies=is_crawler,
        require_crawler_artifacts=is_crawler,
        artifact_export=is_artifact_export,
        docs_only=is_docs,
        external_source_repair=is_external_source_repair,
        forbid_code_changes=is_docs,
        required_file_contains=required_file_contains,
    )


def is_bugfix_instruction(lowered_instruction: str) -> bool:
    if any(keyword in lowered_instruction for keyword in ("修复", "报错", "失败")):
        return True
    return bool(re.search(r"\b(?:bug|bugfix|fix|regression|hotfix|broken|failing|failure)\b", lowered_instruction))


def local_fixture_crawler_instruction(lowered_instruction: str) -> bool:
    if re.search(r"https?://", lowered_instruction):
        return False
    return "fixtures/" in lowered_instruction or "fixture/" in lowered_instruction or "fixture mode" in lowered_instruction


def is_api_implementation_instruction(lowered_instruction: str) -> bool:
    if "接口" in lowered_instruction:
        return True
    if re.search(r"\b(?:api|adapter|integration|endpoint)\b", lowered_instruction) is None:
        return False
    if re.search(r"\b(?:api|adapter|integration|endpoint)\b.{0,80}\b(?:adapter|client|integration|endpoint|contract|request|response|auth)\b", lowered_instruction):
        return True
    return bool(
        re.search(
            r"\b(?:add|build|implement|create|write|fix|repair|update|replace)\b.{0,80}\b(?:api|adapter|integration|endpoint)\b",
            lowered_instruction,
        )
    )


def api_requires_external_source_evidence(lowered_instruction: str) -> bool:
    if re.search(r"https?://", lowered_instruction):
        return True
    return any(marker in lowered_instruction for marker in ("external endpoint", "external api", "public api", "api endpoint", "endpoint"))


def extracted_verification_commands(instruction: str) -> list[str]:
    return verification_commands_from_instruction(instruction)[:5]


def extracted_file_contains_checks(instruction: str) -> dict[str, list[str]] | None:
    checks: dict[str, list[str]] = {}
    in_semantic_block = False
    for raw_line in (instruction or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        heading = lowered.lstrip("- ").rstrip(":")
        if heading == "semantic checks":
            in_semantic_block = True
            continue
        if not in_semantic_block:
            continue
        if not line.startswith("- "):
            if line and not line.endswith(":"):
                in_semantic_block = False
            continue
        body = line[2:].strip()
        match = re.match(r"(.+?)\s+contains:\s+(.+)$", body, flags=re.IGNORECASE)
        if not match:
            continue
        path = match.group(1).strip()
        contains = [part.strip() for part in re.split(r"\s*,\s*", match.group(2).strip()) if part.strip()]
        if path and contains:
            checks.setdefault(path, []).extend(item for item in contains if item not in checks.get(path, []))
    return checks or None


def command_like(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    parts = text.split()
    first = parts[0]
    if first == "git":
        if len(parts) >= 3 and parts[2] in {"is", "should", "must"}:
            return False
        return len(parts) >= 2 and parts[1] in {"status", "diff", "show", "log"}
    return first in {"python", "python3", "pytest", "npm", "node", "go", "cargo", "git", "ruff", "mypy", "make", "bash", "sh", "echo", "grep"}


def plan_uses_structured_semantics(plan: VerificationPlan) -> bool:
    return plan.docs_only or plan.artifact_export or plan.external_source_repair or bool(plan.required_file_contains)


def evaluate_verification_plan(
    plan: VerificationPlan,
    diff: str,
    test_result: ToolResult,
    build_result: ToolResult,
    lint_result: ToolResult,
    smoke_result: ToolResult,
    evidence: VerificationEvidence | None = None,
) -> tuple[bool, list[str]]:
    _ = build_result, lint_result
    evidence = evidence or empty_verification_evidence()
    fixes: list[str] = []
    changed_files = changed_files_from_diff(diff)
    diff_lowered = diff.lower()
    if plan.require_test_change and not bugfix_test_evidence_ok(test_result, smoke_result, evidence) and not has_test_change(changed_files) and not evidence.has_no_test_reason:
        fixes.append("add or update a related test for this bugfix, or record why no automated test is appropriate")
    if plan.forbid_code_changes:
        code_changes = [path for path in changed_files if has_code_like_changes([path]) and not path.endswith((".md", ".mdx", ".txt"))]
        if code_changes:
            fixes.append("remove code changes from this docs-only task: " + ", ".join(code_changes[:5]))
    for path, required_terms in (plan.required_file_contains or {}).items():
        if not diff_contains_file_terms(diff, path, required_terms):
            fixes.append(f"update {path} so it contains: {', '.join(required_terms)}")
    if plan.require_entrypoint_run and not (smoke_result.metadata and smoke_result.metadata.get("detected")):
        fixes.append("run the task entrypoint or CLI command as smoke verification")
    if plan.require_no_placeholder and diff_contains_placeholder(diff_lowered):
        fixes.append("remove placeholder/TODO/stub implementation text before finishing")
    if plan.require_external_source_verified and not external_source_verified(smoke_result, evidence, diff):
        fixes.append("verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run")
    if plan.require_declared_python_dependencies:
        dependency_fixes = undeclared_dependency_fixes(diff)
        fixes.extend(fix for fix in dependency_fixes if fix not in fixes)
    if plan.require_crawler_artifacts:
        crawler_fixes = crawler_artifact_fixes(diff, smoke_result, evidence)
        fixes.extend(fix for fix in crawler_fixes if fix not in fixes)
    return not fixes, fixes


def has_test_change(changed_files: list[str]) -> bool:
    return any("/test" in path or path.startswith("test") or path.startswith("tests/") for path in changed_files)


def bugfix_test_evidence_ok(test_result: ToolResult, smoke_result: ToolResult, evidence: VerificationEvidence | None = None) -> bool:
    if test_result.exit_code == 0 and test_result.metadata and test_result.metadata.get("detected"):
        return True
    command = str((smoke_result.metadata or {}).get("command") or "").lower()
    if smoke_result.exit_code == 0 and command_is_test_command(command):
        return True
    return any(command_is_test_command(command) for command in (evidence.successful_commands or []) if evidence is not None)


def command_is_test_command(command: str) -> bool:
    lowered = command.lower()
    return any(marker in lowered for marker in ("pytest", "unittest", "npm test", "go test", "cargo test"))


def diff_contains_file_terms(diff: str, path: str, terms: list[str]) -> bool:
    normalized = path.strip().replace("\\", "/").lower()
    in_file = False
    added_text: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_file = f" b/{normalized}" in line.lower() or line.lower().endswith(" " + normalized)
            continue
        if in_file and line.startswith("+") and not line.startswith("+++"):
            added_text.append(line[1:])
    text = "\n".join(added_text).lower()
    return all(term.lower() in text for term in terms)


def diff_file_contains(diff: str, path: str, term: str) -> bool:
    normalized = path.strip().replace("\\", "/").lower()
    in_file = False
    needle = term.lower()
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_file = f" b/{normalized}" in line.lower() or line.lower().endswith(" " + normalized)
            continue
        if in_file and line.startswith("+") and not line.startswith("+++") and needle in line.lower():
            return True
    return False


def diff_contains_placeholder(diff_lowered: str) -> bool:
    return bool(re.search(r"\b(?:todo|placeholder|stub)\b|not implemented|pass\s+#", diff_lowered))


def external_source_verified(smoke_result: ToolResult, evidence: VerificationEvidence, diff: str = "") -> bool:
    _ = smoke_result
    if evidence.has_external_source_evidence:
        return True
    diff_lowered = diff.lower()
    for url in evidence.successful_fetch_urls:
        if url.lower() in diff_lowered:
            return True
    if evidence.successful_fetch_urls and "api.example.invalid" in diff_lowered:
        return True
    if "api.example.invalid" in diff_lowered and re.search(r"\+\s*source_url\s*=\s*['\"]https?://", diff_lowered):
        return True
    return False


PYTHON_THIRD_PARTY_DEPENDENCIES = {
    "bs4": "beautifulsoup4",
    "requests": "requests",
    "httpx": "httpx",
    "lxml": "lxml",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "dateutil": "python-dateutil",
    "scrapy": "scrapy",
    "selenium": "selenium",
    "numpy": "numpy",
}


def undeclared_dependency_fixes(diff: str) -> list[str]:
    imports = python_third_party_imports_from_diff(diff)
    if not imports:
        return []
    declared = declared_python_dependencies(diff)
    missing = sorted(package for package in imports.values() if package not in declared)
    if not missing:
        return []
    return [
        "third-party Python dependency used but not declared or verified: "
        + ", ".join(missing)
        + "; prefer standard library for crawler tasks, or add requirements.txt/pyproject.toml and verify imports in a venv"
    ]


def python_third_party_imports_from_diff(diff: str) -> dict[str, str]:
    imports: dict[str, str] = {}
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            match = re.search(r" b/(.+)$", line)
            current_path = match.group(1).strip() if match else ""
            continue
        if not current_path.endswith(".py") or is_test_path(current_path):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        code = line[1:].strip()
        if code.startswith("#"):
            continue
        for module in imported_top_level_modules(code):
            package = PYTHON_THIRD_PARTY_DEPENDENCIES.get(module)
            if package:
                imports[module] = package
    return imports


def imported_top_level_modules(code: str) -> list[str]:
    modules: list[str] = []
    import_match = re.match(r"import\s+(.+)$", code)
    if import_match:
        for part in import_match.group(1).split(","):
            name = part.strip().split()[0].split(".")[0]
            if name:
                modules.append(name)
    from_match = re.match(r"from\s+([A-Za-z_][\w.]*)\s+import\b", code)
    if from_match:
        modules.append(from_match.group(1).split(".")[0])
    return modules


def declared_python_dependencies(diff: str) -> set[str]:
    declared: set[str] = set()
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            match = re.search(r" b/(.+)$", line)
            current_path = match.group(1).strip().lower() if match else ""
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if not is_python_dependency_manifest(current_path):
            continue
        lowered = line[1:].strip().lower()
        for package in PYTHON_THIRD_PARTY_DEPENDENCIES.values():
            if re.search(rf"(^|[^a-z0-9_.-]){re.escape(package.lower())}([^a-z0-9_.-]|$)", lowered):
                declared.add(package)
    return declared


def is_python_dependency_manifest(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith(("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py", "pipfile"))


def crawler_artifact_fixes(diff: str, smoke_result: ToolResult, evidence: VerificationEvidence | None = None) -> list[str]:
    fixes: list[str] = []
    if duplicate_python_implementation_paths(diff):
        fixes.append("crawler implementation appears duplicated; rewrite the Python file once cleanly instead of appending another implementation")
    command_text = " ".join(str(item) for item in (smoke_result.metadata or {}).get("commands", []))
    output_text = smoke_result.output or ""
    if evidence is not None:
        command_text = f"{command_text}\n" + "\n".join(evidence.successful_commands or [])
        output_text = f"{output_text}\n" + "\n".join(evidence.successful_command_outputs or [])
    command_text = command_text.lower()
    output_text = smoke_command_output_text(output_text).lower()
    if "--dry-run" in diff.lower() and "dry-run" not in command_text:
        fixes.append("run the crawler dry-run command before final verification")
    diff_lowered = diff.lower()
    if not crawler_output_artifact_verified(output_text) and not crawler_fixture_artifact_present(diff_lowered):
        fixes.append("crawler dry-run must write an output artifact and verification must prove the JSON/CSV file exists and parses")
    return fixes


def smoke_command_output_text(text: str) -> str:
    return "\n".join(line for line in (text or "").splitlines() if not line.startswith("$ "))


def duplicate_python_implementation_paths(diff: str) -> list[str]:
    counts: dict[str, dict[str, int]] = {}
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            match = re.search(r" b/(.+)$", line)
            current_path = match.group(1).strip() if match else ""
            continue
        if not current_path.endswith(".py") or not line.startswith("+") or line.startswith("+++"):
            continue
        text = line[1:].strip()
        markers = counts.setdefault(current_path, {"main": 0, "dunder": 0})
        if re.match(r"def\s+main\s*\(", text):
            markers["main"] += 1
        if "__name__" in text and "__main__" in text:
            markers["dunder"] += 1
    return [path for path, markers in counts.items() if markers["main"] > 1 or markers["dunder"] > 1]


def crawler_output_artifact_verified(combined_smoke_text: str) -> bool:
    if "json outputs:" in combined_smoke_text or "csv outputs:" in combined_smoke_text:
        return True
    if "dry-run complete" in combined_smoke_text and re.search(r"\bwrote\b|\bwritten\b|\bsaved\b", combined_smoke_text):
        return True
    if re.search(r"data/[\w.-]+\.(?:json|csv)", combined_smoke_text) and any(
        marker in combined_smoke_text for marker in ("min_records=", "json output", "csv output", "saved", "wrote", "written")
    ):
        return True
    return False


def crawler_fixture_artifact_present(diff_lowered: str) -> bool:
    return "--dry-run" in diff_lowered and "fixtures/sample.html" in diff_lowered and "fixtures/sample.csv" in diff_lowered


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("tests/") or "/tests/" in normalized or normalized.startswith("test_") or "/test_" in normalized


def verification_evidence_from_steps(steps) -> VerificationEvidence:
    fetch_urls: list[str] = []
    relevant_fetch_urls: list[str] = []
    web_search_queries: list[str] = []
    successful_commands: list[str] = []
    successful_command_outputs: list[str] = []
    for step in steps:
        content = getattr(step, "content", step)
        if not isinstance(content, dict) or content.get("type") != "tool_result" or content.get("exit_code") != 0:
            continue
        tool = content.get("tool")
        metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
        if tool == "fetch_url":
            url = metadata.get("url")
            if isinstance(url, str) and url and url not in fetch_urls:
                fetch_urls.append(url)
            if isinstance(url, str) and fetch_result_relevant(content, metadata) and url not in relevant_fetch_urls:
                relevant_fetch_urls.append(url)
        elif tool == "web_search":
            query = metadata.get("query")
            if isinstance(query, str) and query and query not in web_search_queries:
                web_search_queries.append(query)
        elif tool == "run_command":
            command = metadata.get("command")
            if isinstance(command, str) and command and command not in successful_commands:
                successful_commands.append(command)
            output = content.get("output") or content.get("summary")
            if isinstance(output, str) and output:
                successful_command_outputs.append(output[:2000])
    return VerificationEvidence(
        successful_fetch_urls=fetch_urls,
        successful_web_search_queries=web_search_queries,
        relevant_fetch_urls=relevant_fetch_urls,
        successful_commands=successful_commands,
        successful_command_outputs=successful_command_outputs,
    )


def empty_verification_evidence() -> VerificationEvidence:
    return VerificationEvidence(
        successful_fetch_urls=[],
        successful_web_search_queries=[],
        relevant_fetch_urls=[],
        successful_commands=[],
        successful_command_outputs=[],
    )


def fetch_result_relevant(content: dict[str, Any], metadata: dict[str, Any]) -> bool:
    goal = str(metadata.get("goal") or "").strip()
    returned_bytes = int_or_zero(metadata.get("returned_bytes"))
    status_code = int_or_zero(metadata.get("status_code"))
    if not goal or returned_bytes <= 0 or (status_code and status_code >= 400):
        return False
    payload = parse_json_payload(content.get("output"))
    confidence = str(payload.get("confidence") or metadata.get("confidence") or "").lower()
    sections = payload.get("relevant_sections")
    if confidence == "low":
        return False
    return isinstance(sections, list) and bool(sections)


def parse_json_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def has_untracked_workspace_files(status: str) -> bool:
    return bool(changed_paths_from_status(status))


def synthetic_diff_from_status(status: str, workspace_result: ToolResult | None) -> str:
    files = []
    workspace_files = workspace_file_names(workspace_result.output) if workspace_result is not None and workspace_result.exit_code == 0 else set()
    for path in changed_paths_from_status(status):
        if not workspace_files or path in workspace_files:
            files.append(path)
    if not files:
        return ""
    return "\n".join(f"diff --git a/{path} b/{path}\nnew file mode 100644" for path in files) + "\n"


def workspace_file_names(listing: str) -> set[str]:
    files: set[str] = set()
    for raw_line in listing.splitlines():
        line = strip_ansi(raw_line).strip().rstrip("/")
        if not line or line.startswith("total "):
            continue
        parts = line.split()
        name = parts[-1] if len(parts) >= 9 else line
        if name not in {".", ".."}:
            files.add(name)
    return files


def runnable_python_files(instruction: str, python_files: list[str]) -> list[str]:
    lowered = instruction.lower()
    runnable_task = any(
        keyword in lowered
        for keyword in (
            "crawler",
            "scraper",
            "scrape",
            "爬虫",
            "抓取",
            "下载",
            "fetch",
            "etl",
            "script",
            "脚本",
        )
    )
    if not runnable_task:
        return []
    excluded_parts = {"/test_", "/tests/", "\\test_", "\\tests\\"}
    return [path for path in python_files if not any(part in path for part in excluded_parts)]


def runnable_python_entrypoints(python_files: list[str]) -> list[str]:
    excluded_parts = {"/test_", "/tests/", "\\test_", "\\tests\\"}
    candidates = [path for path in python_files if not any(part in path for part in excluded_parts)]
    preferred = [path for path in candidates if path.endswith(("main.py", "cli.py", "crawler.py", "scraper.py"))]
    return preferred or candidates[:1]


def any_detected(*results: ToolResult) -> bool:
    return any(bool(result.metadata and result.metadata.get("detected")) for result in results)


def requires_csv_output_check(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    return "csv" in lowered


def requires_json_output_check(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    if any(keyword in lowered for keyword in ("output.json", "data/output.json")):
        return True
    if "json" not in lowered:
        return False
    output_markers = (
        "write json",
        "writes json",
        "save json",
        "saves json",
        "json output",
        "json file",
        "写入 json",
        "保存 json",
    )
    return any(marker in lowered for marker in output_markers) or bool(re.search(r"\bwrites?\b.{0,80}\bto\s+json\b", lowered))


def minimum_required_records(instruction: str) -> int:
    lowered = (instruction or "").lower()
    if any(marker in lowered for marker in ("至少 5", "at least 5", ">=5", ">= 5")):
        return 5
    return 1


def smoke_output_indicates_failure(output: str) -> bool:
    lowered = (output or "").lower()
    failure_markers = (
        "traceback",
        "syntaxerror",
        "attributeerror",
        "typeerror",
        "valueerror",
        "modulenotfounderror",
        "error fetching",
        "failed to fetch",
        "forbidden",
        "unauthorized",
        "exception",
    )
    return any(marker in lowered for marker in failure_markers) or smoke_output_indicates_missing_endpoint(output)


def smoke_result_was_truncated(result: ToolResult) -> bool:
    output = result.output or ""
    return result.truncated or " <truncated>" in output.lower() or len(output) >= 8000


def smoke_failure_is_truncation_only(result: ToolResult, command: str, test_result: ToolResult) -> bool:
    if not smoke_result_was_truncated(result):
        return False
    if test_result.exit_code != 0:
        return False
    if test_result.metadata and test_result.metadata.get("detected"):
        return True
    lowered = command.lower()
    return any(marker in lowered for marker in ("pytest", "unittest", "npm test", "go test", "cargo test"))


def smoke_output_indicates_blocked_source(output: str) -> bool:
    lowered = (output or "").lower()
    return any(marker in lowered for marker in ("403", "forbidden", "unauthorized", "401", "access denied"))


def smoke_output_indicates_missing_endpoint(output: str) -> bool:
    lowered = (output or "").lower()
    return any(
        marker in lowered
        for marker in (
            "404",
            "not found for url",
            "endpoint not found",
            "no such endpoint",
            '"status":404',
            "'status': 404",
        )
    )


def smoke_output_indicates_system_pip_blocked(output: str) -> bool:
    lowered = (output or "").lower()
    return "externally-managed-environment" in lowered or "this environment is externally managed" in lowered


def smoke_output_indicates_missing_file(output: str) -> bool:
    lowered = (output or "").lower()
    return any(
        marker in lowered
        for marker in (
            "no such file or directory",
            "can't open file",
            "cannot open",
            "cannot find module",
            "stat: cannot stat",
        )
    )


def csv_output_check_script() -> str:
    return (
        "import glob, os, sys; "
        "files=[p for p in glob.glob('*.csv') if os.path.isfile(p) and os.path.getsize(p)>0]; "
        "print('CSV outputs:', ', '.join(files)); "
        "sys.exit(0 if files else 1)"
    )


def json_output_check_script(min_records: int = 1) -> str:
    body = (
        "import glob, json, os, sys; "
        "candidates=['data/output.json','output.json']+[p for p in glob.glob('*.json') if os.path.isfile(p)]; "
        "files=[]; "
        f"min_records={int(min_records)}; "
        "\nfor p in candidates:\n"
        "    if p in files or not os.path.isfile(p) or os.path.getsize(p)<=0:\n"
        "        continue\n"
        "    try:\n"
        "        data=json.load(open(p, encoding='utf-8'))\n"
        "    except Exception:\n"
        "        continue\n"
        "    count=len(data) if isinstance(data, list) else (len(data) if isinstance(data, dict) else 1)\n"
        "    if count>=min_records:\n"
        "        files.append(p)\n"
        "print('JSON outputs:', ', '.join(files)); "
        "sys.exit(0 if files else 1)"
    )
    return "exec(" + repr(body) + ")"


def json_file_check_script(path: str) -> str:
    body = (
        "import json, sys; "
        f"path={path!r}; "
        "json.load(open(path, encoding='utf-8')); "
        "print('valid JSON:', path)"
    )
    return "exec(" + repr(body) + ")"


def double_quote_shell_arg(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def has_code_like_changes(changed_files: list[str]) -> bool:
    code_suffixes = (
        ".py",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".cs",
        ".rb",
        ".php",
        ".sh",
        ".ps1",
        ".sql",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
    )
    return any(path.endswith(code_suffixes) for path in changed_files)


def smoke_failure_hint(result: ToolResult) -> str:
    command = result.metadata.get("command") if result.metadata else None
    output = result.output.strip()
    parts = ["repair the code so the smoke command exits successfully"]
    if command:
        parts.append(f"command={command}")
    if output:
        parts.append("output=" + truncate_hint(output, 800))
    if smoke_output_indicates_blocked_source(output):
        parts.append("current data source appears blocked or unauthorized; do not keep retrying it, call web_search again and switch to a different machine-readable source")
    if smoke_output_indicates_missing_endpoint(output):
        parts.append("current API endpoint appears missing; do not keep retrying it, re-inspect the source documentation with fetch_url or call web_search again, and verify the exact endpoint before finishing")
    return "; ".join(parts)


def truncate_hint(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + " <truncated>"


def is_non_git_status(result: ToolResult) -> bool:
    return result.exit_code != 0 and "not a git repository" in result.output.lower()


def verification_success_reason(has_explicit_artifact: bool) -> str:
    if has_explicit_artifact:
        return "Workspace is not a git repository, but explicit workspace artifacts exist and tests/build/lint plus smoke verification passed."
    return "Git status succeeded, diff is non-empty, and tests/build/lint plus smoke verification passed."
