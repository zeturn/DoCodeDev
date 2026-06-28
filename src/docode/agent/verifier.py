from __future__ import annotations

import re
import shlex
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
    smoke_result: ToolResult | None = None
    workspace_result: ToolResult | None = None
    llm_judgement: VerifierJudgement | None = None
    verification_plan: "VerificationPlan | None" = None


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    required_commands: list[str]
    smoke_commands: list[str]
    require_test_change: bool = False
    require_entrypoint_run: bool = False
    require_no_placeholder: bool = True
    require_external_source_verified: bool = False


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
    def __init__(self, judge: VerifierJudge | None = None) -> None:
        self.judge = judge

    async def verify(self, job: CodingJob, tools: DoBoxTools) -> VerificationResult:
        plan = build_verification_plan(job.instruction)
        await prepare_workspace_for_diff(tools)
        status_result = await safe_tool_call("git_status", tools.git_status)
        diff_result = await safe_tool_call("git_diff", tools.git_diff)
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
        has_change_evidence = has_diff or has_explicit_artifact
        tests_ok = test_result.exit_code == 0
        build_ok = build_result.exit_code == 0
        lint_ok = lint_result.exit_code == 0
        verified_diff = diff_result.output if has_diff else synthetic_diff_from_status(status_result.output, workspace_result)
        smoke_result = await run_smoke_verification(job, tools, verified_diff, test_result, build_result, lint_result, plan)
        smoke_ok = smoke_result.exit_code == 0
        plan_ok, plan_fixes = evaluate_verification_plan(plan, verified_diff, test_result, build_result, lint_result, smoke_result)

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
        fixes.extend(fix for fix in plan_fixes if fix not in fixes)

        judgement = await self._judge(job, status_result, verified_diff, test_result, build_result, lint_result, smoke_result)
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
                return await self.judge.judge(
                    instruction=job.instruction,
                    status=status_result,
                    diff=diff,
                    tests=test_result,
                    build=build_result,
                    lint=lint_result,
                    smoke=smoke_result,
                )
            except TypeError as exc:
                if "smoke" in str(exc):
                    return await self.judge.judge(
                        instruction=job.instruction,
                        status=status_result,
                        diff=diff,
                        tests=test_result,
                        build=build_result,
                        lint=lint_result,
                    )
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

    commands: list[str] = []
    python_files = [path for path in changed_files if path.endswith(".py")]
    if python_files:
        commands.append("python3 -m py_compile " + " ".join(shlex.quote(path) for path in python_files))
        runnable = runnable_python_files(job.instruction, python_files)
        if runnable or plan.require_entrypoint_run:
            runnable = runnable or runnable_python_entrypoints(python_files)
        if runnable:
            commands.extend("python3 " + shlex.quote(path) for path in runnable[:1])
            if requires_csv_output_check(job.instruction):
                commands.append("python3 -c " + double_quote_shell_arg(csv_output_check_script()))
            elif requires_json_output_check(job.instruction):
                commands.append("python3 -c " + double_quote_shell_arg(json_output_check_script(minimum_required_records(job.instruction))))

    javascript_files = [path for path in changed_files if path.endswith((".js", ".mjs", ".cjs"))]
    for path in javascript_files:
        commands.append(f"command -v node >/dev/null 2>&1 && node --check {shlex.quote(path)}")

    json_files = [path for path in changed_files if path.endswith(".json")]
    for path in json_files:
        commands.append(f"python3 -m json.tool {shlex.quote(path)} >/dev/null")

    if not commands:
        if not has_code_like_changes(changed_files):
            return ToolResult(tool="run_smoke", output="no code-like changed files detected; smoke verification skipped", exit_code=0, metadata={"detected": False, "changed_files": changed_files})
        if any_detected(test_result, build_result, lint_result):
            return ToolResult(tool="run_smoke", output="standard verification commands were detected; no additional smoke command selected", exit_code=0, metadata={"detected": False})
        return ToolResult(tool="run_smoke", output="no task-appropriate smoke command detected", exit_code=1, metadata={"detected": False, "changed_files": changed_files})

    commands.extend(plan.smoke_commands)

    command = " && ".join(commands)
    result = await safe_optional_tool_call("run_command", tools, command, "/workspace")
    exit_code = result.exit_code
    if exit_code == 0 and smoke_output_indicates_failure(result.output):
        exit_code = 1
    return ToolResult(
        tool="run_smoke",
        output=result.output,
        exit_code=exit_code,
        metadata={"detected": True, "command": command, "changed_files": changed_files},
        truncated=result.truncated,
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
        if path and path != "/dev/null" and path not in files:
            files.append(path)
    return files


def build_verification_plan(instruction: str) -> VerificationPlan:
    lowered = (instruction or "").lower()
    is_crawler = any(keyword in lowered for keyword in ("crawler", "scraper", "scrape", "爬虫", "抓取", "采集", "数据源"))
    is_cli = any(keyword in lowered for keyword in ("cli", "command line", "命令行", "script", "脚本"))
    is_api = any(keyword in lowered for keyword in ("api", "adapter", "integration", "endpoint", "接口"))
    is_bugfix = any(keyword in lowered for keyword in ("bug", "fix", "regression", "修复", "报错", "失败"))
    is_docs = any(keyword in lowered for keyword in ("readme", "docs", "documentation", "文档"))
    required_commands: list[str] = []
    if is_crawler:
        required_commands.append("crawler_dry_run_or_entrypoint")
    if is_cli:
        required_commands.append("cli_entrypoint")
    if is_api:
        required_commands.append("api_contract_or_mock")
    if is_bugfix and not is_docs:
        required_commands.append("related_test")
    return VerificationPlan(
        required_commands=required_commands,
        smoke_commands=[],
        require_test_change=is_bugfix and not is_docs,
        require_entrypoint_run=is_crawler or is_cli,
        require_no_placeholder=not is_docs,
        require_external_source_verified=is_crawler or is_api,
    )


def evaluate_verification_plan(
    plan: VerificationPlan,
    diff: str,
    test_result: ToolResult,
    build_result: ToolResult,
    lint_result: ToolResult,
    smoke_result: ToolResult,
) -> tuple[bool, list[str]]:
    _ = build_result, lint_result
    fixes: list[str] = []
    changed_files = changed_files_from_diff(diff)
    diff_lowered = diff.lower()
    if plan.require_test_change and not has_test_change(changed_files):
        fixes.append("add or update a related test for this bugfix, or record why no automated test is appropriate")
    if plan.require_entrypoint_run and not (smoke_result.metadata and smoke_result.metadata.get("detected")):
        fixes.append("run the task entrypoint or CLI command as smoke verification")
    if plan.require_no_placeholder and diff_contains_placeholder(diff_lowered):
        fixes.append("remove placeholder/TODO/stub implementation text before finishing")
    if plan.require_external_source_verified and not external_source_verified(diff_lowered, smoke_result):
        fixes.append("verify the external API/data source with fetch_url or web_search evidence and a successful smoke/dry-run")
    return not fixes, fixes


def has_test_change(changed_files: list[str]) -> bool:
    return any("/test" in path or path.startswith("test") or path.startswith("tests/") for path in changed_files)


def diff_contains_placeholder(diff_lowered: str) -> bool:
    return any(marker in diff_lowered for marker in ("todo", "placeholder", "stub", "not implemented", "pass  #", "pass\n"))


def external_source_verified(diff_lowered: str, smoke_result: ToolResult) -> bool:
    if smoke_result.exit_code == 0 and smoke_result.metadata and smoke_result.metadata.get("detected"):
        return True
    return any(marker in diff_lowered for marker in ("fetch_url", "web_search", "http://", "https://", "requests.", "httpx.", "urllib."))


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


def changed_paths_from_status(status: str) -> list[str]:
    paths: list[str] = []
    for raw_line in status.splitlines():
        line = strip_ansi(raw_line).rstrip()
        if len(line) < 4:
            continue
        marker = line[:2]
        path = line[3:].strip()
        if not path:
            continue
        if marker == "??" or marker.strip():
            paths.append(path)
    return paths


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


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


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
    return any(keyword in lowered for keyword in ("json", "output.json"))


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


def csv_output_check_script() -> str:
    return (
        "import glob, os, sys; "
        "files=[p for p in glob.glob('*.csv') if os.path.isfile(p) and os.path.getsize(p)>0]; "
        "print('CSV outputs:', ', '.join(files)); "
        "sys.exit(0 if files else 1)"
    )


def json_output_check_script(min_records: int = 1) -> str:
    return (
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
