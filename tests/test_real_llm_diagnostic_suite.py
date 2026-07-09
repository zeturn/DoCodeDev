from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.agent.workflow import commands_equivalent
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.tools import ToolDefinition
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, DocodeStep, JobStatus, new_id

from tests.test_real_llm_smoke import build_real_llm_or_skip
from tests.test_smoke_readme_job import RecordingRepository, normalize_path


REAL_LLM_SMOKE_ENABLED = os.getenv("DOCODE_REAL_LLM_SMOKE", "").lower() in {"1", "true", "yes", "on"}
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "repos" / "diagnostic"


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    name: str
    fixture: str
    instruction: str
    required_commands: tuple[str, ...]


DIAGNOSTIC_CASES = {
    "cli_output_bug": DiagnosticCase(
        name="cli_output_bug",
        fixture="cli_output_bug",
        instruction=(
            "Fix cli.py so the CLI writes the greeting JSON to the path passed via --output.\n\n"
            "Target file: cli.py\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests\n"
            "2. python cli.py --name Ada --output out.json"
        ),
        required_commands=("python -m unittest discover -s tests", "python cli.py --name Ada --output out.json"),
    ),
    "multifile_api_mismatch": DiagnosticCase(
        name="multifile_api_mismatch",
        fixture="multifile_api_mismatch",
        instruction=(
            "Fix the profile formatting bug across app.py and formatter.py.\n\n"
            "Target files: app.py, formatter.py\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests"
        ),
        required_commands=("python -m unittest discover -s tests",),
    ),
    "parser_edge_case": DiagnosticCase(
        name="parser_edge_case",
        fixture="parser_edge_case",
        instruction=(
            "Fix parser.py so it normalizes all item records from fixtures/items.json.\n\n"
            "Target file: parser.py\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests"
        ),
        required_commands=("python -m unittest discover -s tests",),
    ),
    "two_stage_repair": DiagnosticCase(
        name="two_stage_repair",
        fixture="two_stage_repair",
        instruction=(
            "Fix crawler.py so it parses records and the CLI writes JSON to --output.\n\n"
            "Target file: crawler.py\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests\n"
            "2. python crawler.py sample.json --output out.json"
        ),
        required_commands=("python -m unittest discover -s tests", "python crawler.py sample.json --output out.json"),
    ),
}


class DiagnosticLocalTools:
    def __init__(self, workspace: Path, *, test_command: str = "python -m unittest discover -s tests") -> None:
        self.workspace = workspace.resolve()
        self.test_command = test_command
        self.initial_files = self.snapshot_files()
        self.command_results: list[ToolResult] = []

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition("run_command", "Run a shell command in the local fixture workspace.", {"command": "string", "cwd": "string"}, self.run_command),
            ToolDefinition("read_file", "Read a file from the fixture workspace.", {"path": "string"}, self.read_file),
            ToolDefinition("read_file_range", "Read a 1-based inclusive line range from a file.", {"path": "string", "start_line": "integer", "end_line": "integer"}, self.read_file_range),
            ToolDefinition("write_file", "Write a file in the fixture workspace.", {"path": "string", "content": "string"}, self.write_file),
            ToolDefinition("edit_file", "Replace exact text in an existing file.", {"path": "string", "old_text": "string", "new_text": "string", "expected_occurrences": "integer"}, self.edit_file),
            ToolDefinition("replace_in_file", "Replace exact text using find/replace arguments.", {"path": "string", "find": "string", "replace": "string", "expected_occurrences": "integer"}, self.replace_in_file),
            ToolDefinition("apply_patch", "Apply a unified diff patch in the fixture workspace.", {"patch": "string"}, self.apply_patch),
            ToolDefinition("list_files", "List files under a workspace path.", {"path": "string"}, self.list_files),
            ToolDefinition("search", "Search fixture files for text.", {"query": "string", "path": "string"}, self.search),
            ToolDefinition("git_status", "Return git porcelain-like status from fixture snapshots.", {}, self.git_status),
            ToolDefinition("git_diff", "Return a unified diff from fixture snapshots.", {}, self.git_diff),
            ToolDefinition("run_tests", "Run the detected unittest command.", {}, self.run_tests),
            ToolDefinition("run_build", "Report that no build command is detected.", {}, self.run_build),
            ToolDefinition("run_lint", "Report that no lint command is detected.", {}, self.run_lint),
        ]

    def set_detected_command(self, name: str, command: str | None) -> None:
        if name == "test" and command:
            self.test_command = command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        for definition in self.definitions():
            if definition.name != tool_name:
                continue
            if tool_name in {"git_status", "git_diff", "run_tests", "run_build", "run_lint"}:
                return await definition.handler()
            allowed = {key: value for key, value in args.items() if key in definition.parameters}
            return await definition.handler(**allowed)
        return ToolResult(tool=tool_name, output=f"unknown tool: {tool_name}", exit_code=127)

    async def list_files(self, path: str = ".") -> ToolResult:
        base = safe_workspace_path(self.workspace, normalize_path(path) or ".")
        if not base.exists():
            return ToolResult(tool="list_files", output=f"{path} not found", exit_code=1, metadata={"path": normalize_path(path)})
        if base.is_file():
            paths = [base.relative_to(self.workspace).as_posix()]
        else:
            paths = sorted(file.relative_to(self.workspace).as_posix() for file in base.rglob("*") if file.is_file() and "__pycache__" not in file.parts)
        return ToolResult(tool="list_files", output="\n".join(paths) + ("\n" if paths else ""))

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists() or not target.is_file():
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=target.read_text(encoding="utf-8"), metadata={"path": normalized})

    async def read_file_range(self, path: str, start_line: int = 1, end_line: int = 120) -> ToolResult:
        result = await self.read_file(path)
        if result.exit_code != 0:
            return ToolResult(tool="read_file_range", output=result.output, exit_code=result.exit_code, metadata=result.metadata)
        lines = result.output.splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), max(start, int(end_line)))
        output = "\n".join(f"{idx}: {line}" for idx, line in enumerate(lines[start - 1 : end], start=start))
        return ToolResult(tool="read_file_range", output=output, metadata=result.metadata)

    async def write_file(self, path: str, content: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(tool="write_file", output=f"wrote {normalized}", metadata={"path": normalized})

    async def edit_file(self, path: str, old_text: str, new_text: str, expected_occurrences: int | None = None) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists():
            return ToolResult(tool="edit_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        current = target.read_text(encoding="utf-8")
        occurrences = current.count(old_text)
        if occurrences == 0:
            return ToolResult(tool="edit_file", output="old_text not found", exit_code=1, metadata={"path": normalized})
        if expected_occurrences is not None and int(expected_occurrences) > 0 and occurrences != int(expected_occurrences):
            return ToolResult(tool="edit_file", output=f"expected {expected_occurrences} occurrences, found {occurrences}", exit_code=1, metadata={"path": normalized})
        target.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
        return ToolResult(tool="edit_file", output=f"edited {normalized}", metadata={"path": normalized})

    async def replace_in_file(self, path: str, find: str, replace: str, expected_occurrences: int | None = None) -> ToolResult:
        result = await self.edit_file(path, find, replace, expected_occurrences)
        return ToolResult(tool="replace_in_file", output=result.output, exit_code=result.exit_code, metadata=result.metadata, truncated=result.truncated)

    async def apply_patch(self, patch: str) -> ToolResult:
        completed = subprocess.run(
            "git apply --whitespace=nowarn -",
            input=patch,
            cwd=self.workspace,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        return ToolResult(tool="apply_patch", output=completed.stdout + completed.stderr, exit_code=completed.returncode)

    async def search(self, query: str, path: str = ".") -> ToolResult:
        root = safe_workspace_path(self.workspace, normalize_path(path) or ".")
        matches: list[str] = []
        files = [root] if root.is_file() else [file for file in root.rglob("*") if file.is_file()]
        for file in files:
            if "__pycache__" in file.parts or file.suffix == ".pyc":
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(f"{file.relative_to(self.workspace).as_posix()}:{idx}:{line}")
        return ToolResult(tool="search", output="\n".join(matches[:200]))

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        workspace_cwd = self.workspace if cwd in {"", ".", "/workspace"} else safe_workspace_path(self.workspace, normalize_path(cwd))
        executable_command = python_portable_command(command)
        completed = subprocess.run(
            executable_command,
            cwd=workspace_cwd,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
            timeout=45,
        )
        result = ToolResult(
            tool="run_command",
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            metadata={"command": command, "executed_command": executable_command},
        )
        self.command_results.append(result)
        return result

    async def git_status(self) -> ToolResult:
        changed = self.changed_files()
        return ToolResult(tool="git_status", output="".join(f" M {path}\n" for path in changed))

    async def git_diff(self) -> ToolResult:
        parts: list[str] = []
        current = self.snapshot_files()
        for path in sorted(set(self.initial_files) | set(current)):
            before = self.initial_files.get(path, "").splitlines(keepends=True)
            after = current.get(path, "").splitlines(keepends=True)
            if before == after:
                continue
            parts.append(f"diff --git a/{path} b/{path}\n")
            parts.extend(difflib.unified_diff(before, after, fromfile=f"a/{path}", tofile=f"b/{path}"))
        return ToolResult(tool="git_diff", output="".join(parts))

    async def run_tests(self) -> ToolResult:
        result = await self.run_command(self.test_command)
        return ToolResult(tool="run_tests", output=result.output, exit_code=result.exit_code, metadata={"detected": True, "command": self.test_command})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return self.test_command

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None

    def changed_files(self) -> list[str]:
        current = self.snapshot_files()
        return [path for path in sorted(set(self.initial_files) | set(current)) if self.initial_files.get(path) != current.get(path)]

    def snapshot_files(self) -> dict[str, str]:
        return {
            file.relative_to(self.workspace).as_posix(): file.read_text(encoding="utf-8")
            for file in self.workspace.rglob("*")
            if file.is_file() and "__pycache__" not in file.parts and not file.name.endswith(".pyc")
        }


def safe_workspace_path(workspace: Path, path: str) -> Path:
    normalized = normalize_path(path).lstrip("/")
    target = (workspace / normalized).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes workspace: {path}")
    return target


def python_portable_command(command: str) -> str:
    stripped = command.strip()
    for executable in ("python3", "python"):
        if stripped == executable or stripped.startswith(executable + " "):
            return f'"{sys.executable}"{stripped[len(executable):]}'
    return command


def run_command_results(steps: list[DocodeStep]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for step in steps:
        content = step.content
        if content.get("type") == "tool_result" and content.get("tool") == "run_command":
            results.append(
                {
                    "exit_code": content.get("exit_code"),
                    "command": (content.get("metadata") or {}).get("command"),
                    "summary": content.get("summary"),
                }
            )
    return results


def command_successes(steps: list[DocodeStep]) -> set[str]:
    successful: set[str] = set()
    for result in run_command_results(steps):
        if result.get("exit_code") == 0 and result.get("command"):
            successful.add(" ".join(str(result["command"]).split()))
    return successful


def required_command_succeeded(steps: list[DocodeStep], command: str) -> bool:
    return any(
        result.get("exit_code") == 0 and result.get("command") and commands_equivalent(str(result["command"]), command)
        for result in run_command_results(steps)
    )


def final_candidate_attempted(steps: list[DocodeStep]) -> bool:
    return any(
        (step.content.get("type") == "llm_decision" and step.content.get("decision_type") == "final_candidate")
        or step.content.get("type") == "auto_final_candidate"
        for step in steps
    )


def classify_diagnostic_failure(*, job: CodingJob, steps: list[DocodeStep], required_commands: tuple[str, ...]) -> str:
    contents = [step.content for step in steps]
    tool_calls = [content for content in contents if content.get("type") == "tool_call"]
    tool_results = [content for content in contents if content.get("type") == "tool_result"]
    rejected = [content for content in contents if content.get("type") == "decision_rejected"]
    repairs = [content for content in contents if content.get("type") == "repair_action"]
    edits = [content for content in tool_results if content.get("tool") in {"write_file", "edit_file", "replace_in_file", "apply_patch"} and content.get("exit_code") == 0]
    reads = [content for content in tool_results if content.get("tool") in {"read_file", "read_file_range", "list_files", "search"} and content.get("exit_code") == 0]
    runs = run_command_results(steps)
    missing = [command for command in required_commands if not required_command_succeeded(steps, command)]
    duplicate_reads = repeated_successful_read_paths(tool_results)

    if job.failure_reason and "artifact" in job.failure_reason:
        return "artifact_export_failure"
    if repair_edit_blocked_by_test_gate(contents):
        return "repair_edit_blocked_by_test_gate"
    if diagnostic_inspection_blocked_by_must_edit(contents):
        return "inspection_blocked_by_must_edit"
    if not edits and (duplicate_reads or duplicate_inspection_rejected(contents)):
        return "duplicate_inspection_loop"
    if not tool_calls:
        return "planning_failure"
    if edits and any(result.get("exit_code") not in {0, None} for result in runs) and not repairs:
        return "test_failure_not_repaired"
    if not reads and edits:
        return "insufficient_file_inspection"
    if not edits:
        return "wrong_edit_target" if reads else "planning_failure"
    if missing and final_candidate_attempted(steps):
        return "premature_final_blocked"
    if missing:
        return "missing_required_command"
    if repairs and rejected and job.failure_reason == "max_iterations_exceeded":
        return "repair_guidance_loop"
    if job.failure_reason and "verifier" in job.failure_reason:
        return "final_gate_too_strict"
    if job.status != JobStatus.SUCCEEDED:
        return "bad_code_edit"
    return "unknown"


def repeated_successful_read_paths(tool_results: list[dict[str, object]]) -> set[str]:
    counts: dict[str, int] = {}
    for content in tool_results:
        if content.get("tool") != "read_file" or content.get("exit_code") != 0:
            continue
        metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
        path = str(metadata.get("path") or "")
        if not path:
            continue
        counts[path] = counts.get(path, 0) + 1
    return {path for path, count in counts.items() if count >= 3}


def duplicate_inspection_rejected(contents: list[dict[str, object]]) -> bool:
    return any(
        content.get("type") == "decision_rejected"
        and content.get("reason") == "duplicate_inspection_after_edit_pressure"
        for content in contents
    )


def repair_edit_blocked_by_test_gate(contents: list[dict[str, object]]) -> bool:
    failed_required_command = any(
        content.get("type") == "tool_result"
        and content.get("tool") == "run_command"
        and content.get("exit_code") not in {0, None}
        for content in contents
    )
    if not failed_required_command:
        return False
    return any(
        content.get("type") == "decision_rejected"
        and content.get("reason") == "test_required_tool_forbidden"
        and "blocked while TEST_REQUIRED" in str(content.get("detail") or "")
        for content in contents
    )


def diagnostic_inspection_blocked_by_must_edit(contents: list[dict[str, object]]) -> bool:
    rejected = [
        content
        for content in contents
        if content.get("type") == "decision_rejected" and "must_edit_tool_forbidden" in str(content.get("reason") or "").lower()
    ]
    if len(rejected) < 2:
        return False
    if any(content.get("type") == "tool_result" and content.get("tool") == "run_command" for content in contents):
        return False
    for content in contents:
        if content.get("type") == "workflow_state" and content.get("diff_exists") is True:
            return False
        if str(content.get("git_status") or "").strip() or str(content.get("git_diff") or "").strip():
            return False
    return True


def summarize_recent_steps(steps: list[DocodeStep], limit: int = 20) -> str:
    lines: list[str] = []
    for step in steps[-limit:]:
        content = step.content
        step_type = content.get("type", step.kind)
        detail = content.get("reason") or content.get("tool") or content.get("decision_type") or content.get("stage") or ""
        if step_type == "tool_result":
            detail = f"{detail} exit={content.get('exit_code')} summary={content.get('summary')}"
        lines.append(f"{step.step_index}: {step.kind}: {step_type}: {detail}")
    return "\n".join(lines)


async def diagnostic_failure_message(case: DiagnosticCase, job: CodingJob, tools: DiagnosticLocalTools, steps: list[DocodeStep]) -> str:
    status = await tools.git_status()
    diff = await tools.git_diff()
    runs = run_command_results(steps)
    required = {command: required_command_succeeded(steps, command) for command in case.required_commands}
    repairs = [step for step in steps if step.content.get("type") == "repair_action"]
    rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
    category = classify_diagnostic_failure(job=job, steps=steps, required_commands=case.required_commands)
    changed_files = tools.changed_files()
    return (
        f"case name: {case.name}\n"
        f"job status: {job.status.value}\n"
        f"failure reason: {job.failure_reason or '<none>'}\n"
        f"final git status equivalent / changed files:\n{status.output or '<clean>'}\nchanged_files={json.dumps(changed_files)}\n"
        f"git diff equivalent:\n{diff.output or '<empty>'}\n"
        f"all run_command results:\n{json.dumps(runs, indent=2)}\n"
        f"last 20 steps summarized:\n{summarize_recent_steps(steps)}\n"
        f"whether final_candidate was attempted: {str(final_candidate_attempted(steps)).lower()}\n"
        f"whether required commands succeeded: {json.dumps(required, indent=2)}\n"
        f"whether repair_action steps occurred: {str(bool(repairs)).lower()} count={len(repairs)}\n"
        f"whether rejected_decision steps occurred: {str(bool(rejected)).lower()} count={len(rejected)}\n"
        f"likely failure category: {category}\n"
    )


class DiagnosticClassifierTests(TestCase):
    def test_classifier_detects_missing_required_command_after_edit(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="", status=JobStatus.FAILED, failure_reason="max_iterations_exceeded")
        steps = [
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=0, kind="tool", content={"type": "tool_call", "tool": "read_file"}),
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=1, kind="tool", content={"type": "tool_call", "tool": "write_file"}),
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=2, kind="tool", content={"type": "tool_call", "tool": "run_command"}),
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=0, kind="tool", content={"type": "tool_result", "tool": "read_file", "exit_code": 0}),
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=1, kind="tool", content={"type": "tool_result", "tool": "write_file", "exit_code": 0}),
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=2,
                kind="tool",
                content={"type": "tool_result", "tool": "run_command", "exit_code": 0, "metadata": {"command": "python -m unittest discover -s tests"}},
            ),
        ]

        category = classify_diagnostic_failure(
            job=job,
            steps=steps,
            required_commands=("python -m unittest discover -s tests", "python cli.py --name Ada --output out.json"),
        )

        self.assertEqual(category, "missing_required_command")

    def test_classifier_detects_must_edit_inspection_block(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="", status=JobStatus.FAILED, failure_reason="max_consecutive_failures_exceeded")
        steps = [
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=0, kind="system", content={"type": "workflow_state", "diff_exists": False}),
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=1,
                kind="system",
                content={"type": "decision_rejected", "reason": "must_edit_tool_forbidden", "detail": "read_file blocked"},
            ),
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=2,
                kind="system",
                content={"type": "decision_rejected", "reason": "must_edit_tool_forbidden", "detail": "read_file blocked"},
            ),
            DocodeStep(id=new_id("step"), job_id=job.id, step_index=3, kind="tool", content={"type": "tool_result", "tool": "read_file", "exit_code": 0}),
        ]

        category = classify_diagnostic_failure(job=job, steps=steps, required_commands=("python -m unittest discover -s tests",))

        self.assertEqual(category, "inspection_blocked_by_must_edit")

    def test_classifier_detects_duplicate_inspection_loop(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="", status=JobStatus.FAILED, failure_reason="max_iterations_exceeded")
        steps = [
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=index,
                kind="tool",
                content={"type": "tool_result", "tool": "read_file", "exit_code": 0, "metadata": {"path": "app.py"}},
            )
            for index in range(3)
        ]

        category = classify_diagnostic_failure(job=job, steps=steps, required_commands=("python -m unittest discover -s tests",))

        self.assertEqual(category, "duplicate_inspection_loop")

    def test_classifier_detects_duplicate_inspection_rejection_loop(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="", status=JobStatus.FAILED, failure_reason="max_consecutive_failures_exceeded")
        steps = [
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=0,
                kind="system",
                content={
                    "type": "decision_rejected",
                    "reason": "duplicate_inspection_after_edit_pressure",
                    "detail": "You already read app.py",
                },
            )
        ]

        category = classify_diagnostic_failure(job=job, steps=steps, required_commands=("python -m unittest discover -s tests",))

        self.assertEqual(category, "duplicate_inspection_loop")

    def test_classifier_detects_test_gate_blocking_repair(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="", status=JobStatus.FAILED, failure_reason="max_iterations_exceeded")
        steps = [
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=0,
                kind="tool",
                content={"type": "tool_result", "tool": "run_command", "exit_code": 1, "metadata": {"command": "python -m unittest discover -s tests"}},
            ),
            DocodeStep(
                id=new_id("step"),
                job_id=job.id,
                step_index=1,
                kind="system",
                content={
                    "type": "decision_rejected",
                    "reason": "test_required_tool_forbidden",
                    "detail": "edit_file is blocked while TEST_REQUIRED. Run this exact command first: python -m unittest discover -s tests",
                },
            ),
        ]

        category = classify_diagnostic_failure(job=job, steps=steps, required_commands=("python -m unittest discover -s tests",))

        self.assertEqual(category, "repair_edit_blocked_by_test_gate")


@skipUnless(REAL_LLM_SMOKE_ENABLED, "set DOCODE_REAL_LLM_SMOKE=1 to run optional real LLM diagnostic suite")
class RealLLMDiagnosticSuite(IsolatedAsyncioTestCase):
    summaries: list[dict[str, object]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls.summaries:
            return
        print("\ncase | status | iterations | commands run | final attempted | repair actions | likely failure category | short reason")
        for item in cls.summaries:
            print(
                f"{item['case']} | {item['status']} | {item['iterations']} | {item['commands']} | "
                f"{item['final']} | {item['repairs']} | {item['category']} | {item['reason']}"
            )

    async def run_diagnostic_case(self, case: DiagnosticCase) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / case.name
            shutil.copytree(FIXTURE_ROOT / case.fixture, workspace)
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="diagnostic",
                    instruction=case.instruction,
                    max_iterations=36,
                    max_runtime_seconds=900,
                    max_consecutive_failures=10,
                    max_tool_calls=80,
                )
            )
            tools = DiagnosticLocalTools(workspace, test_command=case.required_commands[0])
            llm = await build_real_llm_or_skip(self, job)
            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp) / "artifacts", repo),
                stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=900, max_consecutive_failures=10, max_tool_calls=80),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)
            steps = await repo.list_steps(job.id)
            category = classify_diagnostic_failure(job=result, steps=steps, required_commands=case.required_commands)
            self.summaries.append(
                {
                    "case": case.name,
                    "status": result.status.value,
                    "iterations": len([step for step in steps if step.content.get("type") == "llm_decision"]),
                    "commands": len(run_command_results(steps)),
                    "final": final_candidate_attempted(steps),
                    "repairs": len([step for step in steps if step.content.get("type") == "repair_action"]),
                    "category": category,
                    "reason": result.failure_reason or result.result_summary or "",
                }
            )
            if result.status != JobStatus.SUCCEEDED:
                self.fail(await diagnostic_failure_message(case, result, tools, steps))

    async def test_credential_path_constructs_real_llm(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="diagnostic",
            instruction="Credential construction smoke. Do not call the model.",
        )

        llm = await build_real_llm_or_skip(self, job)

        self.assertIsNotNone(llm)
        self.assertTrue(job.provider)
        self.assertTrue(job.model)

    async def test_cli_output_bug(self) -> None:
        await self.run_diagnostic_case(DIAGNOSTIC_CASES["cli_output_bug"])

    async def test_multifile_api_mismatch(self) -> None:
        await self.run_diagnostic_case(DIAGNOSTIC_CASES["multifile_api_mismatch"])

    async def test_parser_edge_case(self) -> None:
        await self.run_diagnostic_case(DIAGNOSTIC_CASES["parser_edge_case"])

    async def test_two_stage_repair(self) -> None:
        await self.run_diagnostic_case(DIAGNOSTIC_CASES["two_stage_repair"])
