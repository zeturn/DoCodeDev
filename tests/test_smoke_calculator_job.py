from __future__ import annotations

import difflib
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_contract import TaskContract
from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


REQUIRED_COMMAND = "python -m unittest discover -s tests"

FORBIDDEN_SMOKE_STRINGS = (
    "GitHub Trends",
    "GitHub Trending",
    "owner/repo",
    "Box-row",
    "stars today",
    "crawler.py",
)


class RecordingRepository(InMemoryJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.status_updates: list[JobStatus] = []

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        updated = await super().update_job(job_id, **changes)
        if "status" in changes:
            self.status_updates.append(updated.status)
        return updated


class CalculatorSmokeLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.saw_required_command_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "calculator.py"})
        if self.calls == 2:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n"},
            )
        if self.calls == 3:
            return AgentDecision(type="final_candidate", summary="Fixed calculator add before verification.")
        if self.calls == 4:
            self.saw_required_command_feedback = any(
                message.get("kind") == "feedback" and REQUIRED_COMMAND in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": REQUIRED_COMMAND})
        return AgentDecision(type="final_candidate", summary="Fixed calculator add and verified tests.")


class FixtureCalculatorTools:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.initial_files = self.snapshot_files()
        self.commands: list[str] = []

    def definitions(self):
        return []

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        if tool_name == "read_file":
            return await self.read_file(str(args["path"]))
        if tool_name == "write_file":
            return await self.write_file(str(args["path"]), str(args["content"]))
        if tool_name == "edit_file":
            return await self.edit_file(str(args["path"]), str(args["old_text"]), str(args["new_text"]))
        if tool_name == "run_command":
            return await self.run_command(str(args["command"]))
        if tool_name == "git_status":
            return await self.git_status()
        if tool_name == "git_diff":
            return await self.git_diff()
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        paths = sorted(
            file.relative_to(self.workspace).as_posix()
            for file in self.workspace.rglob("*")
            if file.is_file()
        )
        return ToolResult(tool="list_files", output="\n".join(paths) + "\n")

    async def read_file(self, path: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        if not target.exists():
            return ToolResult(tool="read_file", output=f"{normalized} not found", exit_code=1, metadata={"path": normalized})
        return ToolResult(tool="read_file", output=target.read_text(encoding="utf-8"), metadata={"path": normalized})

    async def write_file(self, path: str, content: str) -> ToolResult:
        normalized = normalize_path(path)
        target = safe_workspace_path(self.workspace, normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(tool="write_file", output=f"wrote {normalized}", metadata={"path": normalized})

    async def edit_file(self, path: str, old_text: str, new_text: str) -> ToolResult:
        current = (await self.read_file(path)).output
        if old_text not in current:
            return ToolResult(tool="edit_file", output="old_text not found", exit_code=1, metadata={"path": normalize_path(path)})
        return await self.write_file(path, current.replace(old_text, new_text, 1))

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = cwd
        self.commands.append(command)
        if command.startswith("git add -N"):
            return ToolResult(tool="run_command", output="", metadata={"command": command})
        completed = subprocess.run(
            command,
            cwd=self.workspace,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        return ToolResult(tool="run_command", output=output, exit_code=completed.returncode, metadata={"command": command})

    async def git_status(self) -> ToolResult:
        changed = self.changed_files()
        output = "".join(f" M {path}\n" for path in changed)
        return ToolResult(tool="git_status", output=output)

    async def git_diff(self) -> ToolResult:
        parts: list[str] = []
        current = self.snapshot_files()
        for path in sorted(set(self.initial_files) | set(current)):
            before = self.initial_files.get(path, "").splitlines(keepends=True)
            after = current.get(path, "").splitlines(keepends=True)
            if before == after:
                continue
            parts.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        return ToolResult(tool="git_diff", output="".join(parts))

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="no test command auto-detected", metadata={"detected": False})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self):
        return None

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


class RequiredCommandVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, evidence
        status = await tools.git_status()
        diff = await tools.git_diff()
        command_ok = REQUIRED_COMMAND in tools.commands
        return VerificationResult(
            passed=bool(diff.output.strip()) and command_ok,
            confidence=0.95,
            reason="Calculator smoke command verified.",
            required_fixes=[] if command_ok else [f"run required command: {REQUIRED_COMMAND}"],
            git_status=status.output,
            git_diff=diff.output,
            status_result=status,
            test_result=ToolResult(tool="run_command", output="OK\n", metadata={"command": REQUIRED_COMMAND}) if command_ok else None,
        )


class ExplicitCommandSmokeLoop(CodingAgentLoop):
    async def bootstrap(self, state: AgentState) -> None:
        await super().bootstrap(state)
        assert state.task_contract is not None
        state.task_contract = TaskContract(
            must_modify_files=state.task_contract.must_modify_files,
            must_run_commands=[REQUIRED_COMMAND],
            forbidden_finish_conditions=state.task_contract.forbidden_finish_conditions,
        )
        steps = await self.repository.list_steps(state.job.id)
        bootstrap_step = steps[-1]
        bootstrap_step.content["task_contract"]["must_run_commands"] = [REQUIRED_COMMAND]


class CalculatorSmokeJobTests(IsolatedAsyncioTestCase):
    async def test_calculator_bugfix_runs_required_command_before_success(self) -> None:
        fixture_root = Path(__file__).resolve().parent / "fixtures" / "repos" / "calculator_bug"
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(fixture_root, workspace)
            repo = RecordingRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="smoke",
                    instruction=(
                        "Fix calculator.py so the tests pass.\n\n"
                        "Verification commands:\n"
                        f"1. {REQUIRED_COMMAND}"
                    ),
                )
            )
            tools = FixtureCalculatorTools(workspace)
            llm = CalculatorSmokeLLM()

            loop = ExplicitCommandSmokeLoop(
                llm=llm,
                tools=tools,
                verifier=RequiredCommandVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp) / "artifacts", repo, workspace_file_reader=lambda path: tools.read_file(path)),
                stop_policy=StopPolicy(max_iterations=8, max_runtime_seconds=60),
                quality_gate=QualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn(JobStatus.RUNNING, repo.status_updates)
            self.assertIn(JobStatus.SUCCEEDED, repo.status_updates)
            self.assertTrue(llm.saw_required_command_feedback)
            self.assertEqual(tools.commands.count(REQUIRED_COMMAND), 1)
            self.assertIn("return a + b", (workspace / "calculator.py").read_text(encoding="utf-8"))

            steps = await repo.list_steps(job.id)
            self.assertTrue(any(step.content.get("type") == "tool_result" and step.content.get("tool") == "read_file" for step in steps))
            self.assertTrue(any(step.content.get("type") == "tool_result" and step.content.get("tool") == "write_file" for step in steps))
            command_steps = [
                step
                for step in steps
                if step.content.get("type") == "tool_call"
                and step.content.get("tool") == "run_command"
                and step.content.get("args", {}).get("command") == REQUIRED_COMMAND
            ]
            self.assertEqual(len(command_steps), 1)
            self.assertTrue(any(step.content.get("reason") == "final_candidate_tests_missing" for step in steps if step.content.get("type") == "decision_rejected"))
            self.assertTrue(any(step.kind == "verifier" for step in steps))

            artifacts = await repo.list_artifacts(job.id)
            artifact_kinds = {artifact.kind for artifact in artifacts}
            self.assertIn("report", artifact_kinds)
            self.assertIn("result", artifact_kinds)
            self.assertIsNotNone(result.artifact_id)

            combined_steps = "\n".join(str(step.content) for step in steps)
            combined_files = "\n".join(tools.snapshot_files().values())
            for forbidden in FORBIDDEN_SMOKE_STRINGS:
                self.assertNotIn(forbidden, combined_steps)
                self.assertNotIn(forbidden, combined_files)
            self.assertFalse(any(step.content.get("reason") == "active_repair_controller_forced_target_edit" for step in steps))


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def safe_workspace_path(workspace: Path, path: str) -> Path:
    target = (workspace / path).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(path)
    return target
