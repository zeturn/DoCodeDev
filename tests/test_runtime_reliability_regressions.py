from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGateResult
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.agent.verifier import VerificationResult
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


FIXTURE = Path(__file__).parent / "fixtures" / "repos" / "verification_authority"
EXPLICIT_COMMANDS = (
    "python checks/check_contract.py",
    "python checks/check_semantics.py",
)


class LocalAuthorityTools:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[str] = []

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M source.py\n")

    async def git_diff(self) -> ToolResult:
        content = (self.root / "source.py").read_text(encoding="utf-8")
        return ToolResult(tool="git_diff", output="diff --git a/source.py b/source.py\n" + "".join(f"+{line}\n" for line in content.splitlines()))

    async def detect_test_command(self):
        return None

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = cwd
        if "git rev-parse --is-inside-work-tree" in command:
            return ToolResult(tool="run_command", output="", metadata={"command": command})
        self.commands.append(command)
        args = shlex.split(command)
        if args and args[0] in {"python", "python3"}:
            args[0] = sys.executable
        completed = subprocess.run(args, cwd=self.root, text=True, capture_output=True, check=False)
        return ToolResult(
            tool="run_command",
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            metadata={"command": command},
        )


def authority_instruction() -> str:
    return (
        "Update source.py so the neutral value contract is satisfied.\n"
        "Target file:\n- source.py\n"
        "Verification commands:\n"
        f"1. {EXPLICIT_COMMANDS[0]}\n"
        f"2. {EXPLICIT_COMMANDS[1]}"
    )


class VerificationAuthorityRegressionTests(IsolatedAsyncioTestCase):
    async def test_no_tests_directory_passes_two_exact_explicit_commands(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            shutil.copytree(FIXTURE, root)
            (root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
            tools = LocalAuthorityTools(root)

            result = await CodingVerifier().verify(
                CodingJob(id=new_id("job"), user_id="u1", instruction=authority_instruction()),
                tools,
            )
            has_tests_directory = (root / "tests").exists()

        self.assertTrue(result.passed)
        self.assertFalse(has_tests_directory)
        self.assertEqual([item.exit_code for item in result.explicit_results or []], [0, 0])
        self.assertEqual([item.metadata["command"] for item in result.explicit_results or []], list(EXPLICIT_COMMANDS))
        self.assertTrue(result.test_result.metadata["skipped"])
        self.assertFalse(any("unittest discover" in command for command in tools.commands))

    async def test_nonzero_explicit_semantic_command_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            shutil.copytree(FIXTURE, root)
            tools = LocalAuthorityTools(root)

            result = await CodingVerifier().verify(
                CodingJob(id=new_id("job"), user_id="u1", instruction=authority_instruction()),
                tools,
            )

        self.assertFalse(result.passed)
        self.assertEqual([item.exit_code for item in result.explicit_results or []], [0, 1])
        self.assertIn("fix failing explicit verification command", result.required_fixes)


class RangeRepairLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(
                type="tool_call",
                tool_name="edit_file",
                args={"path": "source_module.py", "old_text": "NOTE = 0", "new_text": "NOTE = 1"},
            )
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python checks/verify_source.py"})
        if self.calls == 3:
            return AgentDecision(
                type="tool_call",
                tool_name="read_file_range",
                args={"path": "source_module.py", "start_line": 205, "end_line": 225},
            )
        if self.calls == 4:
            return AgentDecision(
                type="tool_call",
                tool_name="edit_file",
                args={"path": "source_module.py", "old_text": 'RETURN = "bad"', "new_text": "RETURN = 2"},
            )
        if self.calls == 5:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python checks/verify_source.py"})
        return AgentDecision(type="final_candidate", summary="Updated the source value and verified the explicit check.")


class RangeRepairTools:
    def __init__(self) -> None:
        lines = [f"LINE_{index} = {index}" for index in range(1, 219)]
        lines.extend(["NOTE = 0", 'RETURN = "bad"'])
        self.files = {"source_module.py": "\n".join(lines) + "\n"}
        self.edited = False
        self.commands: list[str] = []

    def definitions(self):
        return [
            SimpleNamespace(name=name)
            for name in (
                "read_file",
                "read_file_range",
                "read_symbol",
                "edit_file",
                "write_file",
                "replace_in_file",
                "apply_patch",
                "run_command",
                "git_status",
                "git_diff",
            )
        ]

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        if tool_name == "edit_file":
            path = str(args["path"])
            old = str(args["old_text"])
            new = str(args["new_text"])
            if old not in self.files[path]:
                return ToolResult(tool=tool_name, output="old text not found", exit_code=1, metadata={"path": path})
            self.files[path] = self.files[path].replace(old, new, 1)
            self.edited = True
            return ToolResult(tool=tool_name, output="edited 1 occurrence", metadata={"path": path})
        if tool_name == "read_file_range":
            path = str(args["path"])
            start = int(args.get("start_line", 1))
            end = int(args.get("end_line", 120))
            output = "\n".join(self.files[path].splitlines()[start - 1 : end]) + "\n"
            return ToolResult(
                tool=tool_name,
                output=output,
                metadata={"path": path, "start_line": start, "end_line": end},
            )
        if tool_name == "run_command":
            command = str(args["command"])
            self.commands.append(command)
            if "RETURN = 2" not in self.files["source_module.py"]:
                return ToolResult(
                    tool=tool_name,
                    output=(
                        "Traceback (most recent call last):\n"
                        '  File "/workspace/source_module.py", line 220, in load_value\n'
                        "ValueError: invalid literal for int() with base 10: 'bad'\n"
                    ),
                    exit_code=1,
                    metadata={"command": command},
                )
            return ToolResult(tool=tool_name, output="source check ok\n", metadata={"command": command})
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        return ToolResult(tool="list_files", output="source_module.py\nchecks/verify_source.py\n")

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M source_module.py\n" if self.edited else "")

    async def git_diff(self) -> ToolResult:
        if not self.edited:
            return ToolResult(tool="git_diff", output="")
        return ToolResult(tool="git_diff", output="diff --git a/source_module.py b/source_module.py\n+NOTE = 1\n+RETURN = 2\n")

    async def detect_test_command(self):
        return None

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None


class PassingQualityGate:
    async def run(self, *, tools, task_contract, instruction):
        _ = tools, task_contract, instruction
        return QualityGateResult(passed=True)


class PassingLoopVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, tools, evidence
        return VerificationResult(
            passed=True,
            confidence=0.9,
            reason="ok",
            required_fixes=[],
            git_diff="diff --git a/source_module.py b/source_module.py\n+RETURN = 2\n",
        )


class RepairToolLoopRegressionTests(IsolatedAsyncioTestCase):
    async def test_edit_forced_range_read_edit_rerun_clears_without_schema_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            repository = InMemoryJobRepository()
            job = await repository.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction=(
                        "Fix the value conversion near the end of source_module.py.\n"
                        "Target file:\n- source_module.py\n"
                        "Verification commands:\n1. python checks/verify_source.py"
                    ),
                )
            )
            tools = RangeRepairTools()
            loop = CodingAgentLoop(
                llm=RangeRepairLLM(),
                tools=tools,
                verifier=PassingLoopVerifier(),
                quality_gate=PassingQualityGate(),
                repository=repository,
                exporter=ArtifactExporter(Path(tmp), repository),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60),
            )

            result = await loop.run(job)
            steps = await repository.list_steps(job.id)

        contents = [step.content for step in steps]
        range_calls = [
            step for step in steps if step.content.get("type") == "tool_call" and step.content.get("tool") == "read_file_range"
        ]
        command_calls = [
            step for step in steps if step.content.get("type") == "tool_call" and step.content.get("tool") == "run_command"
        ]
        source_edits = [
            step
            for step in steps
            if step.content.get("type") == "tool_call"
            and step.content.get("tool") == "edit_file"
            and step.content.get("args", {}).get("path") == "source_module.py"
        ]

        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertEqual(len(range_calls), 1)
        self.assertEqual(len(command_calls), 2)
        self.assertLess(range_calls[0].step_index, source_edits[-1].step_index)
        self.assertLess(source_edits[-1].step_index, command_calls[-1].step_index)
        self.assertFalse(any(content.get("reason") == "tool_not_in_current_schema" for content in contents))
        self.assertFalse(any(content.get("type") == "unavailable_tool_requested" for content in contents))
        self.assertEqual(tools.commands, ["python checks/verify_source.py", "python checks/verify_source.py"])
