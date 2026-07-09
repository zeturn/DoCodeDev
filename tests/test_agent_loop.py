from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.loop import (
    INITIAL_NO_DIFF_EXPLORATION_BUDGET,
    CodingAgentLoop,
    allowed_tools_for_repair_mode_name,
    allowed_tool_definitions_for_state,
    compact_llm_message,
    edit_required_tool_block,
    latest_failed_required_command,
    preferred_targeted_repair_target,
    refine_repair_action_targets,
    required_test_tool_block,
    repair_mode_tool_block,
    task_contract_source_targets,
    targeted_repair_action_block,
    targeted_repair_exploration_block,
    targeted_repair_forced_tool,
    verification_repair_feedback,
)
from docode.agent.quality_gate import QualityGateResult
from docode.agent.repair_planner import RepairAction
from docode.agent.reviewer import ReviewResult
from docode.agent.stop_policy import StopPolicy
from docode.agent.state import AgentState
from docode.agent.task_contract import TaskContract
from docode.agent.verifier import CodingVerifier, VerificationResult
from docode.agent.workflow import WorkflowPhase
from docode.agent.context import instruction_source_urls
from docode.agent.inspector import should_skip_important_file_reads
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision, LLMUsageMeter
from docode.llm.provider_compat import ProviderErrorInfo, ProviderUnavailableError
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[str] = []

    async def decide(self, *, system, messages, tools, context):
        self.contexts.append(context)
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        return AgentDecision(type="final_candidate", summary="Updated README.")


class CancellingLLM:
    def __init__(self, repo: InMemoryJobRepository, job_id: str) -> None:
        self.repo = repo
        self.job_id = job_id

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        await self.repo.update_job(self.job_id, status=JobStatus.STOPPED, failure_reason="cancelled")
        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "should not write\n"})


class BudgetBurningLLM:
    def __init__(self, meter: LLMUsageMeter, *, cost: float = 0.0) -> None:
        self.meter = meter
        self.cost = cost

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.meter.record_text_call(prompt="x" * 100, response='{"type":"tool_call"}', cost=self.cost)
        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "over budget\n"})


class FlakyLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            raise ValueError("invalid json from provider")
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "recovered\n"})
        return AgentDecision(type="final_candidate", summary="Recovered from a model error and updated README.")


class FlakyThenWritesAtBudgetLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls in {1, 2}:
            raise ValueError("unsupported decision type: ")
        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})


class MissingSummaryLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_summary_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        if self.calls == 2:
            return AgentDecision(type="final_candidate", summary="")
        self.observed_summary_feedback = any(
            message.get("kind") == "feedback" and "final_summary_missing" in str(message.get("content"))
            for message in messages
        )
        return AgentDecision(type="final_candidate", summary="Updated README after providing a final summary.")


class ToolRepairLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_tool_error = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "first attempt\n"})
        if self.calls == 2:
            self.observed_tool_error = any(
                message.get("role") == "tool" and message.get("exit_code") == 1 and "sandbox write unavailable" in str(message.get("output"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "repaired\n"})
        return AgentDecision(type="final_candidate", summary="Recovered from a transient tool error and updated README.")


class UnusableDecisionLLM:
    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        return AgentDecision(type="nonsense")


class NoEditToolLoopLLM:
    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo still-clean"})


class UnauthorizedLLM:
    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        raise RuntimeError("Client error '401 Unauthorized' for url 'http://localhost:8103/v1/chat/completions'")


class ProviderUnavailableLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        raise ProviderUnavailableError(
            ProviderErrorInfo(
                category="provider_upstream_unavailable",
                retryable=True,
                status_code=503,
                detail="Server error '503 Service Unavailable' for url 'http://localhost:8103/v1/chat/completions'",
            ),
            attempts=3,
            cause=RuntimeError("503 Service Unavailable"),
        )


class RecoveringProviderUnavailableLLM(ProviderUnavailableLLM):
    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            raise ProviderUnavailableError(
                ProviderErrorInfo(
                    category="provider_upstream_unavailable",
                    retryable=True,
                    status_code=502,
                    detail="Server error '502 Bad Gateway' for url 'http://localhost:8103/v1/chat/completions'",
                ),
                attempts=3,
                cause=RuntimeError("502 Bad Gateway"),
            )
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "recovered\n"})
        return AgentDecision(type="final_candidate", summary="Recovered after provider retry.")


class PrematureFinalLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_clean_status_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="final_candidate", summary="Done without edits.")
        if self.calls == 2:
            self.observed_clean_status_feedback = any(
                message.get("kind") == "feedback" and "Final candidate rejected before verification" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        return AgentDecision(type="final_candidate", summary="Updated README after rejection.")


class RepairModeForbiddenToolLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.tool_names_seen: list[list[str]] = []

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, context
        self.calls += 1
        self.tool_names_seen.append([getattr(tool, "name", "") for tool in tools])
        if self.calls == 1:
            return AgentDecision(type="final_candidate", summary="Done without edits.")
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo nope"})
        if self.calls == 3:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        return AgentDecision(type="final_candidate", summary="Updated README after repair mode.")


class RequiredTestGateLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_test_gate_feedback = False
        self.observed_next_required_command = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        if self.calls == 2:
            return AgentDecision(type="final_candidate", summary="Updated README before running verification.")
        if self.calls == 3:
            self.observed_test_gate_feedback = any(
                message.get("kind") == "feedback" and "final_candidate_tests_missing" in str(message.get("content"))
                for message in messages
            )
            self.observed_next_required_command = any(
                message.get("kind") == "feedback" and "Next required command: echo checked" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo checked"})
        return AgentDecision(type="final_candidate", summary="Updated README after running required verification.")


class FinalReadyToolLoopLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo checked"})
        return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo extra"})


class FinalReadyMalformedLLM(FinalReadyToolLoopLLM):
    async def decide(self, *, system, messages, tools, context):
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "done\n"})
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "echo checked"})
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


class ReviewerRepairLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_review_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "thin\n"})
        if self.calls == 2:
            return AgentDecision(type="final_candidate", summary="Updated README.")
        if self.calls == 3:
            self.observed_review_feedback = any(
                message.get("kind") == "feedback" and "Independent reviewer found blocking quality issues" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "README.md", "content": "substantive\n"})
        return AgentDecision(type="final_candidate", summary="Updated README after independent review.")


class TargetedRepairLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.observed_targeted_feedback = False
        self.observed_rerun_feedback = False
        self.observed_dry_run_feedback = False

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "crawler.py", "content": "def parse_repos():\n    return []\n"})
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 -m unittest discover -s tests"})
        if self.calls == 3:
            self.observed_targeted_feedback = "Active Targeted Repair" in context and any(
                message.get("kind") == "feedback" and "import_error_missing_symbol" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="web_search", args={"query": "parse repositories"})
        if self.calls == 4:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={"path": "crawler.py", "content": "def parse_repositories(html=''):\n    return []\n"},
            )
        if self.calls == 5:
            return AgentDecision(type="final_candidate", summary="Fixed crawler export.")
        if self.calls == 6:
            self.observed_rerun_feedback = any(
                message.get("kind") == "feedback" and "targeted_repair_rerun_missing" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 -m unittest discover -s tests"})
        if self.calls == 7:
            self.observed_dry_run_feedback = any(
                message.get("kind") == "feedback" and "python3 crawler.py --source fixtures/sample_source.html --output data/output.json --dry-run" in str(message.get("content"))
                for message in messages
            )
            return AgentDecision(
                type="tool_call",
                tool_name="run_command",
                args={"command": "python3 crawler.py --source fixtures/sample_source.html --output data/output.json --dry-run"},
            )
        return AgentDecision(type="final_candidate", summary="Fixed crawler export after targeted repair.")


class TargetedRepairTools:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.call_count = 0

    def definitions(self):
        return []

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        self.call_count += 1
        if tool_name == "write_file":
            self.files[str(args["path"])] = str(args["content"])
            return ToolResult(tool="write_file", output="wrote file", metadata={"path": str(args["path"])})
        if tool_name == "run_command":
            command = str(args["command"])
            content = self.files.get("crawler.py", "")
            if "parse_repositories" not in content:
                return ToolResult(
                    tool="run_command",
                    output="ImportError: cannot import name 'parse_repositories' from 'crawler'",
                    exit_code=1,
                    metadata={"command": command},
                )
            return ToolResult(tool="run_command", output="OK\n", metadata={"command": command})
        raise AssertionError(tool_name)

    async def run_command(self, command: str, cwd: str = "/workspace") -> ToolResult:
        _ = cwd
        content = self.files.get("crawler.py", "")
        if "parse_repositories" not in content:
            return ToolResult(
                tool="run_command",
                output="ImportError: cannot import name 'parse_repositories' from 'crawler'",
                exit_code=1,
                metadata={"command": command},
            )
        return ToolResult(tool="run_command", output="OK\n", metadata={"command": command})

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M crawler.py\n" if self.files else "")

    async def git_diff(self) -> ToolResult:
        output = "diff --git a/crawler.py b/crawler.py\n+def parse_repositories(html=''):\n+    return []\n" if self.files else ""
        return ToolResult(tool="git_diff", output=output)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        return ToolResult(tool="list_files", output="README.md\ncrawler.py\ntests/test_parser.py\n")

    async def read_file(self, path: str) -> ToolResult:
        normalized = str(path).replace("/workspace/", "")
        return ToolResult(tool="read_file", output=self.files.get(normalized, ""), metadata={"path": normalized})

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", metadata={"detected": True, "command": "python3 -m unittest discover -s tests"})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return "python3 -m unittest discover -s tests"

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None


class CalculatorRepairLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, system, messages, tools, context):
        _ = system, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "calculator.py", "content": "def add(a, b):\n    return a - b\n"})
        if self.calls == 2:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 -m unittest discover -s tests"})
        if self.calls == 3:
            return AgentDecision(type="tool_call", tool_name="read_file", args={"path": "calculator.py"})
        if self.calls == 4:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 -m unittest discover -s tests"})
        if self.calls == 5:
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n"})
        if self.calls == 6:
            return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 -m unittest discover -s tests"})
        return AgentDecision(type="final_candidate", summary="Fixed calculator add and verified tests.")


class CalculatorRepairTools:
    def __init__(self) -> None:
        self.files: dict[str, str] = {
            "tests/test_calculator.py": "from calculator import add\n\nclass TestCalc:\n    pass\n",
        }
        self.call_count = 0

    def definitions(self):
        return []

    def set_detected_command(self, name: str, command: str | None) -> None:
        _ = name, command

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        self.call_count += 1
        if tool_name == "write_file":
            path = str(args["path"])
            self.files[path] = str(args["content"])
            return ToolResult(tool="write_file", output="wrote file", metadata={"path": path})
        if tool_name == "read_file":
            path = str(args["path"]).replace("/workspace/", "")
            return ToolResult(tool="read_file", output=self.files.get(path, ""), metadata={"path": path})
        if tool_name == "run_command":
            command = str(args["command"])
            if "return a + b" not in self.files.get("calculator.py", ""):
                return ToolResult(
                    tool="run_command",
                    output=(
                        "Traceback (most recent call last):\n"
                        '  File "/workspace/tests/test_calculator.py", line 5, in test_add\n'
                        '  File "/workspace/calculator.py", line 2, in add\n'
                        "AssertionError: -1 != 3\n"
                    ),
                    exit_code=1,
                    metadata={"command": command},
                )
            return ToolResult(tool="run_command", output="OK\n", metadata={"command": command})
        raise AssertionError(tool_name)

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M calculator.py\n" if "calculator.py" in self.files else "")

    async def git_diff(self) -> ToolResult:
        output = "diff --git a/calculator.py b/calculator.py\n+def add(a, b):\n+    return a + b\n" if "calculator.py" in self.files else ""
        return ToolResult(tool="git_diff", output=output)

    async def list_files(self, path: str = ".") -> ToolResult:
        _ = path
        return ToolResult(tool="list_files", output="calculator.py\ntests/test_calculator.py\n")

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", metadata={"detected": True, "command": "python3 -m unittest discover -s tests"})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="no build command detected", metadata={"detected": False})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return "python3 -m unittest discover -s tests"

    async def detect_build_command(self):
        return None

    async def detect_lint_command(self):
        return None


class NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name


class PassingQualityGate:
    async def run(self, *, tools, task_contract, instruction):
        _ = tools, task_contract, instruction
        return QualityGateResult(passed=True)


class PassingVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, tools, evidence
        return VerificationResult(passed=True, confidence=0.9, reason="ok", required_fixes=[], git_diff="diff --git a/crawler.py b/crawler.py\n+ok\n")


class FailingVerifier:
    async def verify(self, job, tools, evidence=None):
        _ = job, tools, evidence
        return VerificationResult(
            passed=False,
            confidence=0.9,
            reason="verifier failed",
            required_fixes=["fix verifier blocker"],
            git_diff="diff --git a/README.md b/README.md\n+done\n",
        )


class BlockingThenPassingReviewer:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tool_results = False

    async def review(self, *, instruction, task_contract, quality, recent_tool_results, final_summary):
        _ = instruction, task_contract, quality, final_summary
        self.calls += 1
        self.seen_tool_results = self.seen_tool_results or bool(recent_tool_results)
        if self.calls == 1:
            return ReviewResult(
                passed=False,
                confidence=0.81,
                blocking_issues=["README update is too thin to satisfy the requested change."],
                repair_plan=["Expand README.md with substantive content, then final again."],
            )
        return ReviewResult(passed=True, confidence=0.91, warnings=["minor wording risk"])


class FakeTools:
    def __init__(self, *, fail_first_write: bool = False) -> None:
        self.files: dict[str, str] = {}
        self.call_count = 0
        self.fail_first_write = fail_first_write

    def definitions(self):
        return []

    async def call(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        self.call_count += 1
        if tool_name == "write_file":
            if self.fail_first_write and self.call_count == 1:
                raise RuntimeError("sandbox write unavailable")
            self.files[str(args["path"])] = str(args["content"])
            return ToolResult(tool="write_file", output="wrote file")
        if tool_name == "run_command":
            command = str(args["command"])
            return ToolResult(tool="run_command", output="checked\n", metadata={"command": command})
        raise AssertionError(tool_name)

    async def list_files(self, path: str = ".") -> ToolResult:
        return ToolResult(tool="list_files", output="README.md\ngo.mod\n")

    async def read_file(self, path: str) -> ToolResult:
        if path == "README.md":
            return ToolResult(tool="read_file", output="# Example\n")
        if path == "go.mod":
            return ToolResult(tool="read_file", output="module example\n")
        return ToolResult(tool="read_file", output="", exit_code=1)

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=" M README.md\n" if self.files else "")

    async def git_diff(self) -> ToolResult:
        output = "diff --git a/README.md b/README.md\n+done\n" if self.files else ""
        return ToolResult(tool="git_diff", output=output)

    async def run_tests(self) -> ToolResult:
        return ToolResult(tool="run_tests", output="ok", metadata={"detected": True, "command": "go test ./..."})

    async def run_build(self) -> ToolResult:
        return ToolResult(tool="run_build", output="ok", metadata={"detected": True, "command": "go build ./..."})

    async def run_lint(self) -> ToolResult:
        return ToolResult(tool="run_lint", output="no lint command detected", metadata={"detected": False})

    async def detect_test_command(self) -> str:
        return "go test ./..."

    async def detect_build_command(self) -> str:
        return "go build ./..."

    async def detect_lint_command(self):
        return None


class AgentLoopTests(IsolatedAsyncioTestCase):
    def test_compact_llm_message_truncates_large_fields(self) -> None:
        message = {
            "role": "tool",
            "tool": "run_command",
            "exit_code": 1,
            "output": "x" * 5000,
            "metadata": {
                "command": "python3 crawler.py --dry-run",
                "path": "/workspace/crawler.py",
                "ignored": "y" * 2000,
            },
        }

        compact = compact_llm_message(message)

        self.assertEqual(compact["role"], "tool")
        self.assertEqual(compact["tool"], "run_command")
        self.assertLess(len(str(compact["output"])), 700)
        self.assertEqual(
            compact["metadata"],
            {
                "command": "python3 crawler.py --dry-run",
                "path": "/workspace/crawler.py",
            },
        )

    def test_verification_feedback_includes_smoke_command_and_output(self) -> None:
        feedback = verification_repair_feedback(
            VerificationResult(
                passed=False,
                confidence=0.2,
                reason="Verification failed",
                required_fixes=["fix failing smoke verification command"],
                smoke_result=ToolResult(
                    tool="run_smoke",
                    output="SyntaxError: f-string: unmatched '('",
                    exit_code=1,
                    metadata={"command": "python3 -m py_compile scraper.py && python scraper.py"},
                ),
            )
        )

        self.assertIn("Required fixes:", feedback)
        self.assertIn("Smoke command:", feedback)
        self.assertIn("python3 -m py_compile scraper.py && python scraper.py", feedback)
        self.assertIn("Smoke output:", feedback)
        self.assertIn("SyntaxError", feedback)

    def test_verification_feedback_includes_must_edit_repair_for_no_diff(self) -> None:
        feedback = verification_repair_feedback(
            VerificationResult(
                passed=False,
                confidence=0.2,
                reason="Verification failed",
                required_fixes=["produce a non-empty git diff or explicit artifact"],
                git_diff="",
            ),
            TaskContract(must_modify_files=["calculator.py"], must_run_commands=["python3 -m unittest discover -s tests"]),
        )

        self.assertIn("Mandatory next step:", feedback)
        self.assertIn("edit_file, write_file, replace_in_file, or apply_patch", feedback)
        self.assertIn("Required file missing from diff:", feedback)
        self.assertIn("calculator.py", feedback)
        self.assertIn("python-bugfix", feedback)

    async def test_agent_loop_tools_verifies_and_exports_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = ScriptedLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(tmp_path, repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )
            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README.")
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"patch", "report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertTrue((tmp_path / job.id / "patch.diff").read_text(encoding="utf-8").startswith("diff --git"))
            self.assertTrue((tmp_path / job.id / "workspace.zip").exists())
            result_payload = json.loads((tmp_path / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "succeeded")
            self.assertEqual(result_payload["changed_files"], ["README.md"])
            self.assertEqual(
                result_payload["artifacts"],
                {
                    "patch": "patch.diff",
                    "report": "final_report.md",
                    "result": "result.json",
                    "zip": "workspace.zip",
                    "log": "test_log.txt",
                },
            )
            checks_by_name = {check["name"]: check for check in result_payload["checks"]}
            self.assertIn("git_status", checks_by_name)
            self.assertIsNone(checks_by_name["test"]["command"])
            self.assertIn("Detected commands: test=go test ./..., build=go build ./..., lint=not detected", llm.contexts[0])

            steps = await repo.list_steps(job.id)
            bootstrap = next(step for step in steps if step.content.get("type") == "bootstrap")
            self.assertEqual(bootstrap.content["detected_commands"]["test"], "go test ./...")
            self.assertIn("`go build ./...` exits successfully.", bootstrap.content["acceptance_criteria"])
            decisions = [step for step in steps if step.content.get("type") == "llm_decision"]
            self.assertEqual(decisions[0].kind, "llm")
            self.assertEqual(decisions[0].content["tool"], "write_file")
            tool_call = next(step for step in steps if step.content.get("type") == "tool_call")
            self.assertEqual(tool_call.content["tool"], "write_file")
            self.assertEqual(tool_call.content["args"]["content"]["bytes"], len("done\n".encode("utf-8")))
            tool_result = next(step for step in steps if step.content.get("type") == "tool_result")
            self.assertEqual(tool_result.content["summary"], "wrote file")

    async def test_agent_loop_observes_cancel_before_tool_call(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=CancellingLLM(repo, job.id),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.STOPPED)
            self.assertEqual(tools.call_count, 0)
            self.assertIsNotNone(result.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "stopped")
            self.assertEqual(result_payload["stopped_reason"], "cancelled")
            steps = await repo.list_steps(job.id)
            self.assertTrue(any(step.content.get("type") == "cancelled_observed" for step in steps))
            self.assertTrue(any(step.content.get("type") == "stopped_artifacts_exported" for step in steps))

    async def test_agent_loop_stops_before_tool_when_llm_budget_is_exhausted(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme", max_llm_tokens=1))
            tools = FakeTools()
            usage = LLMUsageMeter()

            loop = CodingAgentLoop(
                llm=BudgetBurningLLM(usage),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60, max_llm_tokens=1),
                usage_meter=usage,
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_llm_tokens_exceeded")
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_step = next(step for step in steps if step.kind == "llm")
            self.assertGreater(llm_step.content["usage"]["total_tokens"], 1)

    async def test_agent_loop_stops_before_tool_when_llm_cost_budget_is_exhausted(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme", max_llm_cost=0.01))
            tools = FakeTools()
            usage = LLMUsageMeter()

            loop = CodingAgentLoop(
                llm=BudgetBurningLLM(usage, cost=0.02),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60, max_llm_cost=0.01),
                usage_meter=usage,
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_llm_cost_exceeded")
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_step = next(step for step in steps if step.kind == "llm")
            self.assertEqual(llm_step.content["usage"]["cost"], 0.02)

    async def test_agent_loop_recovers_from_transient_llm_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=FlakyLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Recovered from a model error and updated README.")
            self.assertEqual(tools.files["README.md"], "recovered\n")
            steps = await repo.list_steps(job.id)
            llm_error = next(step for step in steps if step.content.get("type") == "llm_error")
            self.assertEqual(llm_error.content["reason"], "llm_decision_failed")
            self.assertIn("invalid json", llm_error.content["detail"])

    async def test_agent_loop_auto_finalizes_when_final_ready_at_iteration_budget(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=FlakyThenWritesAtBudgetLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=3, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Completed the requested changes in README.md.")
            steps = await repo.list_steps(job.id)
            auto = [step for step in steps if step.content.get("type") == "auto_final_candidate"]
            self.assertEqual(auto[0].content["reason"], "final_ready_stop_policy_auto_finalized")

    async def test_agent_loop_requires_final_candidate_summary_before_success(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = MissingSummaryLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=PassingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README after providing a final summary.")
            self.assertTrue(llm.observed_summary_feedback)
            steps = await repo.list_steps(job.id)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(llm_errors[0].content["reason"], "final_summary_missing")
            verifier_steps = [step for step in steps if step.kind == "verifier"]
            self.assertEqual(len(verifier_steps), 1)
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["summary"], "Updated README after providing a final summary.")

    async def test_agent_loop_rejects_final_candidate_when_git_status_is_clean(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update README.md"))
            tools = FakeTools()
            llm = PrematureFinalLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=6, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README after rejection.")
            self.assertTrue(llm.observed_clean_status_feedback)
            steps = await repo.list_steps(job.id)
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertEqual(rejected[0].content["reason"], "final_candidate_clean_git_status")
            verifier_steps = [step for step in steps if step.kind == "verifier"]
            self.assertEqual(len(verifier_steps), 1)

    async def test_agent_loop_repair_mode_rejects_forbidden_tool_until_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update README.md"))
            tools = FakeTools()
            llm = RepairModeForbiddenToolLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=8, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README after repair mode.")
            self.assertEqual(tools.call_count, 1)
            steps = await repo.list_steps(job.id)
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertEqual([step.content["reason"] for step in rejected], ["final_candidate_clean_git_status", "must_edit_tool_forbidden"])

    async def test_agent_loop_rejects_final_candidate_until_required_command_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Update README.md.\nverify with: echo checked",
                )
            )
            tools = FakeTools()
            llm = RequiredTestGateLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=8, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Updated README after running required verification.")
            self.assertTrue(llm.observed_test_gate_feedback)
            self.assertTrue(llm.observed_next_required_command)
            steps = await repo.list_steps(job.id)
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertEqual(rejected[0].content["reason"], "final_candidate_tests_missing")
            self.assertEqual(rejected[0].content["workflow_state"]["phase"], "TEST_REQUIRED")
            self.assertEqual(rejected[0].content["workflow_state"]["missing_commands"], ["echo checked"])
            workflow_steps = [step for step in steps if step.content.get("type") == "workflow_state"]
            self.assertTrue(any(step.content.get("phase") == "TEST_REQUIRED" for step in workflow_steps))

    async def test_agent_loop_does_not_succeed_when_verifier_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update README.md"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=ScriptedLLM(),
                tools=tools,
                verifier=FailingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=3, max_runtime_seconds=60),
                quality_gate=PassingQualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertNotEqual(result.status, JobStatus.SUCCEEDED)
            steps = await repo.list_steps(job.id)
            verifier_steps = [step for step in steps if step.kind == "verifier"]
            self.assertTrue(verifier_steps)
            self.assertFalse(verifier_steps[-1].content["passed"])

    def test_test_required_allows_missing_target_file_edit(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py"],
            must_run_commands=["python3 -m unittest discover -s tests"],
        )
        state.messages.append(
            {
                "role": "tool",
                "tool": "write_file",
                "exit_code": 0,
                "metadata": {"path": "tests/test_parser.py"},
            }
        )
        workflow = SimpleNamespace(phase=WorkflowPhase.TEST_REQUIRED, missing_commands=["python3 -m unittest discover -s tests"])

        self.assertEqual(required_test_tool_block(state, workflow, "write_file", {"path": "crawler.py"}), "")
        self.assertEqual(required_test_tool_block(state, workflow, "write_file", {"path": "/workspace/crawler.py"}), "")
        self.assertIn(
            "required target files are still missing",
            required_test_tool_block(state, workflow, "list_files", {"path": "tests"}),
        )
        missing_test_state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        missing_test_state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py"],
            must_run_commands=["python3 -m unittest discover -s tests"],
        )
        missing_test_state.messages.append(
            {
                "role": "tool",
                "tool": "write_file",
                "exit_code": 0,
                "metadata": {"path": "crawler.py"},
            }
        )
        self.assertIn(
            "tests/test_parser.py",
            required_test_tool_block(missing_test_state, workflow, "run_command", {"command": "cat crawler.py"}),
        )
        self.assertEqual(required_test_tool_block(missing_test_state, workflow, "write_file", {"path": "/workspace/tests/test_parser.py"}), "")

    def test_test_required_does_not_block_on_generated_output_artifacts(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Build crawler."))
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py", "fixtures/sample.html", "data/output.json"],
            must_run_commands=["python3 -m unittest discover -s tests"],
        )
        for path in ("crawler.py", "tests/test_parser.py", "fixtures/sample.html"):
            state.messages.append(
                {
                    "role": "tool",
                    "tool": "write_file",
                    "exit_code": 0,
                    "metadata": {"path": path},
                }
            )
        workflow = SimpleNamespace(phase=WorkflowPhase.TEST_REQUIRED, missing_commands=["python3 -m unittest discover -s tests"])

        self.assertEqual(
            required_test_tool_block(state, workflow, "run_command", {"command": "python3 -m unittest discover -s tests"}),
            "",
        )

    async def test_agent_loop_sets_must_edit_when_tests_blocked_by_missing_target_files(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Build crawler.\nmodify files: crawler.py, tests/test_parser.py\nverify with: python3 crawler.py --dry-run",
                )
            )
            tools = FakeTools()

            class MissingTargetUnderTestLLM:
                def __init__(self) -> None:
                    self.calls = 0
                    self.saw_must_edit_feedback = False

                async def decide(self, *, system, messages, tools, context):
                    _ = system, tools, context
                    self.calls += 1
                    if self.calls == 1:
                        return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "crawler.py", "content": "print('ok')\n"})
                    if self.calls == 2:
                        return AgentDecision(type="tool_call", tool_name="run_command", args={"command": "python3 crawler.py --dry-run"})
                    self.saw_must_edit_feedback = any(
                        message.get("kind") == "feedback" and "required target files are still missing" in str(message.get("content"))
                        for message in messages
                    )
                    return AgentDecision(type="tool_call", tool_name="write_file", args={"path": "tests/test_parser.py", "content": "print('test')\n"})

            llm = MissingTargetUnderTestLLM()
            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=PassingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=4, max_runtime_seconds=60),
                quality_gate=PassingQualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertTrue(llm.saw_must_edit_feedback)
            steps = await repo.list_steps(job.id)
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertTrue(any(step.content["reason"] == "test_required_tool_forbidden" for step in rejected))
            self.assertIn("tests/test_parser.py", tools.files)
            self.assertIn("print('test')", tools.files["tests/test_parser.py"])

    def test_edit_required_blocks_repeated_run_commands_without_diff(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_modify_files=["crawler.py"])
        for _ in range(INITIAL_NO_DIFF_EXPLORATION_BUDGET):
            state.messages.append({"role": "tool", "tool": "run_command", "exit_code": 0, "metadata": {"command": "cat crawler.py"}})
        workflow = SimpleNamespace(phase=WorkflowPhase.EDIT_REQUIRED)

        block = edit_required_tool_block(state, workflow, "run_command", {"command": "cat crawler.py"})

        self.assertIn("repeated inspection without a diff", block)
        self.assertIn("crawler.py", block)
        self.assertEqual(state.repair_mode, "must_edit")
        self.assertEqual(edit_required_tool_block(state, workflow, "write_file", {"path": "crawler.py"}), "")

    def test_edit_required_rejects_non_target_file_edits(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_modify_files=["crawler.py", "manifest.json"])
        workflow = SimpleNamespace(phase=WorkflowPhase.EDIT_REQUIRED)

        block = edit_required_tool_block(state, workflow, "write_file", {"path": "/workspace/.docode_probe"})

        self.assertIn("required target files", block)
        self.assertIn("crawler.py", block)
        self.assertIn("manifest.json", block)

    def test_must_edit_mode_allows_only_edit_tools_and_git_checks(self) -> None:
        allowed = allowed_tools_for_repair_mode_name("must_edit")

        self.assertEqual(allowed, {"edit_file", "write_file", "replace_in_file", "apply_patch", "git_status", "git_diff"})

    def test_edit_required_after_exploration_only_exposes_edit_and_git_tools(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.inspection = SimpleNamespace()
        state.task_contract = TaskContract(must_modify_files=["crawler.py"])
        state.latest_git_status = ToolResult(tool="git_status", output="", exit_code=0)
        for idx in range(INITIAL_NO_DIFF_EXPLORATION_BUDGET):
            state.messages.append({"role": "tool", "tool": "read_file", "exit_code": 0, "metadata": {"path": f"README_{idx}.md"}})
        definitions = [
            NamedTool("read_file"),
            NamedTool("web_search"),
            NamedTool("fetch_url"),
            NamedTool("run_command"),
            NamedTool("write_file"),
            NamedTool("edit_file"),
            NamedTool("replace_in_file"),
            NamedTool("apply_patch"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertEqual(
            set(names),
            {"write_file", "edit_file", "replace_in_file", "apply_patch", "git_status", "git_diff"},
        )
        self.assertNotIn("read_file", names)
        self.assertNotIn("web_search", names)
        self.assertNotIn("fetch_url", names)
        self.assertNotIn("run_command", names)

    def test_edit_required_before_exploration_budget_keeps_search_tools_available(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="crawl github trends from https://github.com/trending"))
        state.inspection = SimpleNamespace()
        state.task_contract = TaskContract(must_modify_files=["crawler.py"])
        state.latest_git_status = ToolResult(tool="git_status", output="", exit_code=0)
        state.messages.append({"role": "tool", "tool": "list_files", "exit_code": 0, "metadata": {"path": "/workspace"}})
        definitions = [
            NamedTool("list_files"),
            NamedTool("read_file"),
            NamedTool("web_search"),
            NamedTool("fetch_url"),
            NamedTool("write_file"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertIn("fetch_url", names)
        self.assertIn("write_file", names)
        self.assertNotIn("web_search", names)
        self.assertNotIn("read_file", names)
        self.assertNotIn("list_files", names)
        self.assertNotIn("git_status", names)
        self.assertNotIn("git_diff", names)

    def test_edit_required_after_explicit_fetch_allows_web_search_for_crawler(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="crawl github trends from https://github.com/trending"))
        state.inspection = SimpleNamespace()
        state.task_contract = TaskContract(must_modify_files=["crawler.py"])
        state.latest_git_status = ToolResult(tool="git_status", output="", exit_code=0)
        state.messages.append(
            {
                "role": "tool",
                "tool": "fetch_url",
                "exit_code": 1,
                "metadata": {"url": "https://github.com/trending"},
            }
        )
        definitions = [NamedTool("fetch_url"), NamedTool("web_search"), NamedTool("write_file")]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertIn("fetch_url", names)
        self.assertIn("web_search", names)
        self.assertIn("write_file", names)

    def test_crawler_instruction_with_public_url_skips_important_file_reads(self) -> None:
        self.assertTrue(should_skip_important_file_reads("crawl https://github.com/trending every two hours"))
        self.assertFalse(should_skip_important_file_reads("update README.md"))

    def test_instruction_source_urls_extracts_literal_urls(self) -> None:
        urls = instruction_source_urls("check https://github.com/trending and https://example.com/feed.xml, then crawl")
        self.assertEqual(urls, ["https://github.com/trending", "https://example.com/feed.xml"])

    def test_test_required_missing_targets_only_exposes_edit_and_git_tools(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.inspection = SimpleNamespace()
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py"],
            must_run_commands=["python3 crawler.py --dry-run"],
        )
        state.latest_git_status = ToolResult(tool="git_status", output=" M crawler.py\n", exit_code=0)
        state.messages.append(
            {
                "role": "tool",
                "tool": "write_file",
                "exit_code": 0,
                "metadata": {"path": "crawler.py"},
            }
        )
        definitions = [
            NamedTool("read_file"),
            NamedTool("list_files"),
            NamedTool("run_command"),
            NamedTool("write_file"),
            NamedTool("edit_file"),
            NamedTool("replace_in_file"),
            NamedTool("apply_patch"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertEqual(
            set(names),
            {"write_file", "apply_patch", "git_status", "git_diff"},
        )
        self.assertNotIn("run_command", names)
        self.assertNotIn("read_file", names)
        self.assertNotIn("edit_file", names)
        self.assertNotIn("replace_in_file", names)

    def test_targeted_repair_guidance_does_not_block_edits_after_prior_edit(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["crawler.py"],
            "allowed_tools": ["read_file", "write_file", "run_command", "git_status"],
            "initial_inspection_budget": 1,
        }
        state.active_repair_started_at = 0
        state.messages.extend(
            [
                {"role": "tool", "tool": "read_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
                {"role": "tool", "tool": "run_command", "exit_code": 0, "metadata": {"command": "cat crawler.py"}},
            ]
        )
        definitions = [
            NamedTool("read_file"),
            NamedTool("read_file_range"),
            NamedTool("write_file"),
            NamedTool("edit_file"),
            NamedTool("apply_patch"),
            NamedTool("run_command"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]
        forced = targeted_repair_forced_tool(state, "edit_file", {"path": "crawler.py", "old": "x", "new": "y"})

        self.assertEqual(set(names), {tool.name for tool in definitions})
        self.assertIsNone(forced)
        self.assertEqual(targeted_repair_action_block(state, "edit_file", {"path": "crawler.py"}), "")
        self.assertEqual(targeted_repair_exploration_block(state, "read_file"), "")

    def test_targeted_repair_inspect_phase_exposes_normal_local_tools(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["crawler.py"],
            "allowed_tools": ["read_file", "write_file", "search", "web_search", "run_command"],
            "initial_inspection_budget": 2,
        }
        definitions = [NamedTool("read_file"), NamedTool("search"), NamedTool("web_search"), NamedTool("write_file"), NamedTool("run_command")]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertEqual(state.targeted_repair_phase, "inspect_allowed")
        self.assertIn("read_file", names)
        self.assertIn("search", names)
        self.assertIn("web_search", names)
        self.assertIn("write_file", names)
        self.assertIn("run_command", names)

    def test_targeted_repair_wrong_target_edit_is_guidance_not_rejected(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["crawler.py"],
            "allowed_tools": ["write_file", "edit_file", "apply_patch", "run_command"],
            "initial_inspection_budget": 0,
        }

        block = targeted_repair_action_block(state, "write_file", {"path": "README.md"})

        self.assertEqual(block, "")
        self.assertEqual(targeted_repair_action_block(state, "write_file", {"path": "/workspace/crawler.py"}), "")
        self.assertEqual(targeted_repair_action_block(state, "run_command", {"command": "python3 -m unittest discover -s tests"}), "")

    def test_targeted_repair_accepts_equivalent_normalized_target_path(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["tests/../tests/fixtures/trending.html"],
            "allowed_tools": ["write_file"],
            "initial_inspection_budget": 0,
        }

        block = targeted_repair_action_block(state, "write_file", {"path": "/workspace/tests/fixtures/trending.html"})

        self.assertEqual(block, "")

    def test_parsed_value_mismatch_after_inspection_requires_model_edit(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Create crawler.py tests/test_parser.py fixtures/sample.html"))
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py", "fixtures/sample.html"],
            must_run_commands=["python3 -m unittest discover -s tests"],
        )
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "category": "parsed_value_mismatch",
            "signature": "parsed_value_mismatch:field:actual:expected",
            "target_files": ["crawler.py", "tests/test_parser.py", "fixtures/sample.html"],
            "instruction": "Inspect the failing assertion, source, and fixture. Fix source logic or fixture/test consistency based on evidence.",
            "rerun_commands": ["python3 -m unittest discover -s tests"],
            "initial_inspection_budget": 1,
        }
        state.active_repair_started_at = 0
        state.messages.append({"role": "tool", "tool": "read_file", "exit_code": 0, "output": "", "metadata": {"path": "crawler.py"}})

        forced = targeted_repair_forced_tool(state, "read_file")

        self.assertIsNone(forced)
        block = targeted_repair_exploration_block(state, "read_file")
        self.assertEqual(block, "")

    def test_parsed_value_numeric_mismatch_prefers_crawler_target(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Create crawler.py tests/test_parser.py fixtures/sample.html"))
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py", "tests/test_parser.py", "fixtures/sample.html"],
            must_run_commands=["python3 -m unittest discover -s tests"],
        )
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "category": "parsed_value_mismatch",
            "signature": "parsed_value_mismatch:stars_today:0:56",
            "target_files": ["crawler.py", "tests/test_parser.py", "fixtures/sample.html"],
            "instruction": "Fix parser logic or fixture/test consistency so the parser returns the expected value directly.",
            "rerun_commands": ["python3 -m unittest discover -s tests"],
            "initial_inspection_budget": 1,
        }
        state.active_repair_started_at = 0
        state.messages.append({"role": "tool", "tool": "read_file", "exit_code": 0, "output": "", "metadata": {"path": "crawler.py"}})

        forced = targeted_repair_forced_tool(state, "read_file")

        self.assertIsNone(forced)
        self.assertEqual(preferred_targeted_repair_target(state, ["tests/test_parser.py", "crawler.py", "fixtures/sample.html"]), "crawler.py")

    def test_test_required_forces_next_exact_command(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Build crawler."))
        state.inspection = SimpleNamespace()
        state.latest_git_status = ToolResult(tool="git_status", output=" M crawler.py\n", exit_code=0)
        state.task_contract = TaskContract(
            must_modify_files=["crawler.py"],
            must_run_commands=[
                "python3 -m unittest discover -s tests",
                "python3 crawler.py --preflight",
                "python3 crawler.py --source fixtures/sample.html --output data/output.json --dry-run",
            ],
        )
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "write_file",
                    "exit_code": 0,
                    "metadata": {"path": "crawler.py"},
                },
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 0,
                    "metadata": {"command": "python3 -m unittest discover -s tests"},
                },
            ]
        )

        forced = targeted_repair_forced_tool(
            state,
            "run_command",
            {"command": "python3 -m unittest discover -s tests"},
        )

        self.assertIsNotNone(forced)
        assert forced is not None
        self.assertEqual(forced[0], "run_command")
        self.assertEqual(forced[1], {"command": "python3 crawler.py --preflight"})
        self.assertEqual(forced[2], "test_required_requires_exact_command")

    def test_targeted_repair_modified_target_does_not_force_exact_rerun_command(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Create GitHub trends crawler."))
        state.repair_mode = "targeted_repair"
        state.active_repair_started_at = 0
        state.active_repair_action = {
            "category": "parsed_value_mismatch",
            "target_files": ["crawler.py"],
            "rerun_commands": ["python3 -m unittest discover -s tests"],
            "initial_inspection_budget": 0,
        }
        state.messages.append(
            {
                "role": "tool",
                "tool": "write_file",
                "exit_code": 0,
                "metadata": {"path": "crawler.py"},
            }
        )

        forced = targeted_repair_forced_tool(
            state,
            "run_command",
            {"command": "python3 crawler.py --source fixtures/sample.html --output data/output.json --dry-run"},
        )

        self.assertIsNone(forced)

    def test_targeted_repair_edit_forced_tool_list_keeps_normal_tools_except_explicit_unsafe_forbidden(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["crawler.py"],
            "allowed_tools": [
                "read_file",
                "search",
                "list_files",
                "web_search",
                "fetch_url",
                "write_file",
                "apply_patch",
                "run_command",
                "git_status",
                "git_diff",
            ],
            "forbidden_tools": ["web_search", "fetch_url", "preview", "logs"],
            "initial_inspection_budget": 0,
        }
        definitions = [
            NamedTool("read_file"),
            NamedTool("search"),
            NamedTool("list_files"),
            NamedTool("web_search"),
            NamedTool("fetch_url"),
            NamedTool("write_file"),
            NamedTool("apply_patch"),
            NamedTool("run_command"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertEqual(state.targeted_repair_phase, "edit_forced")
        self.assertEqual(set(names), {"read_file", "search", "list_files", "write_file", "apply_patch", "run_command", "git_status", "git_diff"})

    def test_targeted_repair_explicit_unsafe_forbidden_tools_are_blocked(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.repair_mode = "targeted_repair"
        state.active_repair_action = {
            "target_files": ["crawler.py"],
            "forbidden_tools": ["web_search", "fetch_url", "run_command"],
            "initial_inspection_budget": 0,
        }
        definitions = [
            NamedTool("web_search"),
            NamedTool("fetch_url"),
            NamedTool("read_file"),
            NamedTool("write_file"),
            NamedTool("apply_patch"),
            NamedTool("run_command"),
            NamedTool("git_status"),
            NamedTool("git_diff"),
        ]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

        self.assertNotIn("web_search", names)
        self.assertNotIn("fetch_url", names)
        self.assertIn("run_command", names)
        self.assertIn("write_file", names)
        self.assertIn("apply_patch", names)
        self.assertIn("git_status", names)
        self.assertIn("git_diff", names)
        self.assertIn("blocked by the active targeted repair action", repair_mode_tool_block(state, "fetch_url"))
        self.assertEqual(repair_mode_tool_block(state, "run_command"), "")

    def test_latest_failed_required_command_finds_classifiable_failure(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_run_commands=["python3 -m unittest discover -s tests"])
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 1,
                    "output": "NameError: name '_GitHubTrendingParser' is not defined. Did you mean: 'GitHubTrendingParser'?",
                    "metadata": {"command": "python3 -m unittest discover -s tests"},
                },
                {
                    "role": "tool",
                    "tool": "read_file",
                    "exit_code": 0,
                    "output": "crawler.py",
                    "metadata": {"path": "crawler.py"},
                },
            ]
        )

        failed = latest_failed_required_command(state)

        self.assertIsNotNone(failed)
        self.assertEqual(failed.metadata["command"], "python3 -m unittest discover -s tests")
        self.assertIn("_GitHubTrendingParser", failed.output)

    def test_latest_failed_required_command_ignores_resolved_failure(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_run_commands=["python3 -m unittest discover -s tests"])
        state.messages.extend(
            [
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 1,
                    "output": "NameError: name 'Old' is not defined",
                    "metadata": {"command": "python3 -m unittest discover -s tests"},
                },
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 0,
                    "output": "OK",
                    "metadata": {"command": "python3 -m unittest discover -s tests"},
                },
            ]
        )

        self.assertIsNone(latest_failed_required_command(state))

    async def test_agent_loop_auto_finalizes_tool_calls_after_final_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Update README.md.\nverify with: echo checked",
                )
            )
            tools = FakeTools()
            llm = FinalReadyToolLoopLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=8, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Completed the requested changes in README.md.")
            self.assertEqual(tools.call_count, 2)
            steps = await repo.list_steps(job.id)
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertEqual(rejected, [])
            auto = [step for step in steps if step.content.get("type") == "auto_final_candidate"]
            self.assertEqual(auto[0].content["reason"], "final_ready_tool_auto_finalized")
            self.assertEqual(auto[0].content["workflow_state"]["phase"], "FINAL_READY")
            tool_calls = [step for step in steps if step.content.get("type") == "tool_call"]
            self.assertEqual([step.content["args"].get("command") for step in tool_calls if step.content["tool"] == "run_command"], ["echo checked"])

    async def test_agent_loop_auto_finalizes_bad_llm_json_after_final_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Update README.md.\nverify with: echo checked",
                )
            )
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=FinalReadyMalformedLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Completed the requested changes in README.md.")
            self.assertEqual(tools.call_count, 2)
            steps = await repo.list_steps(job.id)
            auto = [step for step in steps if step.content.get("type") == "auto_final_candidate"]
            self.assertEqual(auto[0].content["reason"], "final_ready_llm_decision_failed")
            self.assertIn("Expecting value", auto[0].content["detail"])

    async def test_agent_loop_repairs_after_independent_reviewer_blocks(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update README.md"))
            tools = FakeTools()
            llm = ReviewerRepairLLM()
            reviewer = BlockingThenPassingReviewer()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                reviewer=reviewer,
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=8, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn("README", result.result_summary or "")
            self.assertEqual(tools.files["README.md"], "substantive\n")
            self.assertEqual(reviewer.calls, 2)
            self.assertTrue(reviewer.seen_tool_results)
            self.assertTrue(llm.observed_review_feedback)
            steps = await repo.list_steps(job.id)
            review_steps = [step for step in steps if step.content.get("type") == "independent_review"]
            self.assertEqual([step.content["passed"] for step in review_steps], [False, True])
            verifier_steps = [step for step in steps if step.kind == "verifier"]
            self.assertEqual(len(verifier_steps), 2)

    async def test_agent_loop_targeted_repair_blocks_explicitly_forbidden_unsafe_tool_only(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Fix crawler.py.\nverify with: python3 -m unittest discover -s tests",
                )
            )
            tools = TargetedRepairTools()
            llm = TargetedRepairLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=PassingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=12, max_runtime_seconds=60),
                quality_gate=PassingQualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertIn("crawler.py", result.result_summary or "")
            self.assertTrue(llm.observed_targeted_feedback)
            steps = await repo.list_steps(job.id)
            repair_steps = [step for step in steps if step.content.get("type") == "repair_action"]
            self.assertEqual(repair_steps[0].content["repair_action"]["category"], "import_error_missing_symbol")
            rejected = [step for step in steps if step.content.get("type") == "decision_rejected"]
            self.assertTrue(any(step.content["reason"] == "targeted_repair_tool_forbidden" for step in rejected))

    async def test_agent_loop_generic_calculator_repair_reads_edits_reruns_and_finalizes(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="u1",
                    instruction="Fix calculator.py.\nverify with: python3 -m unittest discover -s tests",
                )
            )
            tools = CalculatorRepairTools()
            llm = CalculatorRepairLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=PassingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=12, max_runtime_seconds=60),
                quality_gate=PassingQualityGate(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(tools.files["calculator.py"], "def add(a, b):\n    return a + b\n")
            steps = await repo.list_steps(job.id)
            repair_steps = [step for step in steps if step.content.get("type") == "repair_action"]
            self.assertEqual(repair_steps[0].content["repair_action"]["target_files"], ["calculator.py"])
            commands = [
                step.content["args"]["command"]
                for step in steps
                if step.content.get("type") == "tool_call" and step.content.get("tool") == "run_command"
            ]
            self.assertEqual(
                commands,
                [
                    "python3 -m unittest discover -s tests",
                    "python3 -m unittest discover -s tests",
                    "python3 -m unittest discover -s tests",
                ],
            )

    def test_targeted_repair_edit_forced_does_not_generate_or_force_write_content(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Fix calculator.py."))
        state.repair_mode = "targeted_repair"
        state.active_repair_started_at = 0
        state.active_repair_action = {
            "category": "parsed_value_mismatch",
            "target_files": ["calculator.py"],
            "rerun_commands": ["python3 -m unittest discover -s tests"],
            "initial_inspection_budget": 0,
        }

        forced = targeted_repair_forced_tool(state, "run_command", {"command": "python3 -m unittest discover -s tests"})
        block = targeted_repair_action_block(state, "run_command", {"command": "python3 -m unittest discover -s tests"})

        self.assertIsNone(forced)
        self.assertEqual(block, "")

    def test_parsed_value_mismatch_prefers_contract_source_targets(self) -> None:
        action = RepairAction(
            category="parsed_value_mismatch",
            signature="parsed_value_mismatch:value:False:True",
            reason="Parser returned the wrong value.",
            target_files=["tests/test_parser.py"],
            rerun_commands=["python -m unittest discover -s tests"],
        )
        contract = TaskContract(must_modify_files=["parser.py", "fixtures/products.html"])

        refined = refine_repair_action_targets(action, contract)

        self.assertEqual(task_contract_source_targets(contract), ["parser.py"])
        self.assertEqual(refined.target_files, ["parser.py", "tests/test_parser.py"])

    async def test_activate_targeted_repair_resets_consecutive_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="Fix crawler.py."))
            state = AgentState(job=job)
            state.consecutive_failures = 12
            loop = CodingAgentLoop(
                llm=ScriptedLLM(),
                tools=FakeTools(),
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60, max_consecutive_failures=12),
            )

            await loop.activate_targeted_repair(
                state,
                RepairAction(
                    category="missing_required_field",
                    signature="missing_required_field:owner",
                    reason="Parser records are missing owner.",
                    target_files=["crawler.py"],
                    rerun_commands=["python3 -m unittest discover -s tests"],
                    instruction="Edit crawler.py so parser records include owner.",
                ),
                result=ToolResult(
                    tool="run_command",
                    output="AssertionError: 'owner' not found in {}",
                    exit_code=1,
                    metadata={"command": "python3 -m unittest discover -s tests"},
                ),
            )

            self.assertEqual(state.consecutive_failures, 0)
            self.assertEqual(state.repair_mode, "targeted_repair")
            steps = await repo.list_steps(job.id)
            self.assertEqual(steps[-1].content["type"], "repair_action")

    def test_same_targeted_repair_edit_allows_more_editing_and_rerun(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Fix crawler.py."))
        state.repair_mode = "targeted_repair"
        state.active_repair_started_at = 0
        state.active_repair_action = {
            "category": "dependency_unavailable",
            "signature": "dependency_unavailable",
            "target_files": ["crawler.py"],
            "rerun_commands": ["python -m unittest discover -s tests"],
            "initial_inspection_budget": 1,
        }
        state.messages.append({"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}})
        definitions = [NamedTool("read_file"), NamedTool("write_file"), NamedTool("run_command"), NamedTool("git_status")]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]
        forced = targeted_repair_forced_tool(state, "read_file", {"path": "crawler.py"})

        self.assertEqual(set(names), {"read_file", "write_file", "run_command", "git_status"})
        self.assertIsNone(forced)

    def test_failed_repair_rerun_allows_same_target_edit_again(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="Fix crawler.py."))
        state.repair_mode = "targeted_repair"
        state.active_repair_started_at = 0
        state.active_repair_action = {
            "category": "dependency_unavailable",
            "signature": "dependency_unavailable",
            "target_files": ["crawler.py"],
            "rerun_commands": ["python -m unittest discover -s tests"],
            "initial_inspection_budget": 1,
        }
        state.messages.extend(
            [
                {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 1,
                    "output": "FileNotFoundError: out.json",
                    "metadata": {"command": "python -m unittest discover -s tests"},
                },
            ]
        )
        definitions = [NamedTool("read_file"), NamedTool("write_file"), NamedTool("run_command"), NamedTool("git_status")]

        names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]
        forced = targeted_repair_forced_tool(state, "write_file", {"path": "crawler.py", "content": "fixed"})

        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("run_command", names)
        self.assertIsNone(forced)
        self.assertEqual(targeted_repair_action_block(state, "write_file", {"path": "crawler.py"}), "")

    async def test_new_repair_action_same_target_allows_another_edit_cycle(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="Fix crawler.py."))
            state = AgentState(job=job)
            state.messages.extend(
                [
                    {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
                    {
                        "role": "tool",
                        "tool": "run_command",
                        "exit_code": 1,
                        "output": "AssertionError: {'products': []} != []",
                        "metadata": {"command": "python -m unittest discover -s tests"},
                    },
                ]
            )
            loop = CodingAgentLoop(
                llm=ScriptedLLM(),
                tools=FakeTools(),
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            await loop.activate_targeted_repair(
                state,
                RepairAction(
                    category="parsed_value_mismatch",
                    signature="parsed_value_mismatch:products:dict:list",
                    reason="Parser returned wrapper object instead of product list.",
                    target_files=["crawler.py"],
                    rerun_commands=["python -m unittest discover -s tests"],
                    instruction="Edit crawler.py so fetch_and_parse returns the product list.",
                ),
                result=ToolResult(
                    tool="run_command",
                    output="AssertionError: {'products': []} != []",
                    exit_code=1,
                    metadata={"command": "python -m unittest discover -s tests"},
                ),
            )

            definitions = [NamedTool("read_file"), NamedTool("write_file"), NamedTool("run_command"), NamedTool("git_status")]
            names = [tool.name for tool in allowed_tool_definitions_for_state(definitions, state)]

            self.assertEqual(state.active_repair_started_at, 2)
            self.assertIn("read_file", names)
            self.assertIn("write_file", names)
            self.assertIn("run_command", names)
            self.assertEqual(targeted_repair_action_block(state, "write_file", {"path": "crawler.py"}), "")
            forced = targeted_repair_forced_tool(state, "write_file", {"path": "crawler.py", "content": "fixed"})
            self.assertIsNone(forced)

    async def test_new_repair_action_different_target_still_allows_model_to_choose_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="Fix parser.py."))
            state = AgentState(job=job)
            state.messages.extend(
                [
                    {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "crawler.py"}},
                    {
                        "role": "tool",
                        "tool": "run_command",
                        "exit_code": 1,
                        "output": "AssertionError: 'name' not found in {}",
                        "metadata": {"command": "python -m unittest discover -s tests"},
                    },
                ]
            )
            loop = CodingAgentLoop(
                llm=ScriptedLLM(),
                tools=FakeTools(),
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            await loop.activate_targeted_repair(
                state,
                RepairAction(
                    category="missing_required_field",
                    signature="missing_required_field:name",
                    reason="Parser records are missing name.",
                    target_files=["parser.py"],
                    rerun_commands=["python -m unittest discover -s tests"],
                    instruction="Edit parser.py so records include name.",
                ),
                result=ToolResult(
                    tool="run_command",
                    output="AssertionError: 'name' not found in {}",
                    exit_code=1,
                    metadata={"command": "python -m unittest discover -s tests"},
                ),
            )

            self.assertEqual(state.active_repair_started_at, 2)
            self.assertEqual(targeted_repair_action_block(state, "write_file", {"path": "parser.py"}), "")
            block = targeted_repair_action_block(state, "write_file", {"path": "crawler.py"})
            self.assertEqual(block, "")

    async def test_agent_loop_recovers_from_transient_tool_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools(fail_first_write=True)
            llm = ToolRepairLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Recovered from a transient tool error and updated README.")
            self.assertEqual(tools.files["README.md"], "repaired\n")
            self.assertTrue(llm.observed_tool_error)
            steps = await repo.list_steps(job.id)
            tool_results = [step for step in steps if step.content.get("type") == "tool_result"]
            self.assertEqual(len(tool_results), 2)
            self.assertEqual(tool_results[0].content["exit_code"], 1)
            self.assertIn("sandbox write unavailable", tool_results[0].content["output"])
            self.assertEqual(tool_results[0].content["metadata"]["exception_type"], "RuntimeError")
            self.assertEqual(tool_results[1].content["exit_code"], 0)

    async def test_agent_loop_stops_after_repeated_unusable_model_output(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=UnusableDecisionLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60, max_consecutive_failures=2),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_consecutive_failures_exceeded")
            self.assertEqual(tools.call_count, 0)
            self.assertIsNotNone(result.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(result.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            self.assertTrue((Path(tmp) / job.id / "failure_report.md").exists())
            result_payload = json.loads((Path(tmp) / job.id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result_payload["status"], "failed")
            self.assertEqual(result_payload["failure_reason"], "max_consecutive_failures_exceeded")
            self.assertEqual(result_payload["changed_files"], [])
            steps = await repo.list_steps(job.id)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(len(llm_errors), 2)
            self.assertEqual(llm_errors[0].content["reason"], "model_returned_unusable_decision")

    async def test_agent_loop_fails_fast_on_llm_auth_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=UnauthorizedLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60, max_consecutive_failures=5),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "llm_auth_failed")
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(len(llm_errors), 1)
            self.assertEqual(llm_errors[0].content["reason"], "llm_auth_failed")

    async def test_agent_loop_retries_provider_unavailable_before_failing(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = ProviderUnavailableLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60, max_consecutive_failures=5),
                llm_retry_delays=(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "llm_provider_unavailable:provider_upstream_unavailable")
            self.assertEqual(llm.calls, 3)
            self.assertEqual(tools.call_count, 0)
            steps = await repo.list_steps(job.id)
            llm_retries = [step for step in steps if step.content.get("type") == "llm_retry"]
            self.assertEqual(len(llm_retries), 2)
            llm_errors = [step for step in steps if step.content.get("type") == "llm_error"]
            self.assertEqual(len(llm_errors), 1)
            self.assertEqual(llm_errors[0].content["reason"], "llm_provider_unavailable:provider_upstream_unavailable")

    async def test_agent_loop_recovers_from_retryable_provider_error(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update readme"))
            tools = FakeTools()
            llm = RecoveringProviderUnavailableLLM()

            loop = CodingAgentLoop(
                llm=llm,
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=10, max_runtime_seconds=60, max_consecutive_failures=5),
                llm_retry_delays=(),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.SUCCEEDED)
            self.assertEqual(result.result_summary, "Recovered after provider retry.")
            self.assertEqual(tools.files["README.md"], "recovered\n")
            self.assertEqual(llm.calls, 3)
            steps = await repo.list_steps(job.id)
            llm_retries = [step for step in steps if step.content.get("type") == "llm_retry"]
            self.assertEqual(len(llm_retries), 1)

    async def test_agent_loop_records_stuck_detector_step_after_clean_no_edit_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="update README.md"))
            tools = FakeTools()

            loop = CodingAgentLoop(
                llm=NoEditToolLoopLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(Path(tmp), repo),
                stop_policy=StopPolicy(max_iterations=INITIAL_NO_DIFF_EXPLORATION_BUDGET + 4, max_runtime_seconds=60, max_consecutive_failures=20),
            )

            result = await loop.run(job)

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.failure_reason, "max_iterations_exceeded")
            steps = await repo.list_steps(job.id)
            stuck_steps = [step for step in steps if step.content.get("type") == "stuck_detector"]
            self.assertGreaterEqual(len(stuck_steps), 1)
            self.assertEqual(stuck_steps[0].content["reason"], "no_diff_after_multiple_iterations")
