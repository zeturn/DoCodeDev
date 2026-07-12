"""Integration scenarios for structured outcome + no-progress control.

Three scripted agents test the full loop: repeated reader (blocked),
finalization loop (blocked), and repair recovery (passes).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.agent.action_keys import final_candidate_action_key
from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.runtime import AgentDecision
from docode.storage.models import CodingJob, JobStatus, new_id

from tests.support.local_tools import DiagnosticLocalTools
from tests.support.repository import RecordingRepository


# ── scripted LLMs ────────────────────────────────────────────────────────

class RepeatedReaderLLM:
    """Always returns read_file on guidebook.md — should be blocked."""

    async def decide(self, messages):
        return AgentDecision(
            type="tool_call",
            tool_name="read_file",
            tool_args={"path": "guidebook.md"},
        )


class FinalizationLoopLLM:
    """Always submits final_candidate with empty summary — blocker loop."""

    _count = 0

    async def decide(self, messages):
        self._count += 1
        return AgentDecision(
            type="final_candidate",
            tool_name="final_candidate",
            tool_args={},
            summary="" if self._count <= 2 else "work done",
        )


class RepairRecoveryLLM:
    """Runs command → inspects → edits → re-runs → succeeds."""
    _step = 0

    async def decide(self, messages):
        self._step += 1
        if self._step == 1:
            return AgentDecision(
                type="tool_call",
                tool_name="run_command",
                tool_args={"command": "echo hello"},
            )
        if self._step == 2:
            return AgentDecision(
                type="tool_call",
                tool_name="read_file",
                tool_args={"path": "guidebook.md"},
            )
        if self._step == 3:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                tool_args={"path": "guidebook.md", "content": "# Guidebook\nDone."},
            )
        if self._step == 4:
            return AgentDecision(
                type="tool_call",
                tool_name="run_command",
                tool_args={"command": "echo hello"},
            )
        return AgentDecision(
            type="final_candidate",
            tool_name="final_candidate",
            tool_args={},
            summary="Completed guidebook",
        )


# ── tests ────────────────────────────────────────────────────────────────


class RepeatedReaderTests(IsolatedAsyncioTestCase):
    async def test_repeated_reader_blocked_before_max_iterations(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "guidebook.md").write_text("old", encoding="utf-8")
            repo = RecordingRepository()
            job = await repo.create_job(CodingJob(
                id=new_id("job"), user_id="u",
                instruction="Complete guidebook.md",
                max_iterations=36, max_runtime_seconds=60,
                max_consecutive_failures=10, max_tool_calls=80,
            ))
            tools = DiagnosticLocalTools(ws)
            loop = CodingAgentLoop(
                llm=RepeatedReaderLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(root / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=60, max_consecutive_failures=10, max_tool_calls=80),
                quality_gate=QualityGate(),
            )
            result = await loop.run(job)
            self.assertIsNotNone(result.failure_reason)
            self.assertIn("no_progress", result.failure_reason or "")


class FinalizationLoopTests(IsolatedAsyncioTestCase):
    async def test_finalization_loop_stops_with_blocker(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "guidebook.md").write_text("old", encoding="utf-8")
            repo = RecordingRepository()
            job = await repo.create_job(CodingJob(
                id=new_id("job"), user_id="u",
                instruction="Complete guidebook.md",
                max_iterations=36, max_runtime_seconds=60,
                max_consecutive_failures=10, max_tool_calls=80,
            ))
            tools = DiagnosticLocalTools(ws)
            loop = CodingAgentLoop(
                llm=FinalizationLoopLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(root / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=60, max_consecutive_failures=10, max_tool_calls=80),
                quality_gate=QualityGate(),
            )
            result = await loop.run(job)
            self.assertNotEqual(result.status, JobStatus.SUCCEEDED)


class RepairRecoveryTests(IsolatedAsyncioTestCase):
    async def test_repair_recovery_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ws = root / "workspace"
            ws.mkdir()
            (ws / "guidebook.md").write_text("old", encoding="utf-8")
            repo = RecordingRepository()
            job = await repo.create_job(CodingJob(
                id=new_id("job"), user_id="u",
                instruction="Complete guidebook.md",
                max_iterations=36, max_runtime_seconds=900,
                max_consecutive_failures=10, max_tool_calls=80,
            ))
            tools = DiagnosticLocalTools(ws)
            loop = CodingAgentLoop(
                llm=RepairRecoveryLLM(),
                tools=tools,
                verifier=CodingVerifier(),
                repository=repo,
                exporter=ArtifactExporter(root / "artifacts", repo, workspace_file_reader=tools.read_file),
                stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=900, max_consecutive_failures=10, max_tool_calls=80),
                quality_gate=QualityGate(),
            )
            result = await loop.run(job)
            self.assertEqual(result.status, JobStatus.SUCCEEDED)
