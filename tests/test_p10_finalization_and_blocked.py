"""P10 secondary fixes for the holdout-baseline-v1 forensics.

Two failure classes remained after the P8 apply_patch fix:

1. Finalization false-reject (node_bugfix, small_feature): the workspace already
   held a correct, non-empty diff, but the loop hit a budget limit and threw the
   work away with `max_consecutive_failures_exceeded`. Budget exhaustion is a
   controller-side limit; it is not evidence that the diff is wrong.

2. Missing blocked path (unsatisfiable_task): `final_candidate` was the model's
   only terminal move, so an impossible premise pushed it to fabricate one in
   order to look complete. It failed `premise_not_fabricated`.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from docode.agent.failure_taxonomy import FailureCategory, category_for_reason
from docode.agent.loop import BUDGET_EXHAUSTION_STOP_REASONS, CodingAgentLoop
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.types import ToolResult
from docode.llm.decision import AgentDecision, parse_decision
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository


class BlockedTools:
    """Minimal tool surface; the blocked path only needs git_status."""

    def __init__(self, status: str = "") -> None:
        self.status = status

    def definitions(self):
        return []

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=self.status)

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output="")


class SilentLLM:
    async def decide(self, *, system, messages, tools, context):
        raise AssertionError("the blocked path must not call the LLM again")


def build_loop(repo: InMemoryJobRepository, tmp: str, tools: object) -> CodingAgentLoop:
    return CodingAgentLoop(
        llm=SilentLLM(),
        tools=tools,
        verifier=CodingVerifier(),
        repository=repo,
        exporter=ArtifactExporter(Path(tmp), repo),
        stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
    )


class BlockedDecisionParsingTests(IsolatedAsyncioTestCase):
    def test_parse_blocked_decision(self) -> None:
        decision = parse_decision(
            '{"type":"blocked","blocked_reason":"task_unsatisfiable",'
            '"summary":"parse_config() does not exist anywhere in the repo",'
            '"evidence":["grep -rn parse_config . -> no matches"]}'
        )

        self.assertEqual(decision.type, "blocked")
        self.assertEqual(decision.blocked_reason, "task_unsatisfiable")
        self.assertEqual(decision.evidence, ["grep -rn parse_config . -> no matches"])

    def test_blocked_defaults_reason_and_accepts_scalar_evidence(self) -> None:
        decision = parse_decision('{"type":"blocked","summary":"no such API","evidence":"searched, absent"}')

        self.assertEqual(decision.blocked_reason, "task_unsatisfiable")
        self.assertEqual(decision.evidence, ["searched, absent"])

    def test_prompt_advertises_blocked_so_model_can_refuse(self) -> None:
        from docode.llm.decision import format_decision_prompt

        prompt = format_decision_prompt(system="s", messages=[], tools=[], context={})

        self.assertIn('"type":"blocked"', prompt)
        self.assertIn("Never invent a plausible-looking premise", prompt)


class BlockedTerminalPathTests(IsolatedAsyncioTestCase):
    async def test_blocked_stops_job_without_marking_it_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(id=new_id("job"), user_id="u1", instruction="Refactor parse_config()")
            )
            state = AgentState(job=job)
            state.inspection = SimpleNamespace()
            loop = build_loop(repo, tmp, BlockedTools())

            result = await loop.handle_blocked(
                state,
                AgentDecision(
                    type="blocked",
                    summary="parse_config() does not exist in this repository",
                    blocked_reason="task_unsatisfiable",
                    evidence=["grep -rn 'parse_config' . returned no matches"],
                ),
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.status, JobStatus.STOPPED)
            self.assertEqual(result.failure_reason, "blocked:task_unsatisfiable")
            self.assertEqual(result.terminal_result["category"], FailureCategory.TASK_UNSATISFIABLE.value)
            self.assertFalse(result.terminal_result["strict_success"])
            self.assertTrue(result.terminal_result["harness_valid"])

    async def test_blocked_requires_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(id=new_id("job"), user_id="u1", instruction="Refactor parse_config()")
            )
            state = AgentState(job=job)
            state.inspection = SimpleNamespace()
            loop = build_loop(repo, tmp, BlockedTools())

            result = await loop.handle_blocked(
                state,
                AgentDecision(type="blocked", summary="it is impossible", evidence=[]),
            )

            # Rejected, not terminal: the model must go gather proof first.
            self.assertIsNone(result)

    async def test_blocked_requires_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(
                CodingJob(id=new_id("job"), user_id="u1", instruction="Refactor parse_config()")
            )
            state = AgentState(job=job)
            state.inspection = SimpleNamespace()
            loop = build_loop(repo, tmp, BlockedTools())

            result = await loop.handle_blocked(
                state,
                AgentDecision(type="blocked", summary="   ", evidence=["proof"]),
            )

            self.assertIsNone(result)

    def test_unsatisfiable_is_not_miscategorised_as_runtime_failure(self) -> None:
        self.assertEqual(category_for_reason("blocked:task_unsatisfiable"), FailureCategory.TASK_UNSATISFIABLE)
        self.assertEqual(category_for_reason("task_unsatisfiable"), FailureCategory.TASK_UNSATISFIABLE)


class FinalizationRescueTests(IsolatedAsyncioTestCase):
    def test_consecutive_failure_exhaustion_is_a_budget_limit(self) -> None:
        # The P1 forensics showed node_bugfix / small_feature dying here with a
        # correct diff already on disk.
        self.assertIn("max_consecutive_failures_exceeded", BUDGET_EXHAUSTION_STOP_REASONS)
        self.assertIn("max_iterations_exceeded", BUDGET_EXHAUSTION_STOP_REASONS)

    def test_wrongness_signals_are_not_treated_as_budget_exhaustion(self) -> None:
        # These mean the work is actually bad; they must still fail hard.
        self.assertNotIn("non_convergent", BUDGET_EXHAUSTION_STOP_REASONS)
        self.assertNotIn("cancelled", BUDGET_EXHAUSTION_STOP_REASONS)

    async def test_auto_finalize_declines_when_stop_reason_is_not_exhaustion(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="fix bug"))
            state = AgentState(job=job)
            state.inspection = SimpleNamespace()
            loop = build_loop(repo, tmp, BlockedTools())

            result = await loop.maybe_auto_finalize_before_stop(state, "non_convergent")

            self.assertIsNone(result)
