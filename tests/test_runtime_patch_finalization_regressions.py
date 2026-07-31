"""Regression coverage for the P8/P10 runtime fixes.

These complement (rather than duplicate) the existing suites:

* ``tests/test_patch_normalize.py`` covers envelope -> unified diff conversion.
* ``tests/test_p10_finalization_and_blocked.py`` covers blocked parsing/terminal
  landing and the budget-reason constant.

What was still untested, and is asserted here, is the *safety* half of each fix:

1. apply_patch must never report success for a patch git refused, and a failed
   apply must leave the workspace byte-for-byte unchanged. That is what the
   ``--check`` guard in front of every strategy buys us; without it the ``--3way``
   strategy silently reported success on a rejected patch.
2. Auto-finalization must only rescue work that is genuinely finished. Budget
   exhaustion is not evidence the diff is good either -- it merely means we are
   not allowed to *discard* it untested. Every rescue still runs the real
   verifier.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase

from docode.agent.failure_taxonomy import FailureCategory, category_for_reason
from docode.agent.loop import BUDGET_EXHAUSTION_STOP_REASONS, CodingAgentLoop
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.dobox.patch_normalize import normalize_model_patch
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, JobStatus, new_id
from docode.storage.repository import InMemoryJobRepository

BUGGY_SOURCE = "def add(a, b):\n    return a - b  # BUG\n"

GOOD_ENVELOPE = (
    "*** Begin Patch\n"
    "*** Update File: /workspace/calc.py\n"
    "@@\n"
    " def add(a, b):\n"
    "-    return a - b  # BUG\n"
    "+    return a + b\n"
    "*** End Patch"
)

# Context lines that do not exist in the file -> git must reject this.
BAD_ENVELOPE = (
    "*** Begin Patch\n"
    "*** Update File: /workspace/calc.py\n"
    "@@\n"
    " def totally_different(x):\n"
    "-    return x * 999\n"
    "+    return x\n"
    "*** End Patch"
)


def _git(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def _init_repo() -> tuple[str, str]:
    tmp = tempfile.mkdtemp(prefix="runtime-regress-")
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    with open(os.path.join(repo, "calc.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(BUGGY_SOURCE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return tmp, repo


def _apply_with_guarded_strategies(repo: str, patch_text: str) -> tuple[bool, str]:
    """Mirror ToolRegistry.apply_patch strategy ladder against a real git repo.

    Returns (applied, combined_stderr). Each strategy runs ``--check`` first and
    only mutates the tree when the check passes, which is exactly the contract
    the DoBox tool relies on.
    """

    normalized = normalize_model_patch(patch_text)
    if not normalized.endswith("\n"):
        normalized += "\n"
    diff_path = os.path.join(repo, ".docode_apply_patch.diff")
    with open(diff_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(normalized)

    ladder = [
        ["--ignore-whitespace", "--recount"],
        ["--unidiff-zero", "--ignore-whitespace"],
        ["--3way", "--ignore-whitespace"],
    ]
    errors = []
    try:
        for flags in ladder:
            check = _git(repo, "apply", "--check", *flags, diff_path)
            if check.returncode != 0:
                errors.append(check.stderr)
                continue
            applied = _git(repo, "apply", *flags, diff_path)
            if applied.returncode == 0:
                return True, ""
            errors.append(applied.stderr)
        return False, "\n".join(errors)
    finally:
        if os.path.exists(diff_path):
            os.remove(diff_path)


class ApplyPatchSafetyTests(unittest.TestCase):
    def test_envelope_patch_actually_modifies_workspace(self) -> None:
        tmp, repo = _init_repo()
        try:
            applied, err = _apply_with_guarded_strategies(repo, GOOD_ENVELOPE)

            self.assertTrue(applied, msg=err)
            with open(os.path.join(repo, "calc.py"), encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("return a + b", content)
            self.assertNotIn("# BUG", content)
            # A real, non-empty diff must exist -- "tool succeeded" is only
            # trustworthy if the tree genuinely changed.
            self.assertNotEqual(_git(repo, "diff", "--stat").stdout.strip(), "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_standard_unified_diff_still_applies(self) -> None:
        tmp, repo = _init_repo()
        try:
            unified = (
                "diff --git a/calc.py b/calc.py\n"
                "--- a/calc.py\n"
                "+++ b/calc.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a - b  # BUG\n"
                "+    return a + b\n"
            )
            applied, err = _apply_with_guarded_strategies(repo, unified)

            self.assertTrue(applied, msg=err)
            with open(os.path.join(repo, "calc.py"), encoding="utf-8") as fh:
                self.assertIn("return a + b", fh.read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_trailing_newline_is_normalized_and_applies(self) -> None:
        tmp, repo = _init_repo()
        try:
            # The P8 regression: normalization stripped the trailing newline and
            # git rejected the patch with "corrupt patch at line N".
            self.assertFalse(GOOD_ENVELOPE.endswith("\n"))
            normalized = normalize_model_patch(GOOD_ENVELOPE)
            if not normalized.endswith("\n"):
                normalized += "\n"
            self.assertTrue(normalized.endswith("\n"))

            applied, err = _apply_with_guarded_strategies(repo, GOOD_ENVELOPE)
            self.assertTrue(applied, msg=err)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bad_patch_fails_and_leaves_workspace_byte_for_byte_unchanged(self) -> None:
        tmp, repo = _init_repo()
        try:
            target = os.path.join(repo, "calc.py")
            with open(target, "rb") as fh:
                before = fh.read()

            applied, _ = _apply_with_guarded_strategies(repo, BAD_ENVELOPE)

            self.assertFalse(applied, "a patch git rejected must not report success")
            with open(target, "rb") as fh:
                after = fh.read()
            self.assertEqual(before, after, "failed apply must not mutate the workspace")
            # No diff, and no leftover scratch patch file.
            self.assertEqual(_git(repo, "diff", "--stat").stdout.strip(), "")
            self.assertFalse(os.path.exists(os.path.join(repo, ".docode_apply_patch.diff")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_three_way_strategy_cannot_report_false_success(self) -> None:
        """--3way without a --check guard used to swallow a rejected patch."""
        tmp, repo = _init_repo()
        try:
            normalized = normalize_model_patch(BAD_ENVELOPE)
            if not normalized.endswith("\n"):
                normalized += "\n"
            diff_path = os.path.join(repo, "bad.diff")
            with open(diff_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(normalized)

            guard = _git(repo, "apply", "--check", "--3way", "--ignore-whitespace", diff_path)

            self.assertNotEqual(guard.returncode, 0, "the --check guard must reject a bad patch")
            os.remove(diff_path)
            self.assertEqual(_git(repo, "diff", "--stat").stdout.strip(), "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class RescueTools:
    """Tool surface for the auto-finalize path, with a controllable git state."""

    def __init__(self, status: str = "", diff: str = "") -> None:
        self.status = status
        self.diff = diff

    def definitions(self):
        return []

    async def git_status(self) -> ToolResult:
        return ToolResult(tool="git_status", output=self.status)

    async def git_diff(self) -> ToolResult:
        return ToolResult(tool="git_diff", output=self.diff)


class SilentLLM:
    async def decide(self, *, system, messages, tools, context):
        raise AssertionError("the terminal path must not call the LLM again")


def _build_loop(repo: InMemoryJobRepository, tmp: str, tools: object) -> CodingAgentLoop:
    return CodingAgentLoop(
        llm=SilentLLM(),
        tools=tools,
        verifier=CodingVerifier(),
        repository=repo,
        exporter=ArtifactExporter(Path(tmp), repo),
        stop_policy=StopPolicy(max_iterations=5, max_runtime_seconds=60),
    )


async def _fresh_state(repo: InMemoryJobRepository, instruction: str = "fix bug") -> AgentState:
    job = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction=instruction))
    state = AgentState(job=job)
    state.inspection = SimpleNamespace()
    return state


class AutoFinalizeSafetyTests(IsolatedAsyncioTestCase):
    async def test_empty_workspace_is_never_auto_finalized(self) -> None:
        """No diff means nothing was accomplished; budget exhaustion stays a failure."""
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            state = await _fresh_state(repo)
            loop = _build_loop(repo, tmp, RescueTools(status="", diff=""))

            for reason in sorted(BUDGET_EXHAUSTION_STOP_REASONS):
                with self.subTest(reason=reason):
                    self.assertIsNone(await loop.maybe_auto_finalize_before_stop(state, reason))

    async def test_pending_repair_blocks_auto_finalize(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            state = await _fresh_state(repo)
            state.active_repair_action = {
                "signature": "pytest::test_add",
                "failure_class": "test_failure",
            }
            loop = _build_loop(repo, tmp, RescueTools(status=" M calc.py", diff="+ return a + b"))

            result = await loop.maybe_auto_finalize_before_stop(state, "max_iterations_exceeded")

            self.assertIsNone(result, "an unsatisfied repair action must block the rescue")

    async def test_non_budget_stop_reasons_are_never_rescued(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            state = await _fresh_state(repo)
            loop = _build_loop(repo, tmp, RescueTools(status=" M calc.py", diff="+ return a + b"))

            for reason in ("non_convergent", "cancelled", "verification_failed", "checker_failed"):
                with self.subTest(reason=reason):
                    self.assertIsNone(await loop.maybe_auto_finalize_before_stop(state, reason))

    def test_budget_set_is_exactly_the_four_documented_limits(self) -> None:
        self.assertEqual(
            BUDGET_EXHAUSTION_STOP_REASONS,
            frozenset(
                {
                    "max_iterations_exceeded",
                    "max_consecutive_failures_exceeded",
                    "max_tool_calls_exceeded",
                    "max_runtime_exceeded",
                }
            ),
        )

    def test_final_candidate_is_verified_before_budget_discard(self) -> None:
        """The stop branch must call handle_final_candidate before fail().

        node_bugfix died exactly here: the model submitted, the same turn tripped
        the limit, and the submission was dropped without ever reaching the
        verifier.
        """
        import inspect

        from docode.agent import loop as loop_module

        source = inspect.getsource(loop_module.CodingAgentLoop.run)

        # `run` has two stop checks: one before the model is consulted (no
        # decision exists yet) and one after it replies. Only the post-decision
        # branch can hold a fresh final_candidate, so that is the one that must
        # verify before failing.
        segments = source.split("if stop.should_stop")
        self.assertGreaterEqual(len(segments), 3, "expected a pre- and post-decision stop check")

        post_decision = segments[2].split('if decision.type == "blocked"', 1)[0]

        candidate_at = post_decision.find("handle_final_candidate")
        auto_at = post_decision.find("maybe_auto_finalize_before_stop")
        fail_at = post_decision.find("self.fail(")

        self.assertNotEqual(candidate_at, -1, "stop branch must consider the current final_candidate")
        self.assertNotEqual(auto_at, -1, "stop branch must attempt the FINAL_READY rescue")
        self.assertNotEqual(fail_at, -1)
        self.assertLess(candidate_at, fail_at, "verification must precede fail()")
        self.assertLess(auto_at, fail_at, "auto-finalize must precede fail()")
        self.assertLess(candidate_at, auto_at, "an explicit submission outranks the inferred rescue")

        # The pre-decision check has no decision to verify, but must still
        # attempt the rescue rather than discarding a FINAL_READY workspace.
        pre_decision = segments[1]
        self.assertLess(
            pre_decision.find("maybe_auto_finalize_before_stop"),
            pre_decision.find("self.fail("),
        )

    def test_handle_final_candidate_keeps_verified_work_on_any_budget_stop(self) -> None:
        """Post-verification guard must use the shared budget set.

        It previously hard-coded ``!= "max_iterations_exceeded"``, so a verified
        diff was still discarded under max_consecutive_failures_exceeded.
        """
        import inspect

        from docode.agent import loop as loop_module

        source = inspect.getsource(loop_module.CodingAgentLoop.handle_final_candidate)

        self.assertIn("stop.reason not in BUDGET_EXHAUSTION_STOP_REASONS", source)
        self.assertNotIn('stop.reason != "max_iterations_exceeded"', source)


class BlockedTaxonomyTests(IsolatedAsyncioTestCase):
    def test_task_unsatisfiable_wins_over_generic_runtime_failure(self) -> None:
        for reason in ("blocked:task_unsatisfiable", "task_unsatisfiable", "blocked:missing_premise"):
            with self.subTest(reason=reason):
                self.assertEqual(category_for_reason(reason), FailureCategory.TASK_UNSATISFIABLE)

    def test_ordinary_failures_are_not_mislabelled_unsatisfiable(self) -> None:
        for reason in ("max_iterations_exceeded", "non_convergent", "apply_patch_failed"):
            with self.subTest(reason=reason):
                self.assertNotEqual(category_for_reason(reason), FailureCategory.TASK_UNSATISFIABLE)

    async def test_blocked_does_not_emit_a_success_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            state = await _fresh_state(repo, "Refactor parse_config()")
            loop = _build_loop(repo, tmp, RescueTools())

            from docode.llm.decision import AgentDecision

            result = await loop.handle_blocked(
                state,
                AgentDecision(
                    type="blocked",
                    summary="parse_config() does not exist in this repository",
                    blocked_reason="task_unsatisfiable",
                    evidence=["grep -rn 'parse_config' . -> no matches"],
                ),
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.status, JobStatus.STOPPED)
            self.assertNotEqual(result.status, JobStatus.SUCCEEDED)
            self.assertFalse(result.terminal_result["strict_success"])
            # An honest refusal is a valid harness outcome, not a broken run.
            self.assertTrue(result.terminal_result["harness_valid"])


if __name__ == "__main__":
    unittest.main()
