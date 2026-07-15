"""Non-live unit tests for the release vertical-slice runner.

These tests verify the *harness logic* without a live DoBox or provider:

- configuration parsing and fail-closed behavior
- secret redaction
- evidence-bundle serialization (no secrets leaked)
- hidden-checker behavior against a local filesystem workspace
- required-command-after-edit ordering logic

They do NOT claim the live vertical slice passed. The live run is performed
separately by ``scripts/run_release_vertical_slice.py`` against a real DoBox.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_release_vertical_slice import (  # noqa: E402
    FIXTURE_ROOT,
    LocalWorkspaceInspector,
    build_job_record,
    build_outcomes_record,
    build_steps_record,
    build_summary,
    redact_endpoint,
    resolve_provider_and_config,
    run_hidden_checker,
)

from docode.storage.models import CodingJob, JobStatus, new_id  # noqa: E402
from docode.storage.repository import InMemoryJobRepository  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_workspace(tmp: Path, *, fixed: bool, weaken_tests: bool = False) -> Path:
    ws = tmp / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    calc = "def add(a, b):\n    return a + b\n" if fixed else "def add(a, b):\n    return a - b\n"
    _write(ws / "calculator.py", calc)
    test_text = (
        "import unittest\nfrom calculator import add\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
        "        self.assertEqual(add(-2, 2), 0)\n"
        "if __name__ == '__main__':\n    unittest.main()\n"
    )
    if weaken_tests:
        test_text = "import unittest\nclass T(unittest.TestCase):\n    pass\n"
    _write(ws / "tests" / "__init__.py", "")
    _write(ws / "tests" / "test_calculator.py", test_text)
    return ws


def _make_fixture_root(tmp: Path) -> Path:
    fx = tmp / "fixture"
    _write(fx / "calculator.py", "def add(a, b):\n    return a - b\n")
    _write(
        fx / "tests" / "test_calculator.py",
        "import unittest\nfrom calculator import add\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n"
        "        self.assertEqual(add(-2, 2), 0)\n"
        "if __name__ == '__main__':\n    unittest.main()\n",
    )
    return fx


class SimpleStep:
    """Minimal step double for unit-testing checker ordering (no live runtime)."""

    def __init__(self, step_index: int, kind: str, content: dict) -> None:
        self.step_index = step_index
        self.kind = kind
        self.content = content


class ConfigFailClosedTests(unittest.TestCase):
    def test_missing_dobox_url_is_detected(self):
        env = {
            "DOCODE_DOBOX_BASE_URL": "",
            "DOCODE_OPENAI_API_KEY": "sk-test",
            "DOCODE_PROVIDER": "openai",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            _ = os.environ.pop("DOCODE_DOBOX_BASE_URL", None)
            os.environ["DOCODE_DOBOX_BASE_URL"] = ""
            config, creds, provider, model, reasons = resolve_provider_and_config()
        self.assertIn("DOCODE_DOBOX_BASE_URL missing", reasons[0])
        self.assertTrue(creds)  # provider key still resolved

    def test_missing_provider_key_is_detected(self):
        env_keys = [
            "DOCODE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
            "DOCODE_PROVIDER_API_KEY",
            "DOCODE_DEEPSEEK_API_KEY",
            "DEEPSEEK_API_KEY",
        ]
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in env_keys:
                os.environ.pop(key, None)
            os.environ["DOCODE_PROVIDER"] = "openai"
            os.environ["DOCODE_DOBOX_BASE_URL"] = "http://localhost:3000"
            config, creds, provider, model, reasons = resolve_provider_and_config()
        self.assertTrue(any("openai provider API key missing" in r for r in reasons), reasons)
        self.assertEqual(creds, {})

    def test_redact_endpoint_never_leaks(self):
        self.assertEqual(redact_endpoint("http://localhost:3000"), "redacted")
        self.assertEqual(redact_endpoint(None), "redacted")
        self.assertEqual(redact_endpoint("https://api.openai.com/v1"), "redacted")


class EvidenceSerializationTests(unittest.TestCase):
    def test_job_record_strips_sensitive_token(self):
        job = CodingJob(
            id=new_id("job"),
            user_id="u",
            instruction="fix it",
            apicred_access_token="super-secret-token",
            dobox_project_id="proj-1",
            status=JobStatus.SUCCEEDED,
            artifact_id="art-1",
        )
        record = build_job_record(job)
        self.assertNotIn("apicred_access_token", record)
        self.assertEqual(record["dobox_project_id"], "proj-1")
        self.assertEqual(record["status"], "succeeded")

    def test_summary_never_contains_endpoint(self):
        job = CodingJob(
            id=new_id("job"),
            user_id="u",
            instruction="fix it",
            provider="openai",
            model="gpt-5.4",
            dobox_project_id="proj-1",
            status=JobStatus.SUCCEEDED,
            artifact_id="art-1",
        )
        summary = build_summary(
            run_id=job.id,
            fixture="simple_bugfix",
            job=job,
            iterations=3,
            tool_calls=5,
            outcome_count=8,
            components={"runner": "JobRunnerService", "llm": "DoCodeDecisionAdapter", "tools": "DoBoxTools", "repository": "InMemoryJobRepository", "exporter": "ArtifactExporter"},
            started_at="t0",
            finished_at="t1",
        )
        self.assertEqual(summary["provider"]["base_url"], "redacted")
        self.assertEqual(summary["dobox"]["endpoint"], "redacted")
        self.assertEqual(summary["status"], "succeeded")

    def test_steps_and_outcomes_serialization(self):
        repo: InMemoryJobRepository = InMemoryJobRepository()
        job = asyncio.run(
            repo.create_job(
                CodingJob(id=new_id("job"), user_id="u", instruction="x", status=JobStatus.RUNNING)
            )
        )
        asyncio.run(repo.add_step(job.id, "llm", {"type": "llm_decision", "decision_type": "continue"}))
        asyncio.run(repo.add_step(job.id, "outcome", {"type": "step_outcome", "kind": "tool"}))
        steps = asyncio.run(repo.list_steps(job.id))
        steps_json = build_steps_record(steps)
        self.assertEqual(len(steps_json), 2)
        self.assertNotIn("apicred", str(steps_json))
        outcomes_json = build_outcomes_record(steps)
        self.assertEqual(len(outcomes_json), 1)


class HiddenCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def _run_checker(self, *, fixed: bool, weaken_tests: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ws = _make_workspace(tmp_path, fixed=fixed, weaken_tests=weaken_tests)
            fx = _make_fixture_root(tmp_path)
            job = CodingJob(
                id=new_id("job"),
                user_id="u",
                instruction="fix it",
                status=JobStatus.SUCCEEDED,
                artifact_id="art-1",
            )
            inspector = LocalWorkspaceInspector(ws)
            return await run_hidden_checker(inspector, fx, job, steps=[])

    async def test_passing_workspace_passes_all_checks(self):
        result = await self._run_checker(fixed=True)
        self.assertTrue(result["passed"], result["failures"])
        for name, value in result["checks"].items():
            self.assertTrue(value, f"{name} failed: {result['failures']}")

    async def test_unfixed_workspace_fails_functional(self):
        result = await self._run_checker(fixed=False)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["functional_behavior"])
        self.assertFalse(result["checks"]["required_command_passed"])
        self.assertFalse(result["checks"]["implementation_modified"])

    async def test_weakened_tests_fail(self):
        result = await self._run_checker(fixed=True, weaken_tests=True)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["tests_not_weakened"])

    async def test_terminal_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ws = _make_workspace(tmp_path, fixed=True)
            fx = _make_fixture_root(tmp_path)
            job = CodingJob(
                id=new_id("job"),
                user_id="u",
                instruction="fix it",
                status=JobStatus.FAILED,
                artifact_id=None,
            )
            inspector = LocalWorkspaceInspector(ws)
            result = await run_hidden_checker(inspector, fx, job, steps=[])
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["terminal_success"])
            self.assertFalse(result["checks"]["artifact_present"])

    async def test_required_command_after_edit_ordering(self):
        # edit at index 2, required command passing at index 5 -> fresh.
        good_steps = [
            SimpleStep(1, "tool", {"type": "tool_result", "tool": "read_file", "exit_code": 0}),
            SimpleStep(2, "tool", {"type": "tool_result", "tool": "edit_file", "exit_code": 0, "metadata": {"path": "calculator.py"}}),
            SimpleStep(5, "tool", {"type": "tool_result", "tool": "run_command", "exit_code": 0, "metadata": {"command": "python -m unittest -q"}}),
        ]
        # required command BEFORE the edit -> not fresh.
        bad_steps = [
            SimpleStep(1, "tool", {"type": "tool_result", "tool": "run_command", "exit_code": 0, "metadata": {"command": "python -m unittest -q"}}),
            SimpleStep(4, "tool", {"type": "tool_result", "tool": "edit_file", "exit_code": 0, "metadata": {"path": "calculator.py"}}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ws = _make_workspace(tmp_path, fixed=True)
            fx = _make_fixture_root(tmp_path)
            job = CodingJob(id=new_id("job"), user_id="u", instruction="x", status=JobStatus.SUCCEEDED, artifact_id="a")
            inspector = LocalWorkspaceInspector(ws)
            good = await run_hidden_checker(inspector, fx, job, good_steps)
            bad = await run_hidden_checker(inspector, fx, job, bad_steps)
        self.assertTrue(good["checks"]["fresh_after_edit"], good["failures"])
        self.assertFalse(bad["checks"]["fresh_after_edit"], bad["failures"])


if __name__ == "__main__":
    unittest.main()
