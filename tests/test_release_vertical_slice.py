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
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from docode.config import DocodeConfig  # noqa: E402
from docode.runtime.smoke import (  # noqa: E402
    CommandProbe,
    SmokeCheck,
    ensure_dobox_smoke_token,
    managed_local_dobox,
)
from run_release_vertical_slice import (  # noqa: E402
    FIXTURE_ROOT,
    DoboxReadiness,
    LocalWorkspaceInspector,
    _build_coding_job,
    _json_default,
    build_job_record,
    build_live_manifest,
    build_outcomes_record,
    build_steps_record,
    build_summary,
    count_successful_live_runs,
    is_successful_live_run,
    plan_dobox_readiness,
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
    def test_missing_dobox_url_defaults_to_localhost(self):
        env = {
            "DOCODE_OPENAI_API_KEY": "sk-test",
            "DOCODE_PROVIDER": "openai",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config, creds, provider, model, reasons = resolve_provider_and_config()
        self.assertEqual(config.dobox_base_url, "http://localhost:3000")
        self.assertFalse(any("DOCODE_DOBOX_BASE_URL missing" in r for r in reasons), reasons)
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


class DoBoxReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_dobox_url_defaults_to_localhost(self):
        env = {"DOCODE_OPENAI_API_KEY": "sk-test", "DOCODE_PROVIDER": "openai"}
        with mock.patch.dict(os.environ, env, clear=True):
            config, _creds, _p, _m, reasons = resolve_provider_and_config()
        self.assertEqual(config.dobox_base_url, "http://localhost:3000")
        self.assertFalse(any("DOCODE_DOBOX_BASE_URL missing" in r for r in reasons), reasons)

    async def test_reachable_uses_existing_no_start(self):
        config = DocodeConfig()
        readiness = await plan_dobox_readiness(
            config,
            start_dobox=False,
            health_checker=lambda u: asyncio.sleep(0, result=(True, "ok")),
            command_runner=lambda c, d, t: CommandProbe(True, "ok"),
        )
        self.assertTrue(readiness.reachable)
        self.assertEqual(readiness.mode, "existing")
        self.assertFalse(readiness.started_by_runner)
        self.assertIsNone(readiness.fail_reason)

    async def test_unreachable_without_start_dobox_fails_closed(self):
        config = DocodeConfig()
        readiness = await plan_dobox_readiness(
            config,
            start_dobox=False,
            health_checker=lambda u: asyncio.sleep(0, result=(False, "down")),
            command_runner=lambda c, d, t: CommandProbe(True, "ok"),
        )
        self.assertFalse(readiness.reachable)
        self.assertIsNotNone(readiness.fail_reason)
        self.assertIn("--start-dobox", readiness.fail_reason)
        self.assertFalse(readiness.started_by_runner)

    async def test_unreachable_with_start_dobox_plans_autostart(self):
        config = DocodeConfig()
        readiness = await plan_dobox_readiness(
            config,
            start_dobox=True,
            health_checker=lambda u: asyncio.sleep(0, result=(False, "down")),
            command_runner=lambda c, d, t: CommandProbe(True, "ok"),
        )
        self.assertFalse(readiness.reachable)
        self.assertEqual(readiness.mode, "autostarted")
        self.assertTrue(readiness.started_by_runner)
        self.assertIsNone(readiness.fail_reason)


class ManagedLocalDoBoxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = DocodeConfig()
        self.config.dobox_backend_dir = Path("DoBoxDev/backend")

    async def test_reachable_does_not_start_or_stop(self):
        checks = [SmokeCheck("dobox_backend_dir", "passed", "ok"), SmokeCheck("docker_daemon", "passed", "ok")]
        with mock.patch("docode.runtime.smoke.start_dobox_process") as start, mock.patch(
            "docode.runtime.smoke.stop_process"
        ) as stop:
            async with managed_local_dobox(
                self.config, lambda u: asyncio.sleep(0, result=(True, "ok")), False, checks
            ) as sc:
                self.assertEqual(sc, [])
            start.assert_not_called()
            stop.assert_not_called()

    async def test_starts_and_stops_own_process(self):
        checks = [
            SmokeCheck("dobox_backend_dir", "passed", "ok"),
            SmokeCheck("docker_daemon", "passed", "ok"),
            SmokeCheck("dobox_sandbox_image", "passed", "ok"),
        ]

        class _Health:
            def __init__(self):
                self.calls = 0

            async def __call__(self, url):
                self.calls += 1
                # First call: endpoint down -> trigger autostart. Later calls: up.
                return (False, "down") if self.calls == 1 else (True, "ok")

        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None
        with mock.patch("docode.runtime.smoke.start_dobox_process", return_value=fake_proc) as start, mock.patch(
            "docode.runtime.smoke.stop_process"
        ) as stop:
            async with managed_local_dobox(
                self.config, _Health(), True, checks
            ) as sc:
                self.assertTrue(any(c.name == "dobox_autostart" and c.status == "passed" for c in sc))
            start.assert_called_once()
            stop.assert_called_once_with(fake_proc)

    async def test_docker_unavailable_structured_failure(self):
        checks = [SmokeCheck("dobox_backend_dir", "passed", "ok"), SmokeCheck("docker_daemon", "warning", "down")]
        with mock.patch("docode.runtime.smoke.start_dobox_process") as start:
            async with managed_local_dobox(
                self.config, lambda u: asyncio.sleep(0, result=(False, "down")), True, checks
            ) as sc:
                failed = [c for c in sc if c.name == "dobox_autostart" and c.status == "failed"]
                self.assertTrue(failed, sc)
            start.assert_not_called()

    async def test_backend_dir_missing_structured_failure(self):
        checks = [SmokeCheck("dobox_backend_dir", "warning", "no go.mod"), SmokeCheck("docker_daemon", "passed", "ok")]
        with mock.patch("docode.runtime.smoke.start_dobox_process") as start:
            async with managed_local_dobox(
                self.config, lambda u: asyncio.sleep(0, result=(False, "down")), True, checks
            ) as sc:
                failed = [c for c in sc if c.name == "dobox_autostart" and c.status == "failed"]
                self.assertTrue(failed, sc)
            start.assert_not_called()

    async def test_keep_dobox_skips_stop(self):
        checks = [
            SmokeCheck("dobox_backend_dir", "passed", "ok"),
            SmokeCheck("docker_daemon", "passed", "ok"),
            SmokeCheck("dobox_sandbox_image", "passed", "ok"),
        ]
        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None
        with mock.patch("docode.runtime.smoke.start_dobox_process", return_value=fake_proc), mock.patch(
            "docode.runtime.smoke.stop_process"
        ) as stop:
            async with managed_local_dobox(
                self.config, lambda u: asyncio.sleep(0, result=(True, "ok")), True, checks, keep=True
            ):
                pass
            stop.assert_not_called()


class DoBoxTokenTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_token_returned(self):
        config = DocodeConfig()
        config.dobox_token = "preconfigured-token"
        token, check = await ensure_dobox_smoke_token(config)
        self.assertEqual(token, "preconfigured-token")
        self.assertEqual(check.status, "passed")

    async def test_token_resolved_via_register(self):
        config = DocodeConfig()
        config.dobox_token = ""

        class _FakeResp:
            status_code = 200

            def json(self):
                return {"token": "resolved-token"}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, json=None):
                return _FakeResp()

        with mock.patch("httpx.AsyncClient", side_effect=_FakeClient):
            token, check = await ensure_dobox_smoke_token(config)
        self.assertEqual(token, "resolved-token")
        self.assertEqual(check.status, "passed")


class EvidenceDoBoxRuntimeTests(unittest.TestCase):
    def test_summary_includes_dobox_runtime_no_secrets(self):
        job = CodingJob(
            id=new_id("job"),
            user_id="u",
            instruction="x",
            provider="openai",
            model="gpt-5.4",
            dobox_project_id="proj-1",
            status=JobStatus.SUCCEEDED,
            artifact_id="art-1",
        )
        dobox_runtime = {
            "dobox_mode": "autostarted",
            "dobox_backend_dir": "../DoBoxDev/backend",
            "dobox_started_by_runner": True,
            "docker_daemon_available": True,
            "sandbox_image_available": True,
        }
        summary = build_summary(
            run_id=job.id,
            fixture="simple_bugfix",
            job=job,
            iterations=1,
            tool_calls=2,
            outcome_count=3,
            components={
                "runner": "JobRunnerService",
                "llm": "DecisionAdapter",
                "tools": "DoBoxTools",
                "repository": "InMemoryJobRepository",
                "exporter": "ArtifactExporter",
            },
            started_at="t0",
            finished_at="t1",
            dobox_runtime=dobox_runtime,
        )
        blob = json.dumps(summary)
        self.assertNotIn("resolved-token", blob)
        self.assertNotIn("sk-", blob)
        self.assertNotIn("preconfigured-token", blob)
        self.assertIn("dobox_runtime", summary)
        self.assertEqual(summary["dobox_runtime"]["dobox_backend_dir"], "../DoBoxDev/backend")
        self.assertEqual(summary["dobox_runtime"]["dobox_mode"], "autostarted")


class BuildCodingJobTests(unittest.TestCase):
    def test_minimal_docode_config_does_not_raise(self):
        # A minimal real DocodeConfig() must be usable to build a release
        # vertical-slice job without AttributeError on max_consecutive_failures.
        config = DocodeConfig()
        job = _build_coding_job(
            config,
            user_id="release-vertical-slice",
            instruction="fix the calculator bug",
            provider="openai",
            model="gpt-5.4-mini",
        )
        self.assertEqual(job.user_id, "release-vertical-slice")
        self.assertEqual(job.provider, "openai")
        self.assertEqual(job.model, "gpt-5.4-mini")
        self.assertEqual(job.artifact_mode, "patch")
        # model-defined default, not a runner hard-coded value.
        model_default = CodingJob.__dataclass_fields__["max_consecutive_failures"].default
        self.assertEqual(job.max_consecutive_failures, model_default)
        self.assertEqual(job.max_iterations, config.max_iterations)
        self.assertEqual(job.max_runtime_seconds, config.max_runtime_seconds)
        self.assertEqual(job.max_tool_calls, config.max_tool_calls)


class JsonDefaultSerializerTests(unittest.TestCase):
    def test_datetime_iso8601(self):
        dt = datetime(2026, 7, 15, 20, 37, 5)
        self.assertEqual(_json_default(dt), "2026-07-15T20:37:05")

    def test_enum_value(self):
        self.assertEqual(_json_default(JobStatus.SUCCEEDED), "succeeded")
        self.assertEqual(_json_default(JobStatus.FAILED), "failed")

    def test_path_string(self):
        p = Path("/tmp/workspace/calculator.py")
        self.assertIsInstance(_json_default(p), str)
        self.assertEqual(_json_default(p), str(p))

    def test_unknown_object_raises_type_error(self):
        class _Unserializable:
            pass

        with self.assertRaises(TypeError):
            _json_default(_Unserializable())

    def test_job_and_summary_bundle_writes_without_secrets(self):
        job = CodingJob(
            id=new_id("job"),
            user_id="u",
            instruction="fix it",
            provider="openai",
            model="gpt-5.4-mini",
            apicred_access_token="super-secret-token",
            dobox_project_id="proj-1",
            status=JobStatus.SUCCEEDED,
            artifact_id="art-1",
        )
        # build_job_record / build_summary redact sensitive fields; the
        # serializer must round-trip datetimes/enums without swallowing errors.
        record = build_job_record(job)
        serialized = json.dumps(record, default=_json_default)
        self.assertNotIn("super-secret-token", serialized)
        self.assertNotIn("sk-", serialized)
        summary = build_summary(
            run_id=job.id,
            fixture="simple_bugfix",
            job=job,
            iterations=3,
            tool_calls=5,
            outcome_count=8,
            components={"runner": "JobRunnerService", "llm": "DoCodeDecisionAdapter", "tools": "DoBoxTools", "repository": "InMemoryJobRepository", "exporter": "ArtifactExporter"},
            started_at=job.created_at.isoformat(),
            finished_at=job.completed_at.isoformat() if job.completed_at else job.updated_at.isoformat(),
        )
        summary_blob = json.dumps(summary, default=_json_default)
        self.assertIn("redacted", summary_blob)
        self.assertNotIn("sk-", summary_blob)


class LiveSuccessAccountingTests(unittest.TestCase):
    """The runner must count a live run as success only when the job reached
    the lowercase ``"succeeded"`` terminal state AND the hidden checker passed.

    These guard against the bug where ``passed`` was compared against the
    hardcoded uppercase ``"SUCCEEDED"`` while the stored status is lowercase,
    which silently failed every live run and made the runner exit non-zero
    even though the job succeeded.
    """

    def _run(self, *, status: str, checker_passed: bool) -> dict[str, Any]:
        return {
            "run_id": "job_account",
            "status": status,
            "checker": {"passed": checker_passed},
            "failure_reason": None,
        }

    def test_succeeded_with_passed_checker_counts(self):
        self.assertTrue(is_successful_live_run(self._run(status="succeeded", checker_passed=True)))

    def test_failed_status_not_counted(self):
        self.assertFalse(is_successful_live_run(self._run(status="failed", checker_passed=True)))

    def test_succeeded_with_failed_checker_not_counted(self):
        self.assertFalse(is_successful_live_run(self._run(status="succeeded", checker_passed=False)))

    def test_single_success_manifest_rate_is_1_of_1(self):
        runs = [self._run(status="succeeded", checker_passed=True)]
        manifest = build_live_manifest(
            fixture="simple_bugfix",
            provider="openai",
            model="gpt-5.4-mini",
            runs=runs,
            required_runs=1,
            output_dir=None,
        )
        self.assertEqual(manifest["success_rate"], "1/1")
        self.assertEqual(manifest["required"], "1/1")
        self.assertEqual(manifest["runs"][0]["status"], "succeeded")

    def test_single_success_exit_code_is_zero(self):
        runs = [self._run(status="succeeded", checker_passed=True)]
        exit_code = 0 if count_successful_live_runs(runs) >= 1 else 1
        self.assertEqual(exit_code, 0)

    def test_failed_run_does_not_produce_exit_zero(self):
        runs = [self._run(status="failed", checker_passed=True)]
        exit_code = 0 if count_successful_live_runs(runs) >= 1 else 1
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
