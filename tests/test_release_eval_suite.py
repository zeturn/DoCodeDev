"""Unit tests for the deterministic holdout evaluation harness (V1).

These tests are deterministic and network-free: they exercise the manifest
loader, the fixture validator, the outcome/classification logic, the metrics
aggregator, the exit-code policy, secret redaction, and the runner's
evidence-writing path (using a patched JobRunnerService so no real provider or
DoBox is contacted).

Run with:
    python -m unittest tests.test_release_eval_suite -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# Make the vertical-slice script importable (it provides shared infra).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from docode.eval.manifest import (  # noqa: E402
    FixtureManifestError,
    load_fixture_manifest,
    load_suite_manifests,
)
from docode.eval.fixture import load_fixture, validate_all_fixtures, validate_fixture  # noqa: E402
from docode.eval.models import RunResult, classify_run_outcome, derive_false_flags, extract_failure_signals  # noqa: E402
from docode.eval.metrics import aggregate_results, suite_exit_code, write_suite_outputs  # noqa: E402
from docode.eval.evidence import build_summary, redact_endpoint  # noqa: E402
from docode.eval.runner import _build_coding_job, run_case  # noqa: E402
from docode.eval.checker import CheckerContext, FilesystemInspector, check, run_checker_module  # noqa: E402
from docode.storage.models import JobStatus  # noqa: E402
from run_release_vertical_slice import FixtureSeedingDoBoxClient  # noqa: E402

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "release_eval"
SUCCESS_CASES = [
    "single_file_bugfix",
    "multi_file_bugfix",
    "parser_edge_cases",
    "small_feature",
    "node_bugfix",
    "go_bugfix",
    "anti_cheat",
]
UNSAFE_CASES = ["unsatisfiable_task"]


def _write_fixture(tmp: Path, **overrides: object) -> Path:
    data = {
        "schema_version": 1,
        "id": "x",
        "title": "t",
        "category": "bugfix",
        "difficulty": "easy",
        "language": "python",
        "workspace": "workspace",
        "instruction": "instruction.md",
        "checker": "checker.py",
        "required_commands": ["python -m unittest -q"],
        "expected_terminal": "succeeded",
        "network_mode": "no_internet",
        "tags": [],
    }
    data.update(overrides)
    (tmp / "fixture.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp


class ManifestValidationTests(unittest.TestCase):
    def test_loads_valid(self):
        manifest = load_fixture_manifest(FIXTURES_ROOT / "single_file_bugfix")
        self.assertEqual(manifest.id, "single_file_bugfix")
        self.assertEqual(manifest.schema_version, 1)
        self.assertEqual(manifest.required_commands, ("python -m unittest -q",))
        self.assertEqual(manifest.expected_terminal, "succeeded")

    def test_rejects_empty_required_commands(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), required_commands=[])
            with self.assertRaises(FixtureManifestError):
                load_fixture_manifest(Path(tmp))

    def test_rejects_invalid_expected_terminal(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), expected_terminal="bogus")
            with self.assertRaises(FixtureManifestError):
                load_fixture_manifest(Path(tmp))

    def test_rejects_path_traversal_workspace(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), workspace="../escape")
            with self.assertRaises(FixtureManifestError):
                load_fixture_manifest(Path(tmp))

    def test_rejects_absolute_path_workspace(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), workspace="/abs/workspace")
            with self.assertRaises(FixtureManifestError):
                load_fixture_manifest(Path(tmp))

    def test_rejects_checker_inside_workspace(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), workspace="workspace", checker="workspace/checker.py")
            with self.assertRaises(FixtureManifestError):
                load_fixture_manifest(Path(tmp))

    def test_rejects_duplicate_id(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                d = root / name
                d.mkdir()
                _write_fixture(d, id="dup")
            with self.assertRaises(FixtureManifestError):
                load_suite_manifests(root)

    def test_stable_error_message_mentions_field(self):
        with TemporaryDirectory() as tmp:
            _write_fixture(Path(tmp), required_commands=[])
            try:
                load_fixture_manifest(Path(tmp))
                self.fail("expected FixtureManifestError")
            except FixtureManifestError as exc:
                self.assertIn("required_commands", str(exc))


class CheckerGoldIsolationTests(unittest.TestCase):
    def test_workspace_excludes_checker_and_gold(self):
        fixture = load_fixture(FIXTURES_ROOT / "single_file_bugfix")
        names = {p.name for p in fixture.workspace_dir.iterdir()}
        self.assertNotIn("checker.py", names)
        self.assertNotIn("gold", names)
        # The hidden checker lives outside the agent workspace.
        self.assertFalse(fixture.checker_path.is_relative_to(fixture.workspace_dir))

    def test_checker_runs_via_filesystem_inspector(self):
        fixture = load_fixture(FIXTURES_ROOT / "single_file_bugfix")
        # Initial (buggy) state must fail the hidden checker.
        inspector = FilesystemInspector(fixture.workspace_dir)
        ctx = CheckerContext(
            inspector=inspector,
            fixture_root=fixture.workspace_dir,
            expected_terminal="succeeded",
            required_commands=fixture.manifest.required_commands,
        )
        initial = asyncio.run(run_checker_module(fixture.checker_path, ctx, safe=False))
        self.assertFalse(initial.passed)

        # Gold overlay must pass the hidden checker (no DoBox involved).
        import shutil
        from tempfile import TemporaryDirectory as TD

        with TD() as t:
            ws = Path(t) / "ws"
            shutil.copytree(fixture.workspace_dir, ws)
            shutil.copytree(fixture.gold_dir, ws, dirs_exist_ok=True)
            ginspector = FilesystemInspector(ws)
            gctx = CheckerContext(
                inspector=ginspector,
                fixture_root=fixture.workspace_dir,
                expected_terminal="succeeded",
                required_commands=fixture.manifest.required_commands,
            )
            gold = asyncio.run(run_checker_module(fixture.checker_path, gctx, safe=False))
            self.assertTrue(gold.passed)


class ValidatorStatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_success_fixtures_valid(self):
        for case in SUCCESS_CASES:
            fixture = load_fixture(FIXTURES_ROOT / case)
            report = await validate_fixture(fixture)
            self.assertTrue(report.ok, f"{case} invalid: {report.states}")
            self.assertTrue(report.states.get("initial_required_commands_failed"))
            self.assertTrue(report.states.get("gold_checker_passed"))
            self.assertTrue(report.states.get("cheat_naive_rejected"))

    async def test_unsatisfiable_valid(self):
        fixture = load_fixture(FIXTURES_ROOT / "unsatisfiable_task")
        report = await validate_fixture(fixture)
        self.assertTrue(report.ok, f"unsatisfiable invalid: {report.states}")
        self.assertTrue(report.states.get("initial_premise_absent"))
        self.assertTrue(report.states.get("expected_safe_failure_accepted"))
        self.assertTrue(report.states.get("cheat_fabricated_rejected"))

    async def test_validate_all(self):
        reports = await validate_all_fixtures(FIXTURES_ROOT)
        self.assertEqual(set(reports), set(SUCCESS_CASES) | set(UNSAFE_CASES))
        for case, report in reports.items():
            self.assertTrue(report.ok, f"{case}: {report.states}")


class ClassificationTests(unittest.TestCase):
    def test_passed(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="succeeded", checker_passed=True, expected_terminal="succeeded"),
            "passed",
        )

    def test_checker_failure_is_false_success(self):
        outcome = classify_run_outcome(terminal_status="succeeded", checker_passed=False, expected_terminal="succeeded")
        self.assertEqual(outcome, "checker_failure")
        ff, fs = derive_false_flags(outcome, terminal_status="succeeded", checker_passed=False)
        self.assertTrue(fs)
        self.assertFalse(ff)

    def test_agent_failure_is_false_failure(self):
        outcome = classify_run_outcome(terminal_status="failed", checker_passed=True, expected_terminal="succeeded")
        self.assertEqual(outcome, "agent_failure")
        ff, fs = derive_false_flags(outcome, terminal_status="failed", checker_passed=True)
        self.assertTrue(ff)
        self.assertFalse(fs)

    def test_expected_outcome_pass_safe_failure(self):
        outcome = classify_run_outcome(terminal_status="failed", checker_passed=True, expected_terminal="failed")
        self.assertEqual(outcome, "expected_outcome_pass")
        ff, fs = derive_false_flags(outcome, terminal_status="failed", checker_passed=True)
        self.assertFalse(ff)
        self.assertFalse(fs)

    def test_provider_failure(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", failure_class="model_unavailable"),
            "provider_failure",
        )

    def test_decision_parse_failure(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", failure_reason="unsupported decision type"),
            "decision_parse_failure",
        )

    def test_transport_failure(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", failure_class="transport_failed"),
            "dobox_transport_failure",
        )

    def test_budget_exceeded(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", failure_class="budget_exceeded"),
            "budget_exceeded",
        )

    def test_no_progress(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", failure_class="no_progress"),
            "no_progress",
        )

    def test_harness_failure(self):
        self.assertEqual(
            classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", harness_error=True),
            "harness_failure",
        )


class AggregateMetricsTests(unittest.TestCase):
    def _results(self, outcomes):
        return [
            RunResult(suite_run_id="s", case_id=f"case{i}", run_index=i, outcome=o, expected_terminal="succeeded")
            for i, o in enumerate(outcomes)
        ]

    def test_rates_and_counts(self):
        results = [
            RunResult(suite_run_id="s", case_id="c0", run_index=0, outcome="passed", checker_passed=True, expected_terminal="succeeded"),
            RunResult(suite_run_id="s", case_id="c1", run_index=0, outcome="expected_outcome_pass", checker_passed=True, expected_terminal="failed"),
            RunResult(suite_run_id="s", case_id="c2", run_index=0, outcome="agent_failure", checker_passed=True, false_failure=True, expected_terminal="succeeded"),
            RunResult(suite_run_id="s", case_id="c3", run_index=0, outcome="checker_failure", checker_passed=False, false_success=True, expected_terminal="succeeded"),
        ]
        summary = aggregate_results(results, suite_run_id="s")
        self.assertEqual(summary["total_runs"], 4)
        self.assertEqual(summary["strict_pass_rate"], 0.25)
        self.assertEqual(summary["expected_adjusted_pass_rate"], 0.5)
        self.assertEqual(summary["false_success_count"], 1)
        self.assertEqual(summary["false_failure_count"], 1)
        self.assertEqual(summary["agent_failure_count"], 1)
        self.assertEqual(summary["checker_failure_count"], 1)
        self.assertIn("c0", summary["cases"])

    def test_write_suite_outputs_preserves_all_runs(self):
        results = self._results(["passed", "agent_failure"])
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            paths = write_suite_outputs(out, manifests={}, run_results=results, suite_run_id="s")
            self.assertTrue(paths["results"].is_file())
            lines = paths["results"].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            for name in ("manifest.json", "summary.json", "report.md"):
                self.assertTrue((out / name).is_file())
            manifest_doc = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_doc["schema_version"], 1)
            self.assertIn("suite_run_id", manifest_doc)


class ExitCodeTests(unittest.TestCase):
    def _summary(self, outcomes):
        return aggregate_results(
            [RunResult(suite_run_id="s", case_id="c", run_index=0, outcome=o, expected_terminal="succeeded") for o in outcomes],
            suite_run_id="s",
        )

    def test_all_expected_zero(self):
        self.assertEqual(suite_exit_code(self._summary(["passed", "expected_outcome_pass"]), harness_valid=True, infra_fail_closed=False), 0)

    def test_one_agent_failure_one(self):
        self.assertEqual(suite_exit_code(self._summary(["passed", "agent_failure"]), harness_valid=True, infra_fail_closed=False), 1)

    def test_infra_fail_closed_two(self):
        self.assertEqual(suite_exit_code(self._summary(["infrastructure_failure"]), harness_valid=True, infra_fail_closed=True), 2)

    def test_harness_invalid_three(self):
        self.assertEqual(suite_exit_code(self._summary(["passed"]), harness_valid=False, infra_fail_closed=False), 3)


class SecretRedactionTests(unittest.TestCase):
    def test_redact_endpoint(self):
        self.assertEqual(redact_endpoint("https://key@host/secret"), "redacted")
        self.assertEqual(redact_endpoint(None), "redacted")

    def test_build_summary_redacts_endpoints(self):
        job = types.SimpleNamespace(
            id="j",
            status=JobStatus.SUCCEEDED,
            failure_reason=None,
            model="m",
            artifact_id=None,
            dobox_project_id=None,
        )
        summary = build_summary(
            run_id="r",
            fixture="f",
            job=job,
            iterations=1,
            tool_calls=1,
            outcome_count=1,
            components={},
            started_at="s",
            finished_at="f",
        )
        self.assertEqual(summary["provider"]["base_url"], "redacted")
        self.assertEqual(summary["dobox"]["endpoint"], "redacted")


class RunnerEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_case_writes_evidence_and_distinct_jobs(self):
        from docode.worker.runner import JobRunnerService

        fixture = load_fixture(FIXTURES_ROOT / "single_file_bugfix")

        class _Cfg:
            max_iterations = 10
            max_runtime_seconds = 300
            max_tool_calls = 50
            apicred_base_url = "http://apicred"
            apicred_token = "secret-token"
            apicred_mode = "local"

        async def _raise(*_a, **_k):
            raise RuntimeError("injected harness error")

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            results = []
            with mock.patch.object(JobRunnerService, "run_job", _raise):
                for i in range(2):
                    res = await run_case(
                        suite_run_id="s",
                        case_id="single_file_bugfix",
                        run_index=i,
                        config=_Cfg(),
                        local_credentials={},
                        provider="openai",
                        model="gpt-4",
                        fixture=fixture,
                        dobox=object(),
                        output_dir=out,
                    )
                    results.append(res)
            self.assertEqual(results[0].outcome, "harness_failure")
            self.assertNotEqual(results[0].job_id, results[1].job_id)
            self.assertTrue((out / results[0].job_id / "summary.json").is_file())
            self.assertTrue((out / results[1].job_id / "summary.json").is_file())


class VerticalSliceCompatTests(unittest.TestCase):
    def test_shared_infra_still_importable(self):
        # The vertical-slice script must keep exporting the names reused by the
        # eval-suite runner (CLI compatibility contract).
        from run_release_vertical_slice import (  # noqa: F401
            FixtureSeedingDoBoxClient,
            redact_endpoint,
            resolve_provider_and_config,
            write_evidence_bundle,
        )


class StructuredClassificationTests(unittest.TestCase):
    def _job(self, *, project_id=None, failure_reason=None, category=None, status="failed"):
        from docode.agent.failure_taxonomy import FailureCategory, TerminalResult

        terminal = None
        if category is not None:
            terminal = TerminalResult(status, FailureCategory(category), failure_reason or "")
        return types.SimpleNamespace(
            dobox_project_id=project_id,
            failure_reason=failure_reason,
            terminal_result=terminal,
            status=status,
        )

    def _classify(self, job, steps, *, checker_passed=None, expected_terminal="succeeded", harness_error=False):
        signals = extract_failure_signals(job, steps, harness_error=harness_error)
        return classify_run_outcome(
            terminal_status=job.status,
            checker_passed=checker_passed,
            expected_terminal=expected_terminal,
            failure_reason=job.failure_reason,
            signals=signals,
            harness_error=harness_error,
        )

    def test_dobox_project_creation_http_500_is_infra(self):
        job = self._job(
            project_id=None,
            failure_reason="POST /api/projects failed with HTTP 500: Failed to create project network: all predefined address pools have been fully subnetted",
            category="runtime_failure",
        )
        self.assertEqual(self._classify(job, []), "infrastructure_failure")

    def test_docker_address_pool_exhausted_is_infra(self):
        job = self._job(
            project_id=None,
            failure_reason="Failed to create project network: Error response from daemon: all predefined address pools have been fully subnetted",
            category="environment_failure",
        )
        self.assertEqual(self._classify(job, []), "infrastructure_failure")

    def test_provider_auth_before_workspace_is_provider_failure(self):
        job = self._job(
            project_id=None,
            failure_reason="apicred_authorize_failed: apicred unavailable",
            category="provider_failure",
        )
        self.assertEqual(self._classify(job, []), "provider_failure")

    def test_model_unavailable_is_provider_failure(self):
        job = self._job(
            project_id=None,
            failure_reason="The model gpt-5.4-mini is currently unavailable",
            category="provider_failure",
        )
        self.assertEqual(self._classify(job, []), "provider_failure")

    def test_unsupported_decision_type_is_decision_parse(self):
        steps = [{"kind": "tool", "content": {"type": "tool_result", "error": "unsupported decision type: frobnicate"}}]
        job = self._job(project_id="proj-1", failure_reason="unsupported decision type: frobnicate")
        self.assertEqual(self._classify(job, steps), "decision_parse_failure")

    def test_dobox_transport_error_during_tool_call(self):
        steps = [{"kind": "tool", "content": {"type": "transport_error", "error": "RemoteProtocolError: connection reset"}}]
        job = self._job(project_id="proj-1", failure_reason="DoBoxTransportError: RemoteProtocolError")
        self.assertEqual(self._classify(job, steps), "dobox_transport_failure")

    def test_workspace_established_model_decided_ordinary_failure_is_agent(self):
        steps = [
            {"kind": "llm", "content": {"type": "llm_decision"}},
            {"kind": "tool", "content": {"type": "tool_call", "tool": "run_command"}},
        ]
        job = self._job(project_id="proj-1", failure_reason="max_iterations_exceeded")
        self.assertEqual(self._classify(job, steps, checker_passed=False), "agent_failure")

    def test_max_consecutive_failures_after_tool_activity_is_agent(self):
        steps = [
            {"kind": "llm", "content": {"type": "llm_decision"}},
            {"kind": "tool", "content": {"type": "tool_call", "tool": "run_command"}},
            {"kind": "tool", "content": {"type": "tool_result", "error": "command failed"}},
        ]
        job = self._job(project_id="proj-1", failure_reason="max_consecutive_failures_exceeded")
        self.assertEqual(self._classify(job, steps, checker_passed=False), "agent_failure")

    def test_max_consecutive_failures_with_transport_signal_is_transport(self):
        steps = [
            {"kind": "llm", "content": {"type": "llm_decision"}},
            {"kind": "tool", "content": {"type": "tool_call", "tool": "run_command"}},
            {"kind": "tool", "content": {"type": "transport_error", "error": "ConnectError: timed out"}},
        ]
        job = self._job(project_id="proj-1", failure_reason="max_consecutive_failures_exceeded")
        self.assertEqual(self._classify(job, steps, checker_passed=False), "dobox_transport_failure")

    def test_job_succeeded_checker_failed_is_false_success(self):
        job = self._job(project_id="proj-1", status="succeeded")
        outcome = self._classify(job, [{"kind": "llm", "content": {"type": "llm_decision"}}], checker_passed=False)
        self.assertEqual(outcome, "checker_failure")
        ff, fs = derive_false_flags(outcome, terminal_status="succeeded", checker_passed=False)
        self.assertTrue(fs)

    def test_unsatisfiable_safe_failure_with_checker_pass(self):
        job = self._job(project_id="proj-1", status="failed", category="runtime_failure")
        outcome = self._classify(job, [], checker_passed=True, expected_terminal="failed")
        self.assertEqual(outcome, "expected_outcome_pass")

    def test_malformed_terminal_result_does_not_crash(self):
        job = types.SimpleNamespace(
            dobox_project_id=None,
            failure_reason="boom",
            terminal_result={"category": "not_a_real_enum"},
            status="failed",
        )
        signals = extract_failure_signals(job, [])
        self.assertFalse(signals.workspace_created)
        outcome = classify_run_outcome(
            terminal_status="failed", checker_passed=False, expected_terminal="succeeded", signals=signals
        )
        self.assertEqual(outcome, "infrastructure_failure")

    def test_terminal_result_none_does_not_crash(self):
        job = types.SimpleNamespace(dobox_project_id=None, failure_reason="boom", terminal_result=None, status="failed")
        signals = extract_failure_signals(job, [])
        self.assertIsNone(signals.failure_category)
        outcome = classify_run_outcome(terminal_status="failed", checker_passed=False, expected_terminal="succeeded", signals=signals)
        self.assertEqual(outcome, "infrastructure_failure")

    def test_structured_signals_win_over_conflicting_text(self):
        # Free text says decision parse, but structured signals show a
        # pre-workspace provisioning failure -> infrastructure must win.
        job = self._job(
            project_id=None,
            failure_reason="unsupported decision type: but project never created",
            category="runtime_failure",
        )
        outcome = self._classify(job, [])
        self.assertEqual(outcome, "infrastructure_failure")

    def test_invalid_baseline_regression_not_agent_failure(self):
        # De-sensitized minimal reproduction of the original INVALID baseline:
        # workspace provisioning failures must NOT be counted as agent_failure.
        job = self._job(
            project_id=None,
            failure_reason="POST /api/projects failed with HTTP 500: Failed to create project network: all predefined address pools have been fully subnetted",
            category="runtime_failure",
        )
        outcome = self._classify(job, [])
        self.assertNotEqual(outcome, "agent_failure")
        self.assertEqual(outcome, "infrastructure_failure")


class AggregateInfraNotAgentTests(unittest.TestCase):
    def test_infra_failure_not_counted_as_agent(self):
        results = [
            RunResult(suite_run_id="s", case_id="c0", run_index=0, outcome="infrastructure_failure", expected_terminal="succeeded"),
            RunResult(suite_run_id="s", case_id="c1", run_index=0, outcome="agent_failure", expected_terminal="succeeded"),
        ]
        summary = aggregate_results(results, suite_run_id="s")
        self.assertEqual(summary["agent_failure_count"], 1)
        self.assertEqual(summary["infrastructure_failure_count"], 1)
        self.assertEqual(summary["by_outcome"].get("infrastructure_failure"), 1)
        self.assertEqual(summary["by_outcome"].get("agent_failure"), 1)


class _RecordingSeedingClient(FixtureSeedingDoBoxClient):
    """In-memory stand-in for the real DoBox seeding client.

    It records every ``write_file`` payload (path + base64 content) and every
    ``run_command`` without contacting a network or a Provider. This lets the
    binary-safe seeding logic be exercised deterministically and network-free.
    """

    def __init__(self, fixture_root: Path) -> None:
        super().__init__("http://dobox.example", "token", fixture_root)
        self.writes: list[tuple[str, str | None]] = []
        self.commands: list[object] = []

    async def create_project(self, *, name: str, repo_url: str | None = None, branch: str | None = None, image: str | None = None, network_mode: str | None = None):
        # Mirror production: seed immediately after project creation.
        await self._seed_fixture("proj-seeded")
        return types.SimpleNamespace(project_id="proj-seeded", sandbox_id="sb", raw={})

    async def write_file(self, project_id: str, path: str, content: str | None = None, *, content_base64: str | None = None, agent_session_id: str | None = None) -> None:
        self.writes.append((path, content_base64))

    async def run_command(self, project_id: str, command, cwd: str = "/workspace", timeout_sec: int = 30, output_limit: int = 200_000, agent_session_id: str | None = None):
        self.commands.append(command)
        return types.SimpleNamespace(output="ok", exit_code=0, truncated=False)


class BinarySafeSeedingTests(unittest.IsolatedAsyncioTestCase):
    async def _seed(self, fixture_root: Path) -> _RecordingSeedingClient:
        client = _RecordingSeedingClient(fixture_root)
        await client._seed_fixture("proj-seeded")
        return client

    async def test_utf8_text_file_seeded_as_base64(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("héllo ☃\n", encoding="utf-8", newline="")
            client = await self._seed(root)
            self.assertEqual(len(client.writes), 1)
            rel, b64 = client.writes[0]
            self.assertEqual(rel, "a.txt")
            self.assertEqual(base64.b64decode(b64), "héllo ☃\n".encode("utf-8"))

    async def test_non_utf8_bytes_seeded_byte_for_byte(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = bytes([0x00, 0xFF, 0xFE, 0x80, 0x41, 0xE3, 0x00])
            (root / "bin.dat").write_bytes(data)
            client = await self._seed(root)
            self.assertEqual(len(client.writes), 1)
            rel, b64 = client.writes[0]
            self.assertEqual(rel, "bin.dat")
            self.assertEqual(base64.b64decode(b64), data)

    async def test_empty_file_seeded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty.txt").write_bytes(b"")
            client = await self._seed(root)
            self.assertEqual(len(client.writes), 1)
            rel, b64 = client.writes[0]
            self.assertEqual(rel, "empty.txt")
            self.assertEqual(base64.b64decode(b64), b"")

    async def test_nested_path_seeded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub" / "dir").mkdir(parents=True)
            (root / "sub" / "dir" / "deep.py").write_text("x=1\n", encoding="utf-8")
            (root / "top.txt").write_text("y", encoding="utf-8")
            client = await self._seed(root)
            rels = {rel for rel, _ in client.writes}
            self.assertIn("sub/dir/deep.py", rels)
            self.assertIn("top.txt", rels)

    async def test_symlink_not_treated_as_regular_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "secret.txt"
            target.write_text("TOP SECRET", encoding="utf-8")
            try:
                (root / "link.txt").symlink_to(target)
            except OSError as exc:  # symlink privilege may be unavailable (e.g. Windows)
                self.skipTest(f"symlink unavailable: {exc}")
            client = await self._seed(root)
            rels = {rel for rel, _ in client.writes}
            self.assertNotIn("link.txt", rels)
            self.assertIn("secret.txt", rels)

    async def test_checker_and_gold_outside_workspace_not_seeded(self):
        # The harness seeds only the workspace dir, never the fixture root's
        # checker.py / gold / naive (those must stay hidden from the agent).
        with TemporaryDirectory() as tmp:
            fx = Path(tmp) / "fixture"
            ws = fx / "workspace"
            ws.mkdir(parents=True)
            (ws / "app.py").write_text("print(1)\n", encoding="utf-8")
            (fx / "checker.py").write_text("...", encoding="utf-8")
            (fx / "gold").mkdir()
            (fx / "gold" / "app.py").write_text("print(2)\n", encoding="utf-8")
            client = _RecordingSeedingClient(ws)
            await client._seed_fixture("proj-seeded")
            rels = {rel for rel, _ in client.writes}
            self.assertEqual(rels, {"app.py"})

    async def test_no_path_traversal_entries_seeded(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            client = await self._seed(root)
            for rel, _ in client.writes:
                self.assertNotIn("..", rel.split("/"))

    async def test_no_llm_provisioning_seeds_binary_without_provider(self):
        # A fixture containing a non-UTF-8 file is seeded to a recording DoBox
        # interface; the uploaded bytes are the raw base64 of the file and no
        # Provider is contacted (the seeding client is pure DoBox I/O).
        with TemporaryDirectory() as tmp:
            fx = Path(tmp) / "fixture"
            ws = fx / "workspace"
            ws.mkdir(parents=True)
            data = bytes([0x89, 0x50, 0x4E, 0x47, 0xFF, 0x00])  # PNG-like header + invalid byte
            (ws / "image.bin").write_bytes(data)
            (ws / "main.py").write_text("x=1\n", encoding="utf-8", newline="")
            client = _RecordingSeedingClient(ws)
            # create_project triggers seeding; no Provider credential is touched.
            project = await client.create_project(name="docode-provision")
            self.assertEqual(project.project_id, "proj-seeded")
            by_rel = {rel: b64 for rel, b64 in client.writes}
            self.assertIn("image.bin", by_rel)
            self.assertEqual(base64.b64decode(by_rel["image.bin"]), data)
            self.assertEqual(base64.b64decode(by_rel["main.py"]), b"x=1\n")
            self.assertTrue(client.commands)  # git init/commit ran


if __name__ == "__main__":
    unittest.main()
