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
import json
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
from docode.eval.models import RunResult, classify_run_outcome, derive_false_flags  # noqa: E402
from docode.eval.metrics import aggregate_results, suite_exit_code, write_suite_outputs  # noqa: E402
from docode.eval.evidence import build_summary, redact_endpoint  # noqa: E402
from docode.eval.runner import _build_coding_job, run_case  # noqa: E402
from docode.eval.checker import CheckerContext, FilesystemInspector, check, run_checker_module  # noqa: E402
from docode.storage.models import JobStatus  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
