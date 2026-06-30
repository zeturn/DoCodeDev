from __future__ import annotations

import json
import io
import shutil
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from docode.cli import run_eval_assert_command, run_eval_command, run_eval_scaffold_command
from docode.eval import (
    EvalThresholds,
    eval_case_result_from_job,
    manifest_with_served_local_repos,
    run_eval,
    scaffold_eval_suite,
    summarize_eval_matrix,
    with_eval_comparison,
)
from docode.storage.models import JobStatus, CodingJob, new_id


class EvalTests(TestCase):
    def test_run_eval_aggregates_case_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp)
            (fixtures / "python-bugfix.json").write_text(
                json.dumps(
                    {
                        "name": "python-bugfix",
                        "status": "succeeded",
                        "iterations": 3,
                        "tool_calls": 8,
                        "usage": {"total_tokens": 1200, "cost": 0.04},
                    }
                ),
                encoding="utf-8",
            )
            (fixtures / "crawler.json").write_text(
                json.dumps(
                    {
                        "name": "crawler",
                        "status": "failed",
                        "iterations": 5,
                        "tool_calls": 14,
                        "tokens": 2000,
                        "cost": 0.08,
                        "failure_reason": "external source blocked",
                        "verification": {"required_fixes": ["verify data source"]},
                    }
                ),
                encoding="utf-8",
            )

            report = run_eval(fixtures)

            self.assertEqual(report.total, 2)
            self.assertEqual(report.succeeded, 1)
            self.assertEqual(report.failed, 1)
            self.assertEqual(report.success_rate, 0.5)
            self.assertEqual(report.iterations, 8)
            self.assertEqual(report.tool_calls, 22)
            self.assertEqual(report.tokens, 3200)
            self.assertAlmostEqual(report.cost, 0.12)
            self.assertEqual(report.failure_reasons, {"external source blocked": 1})
            self.assertEqual(report.verification_plan_failures, {"verify data source": 1})
            self.assertEqual(report.failure_classes, {})

    def test_eval_run_command_writes_report(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp) / "fixtures"
            fixtures.mkdir()
            report_path = Path(tmp) / "report.json"
            (fixtures / "readme.json").write_text('{"status":"succeeded","iterations":1}', encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                run_eval_command(
                    Namespace(
                        fixtures_dir=str(fixtures),
                        report=str(report_path),
                        min_success_rate=None,
                        max_average_tool_calls=None,
                        max_total_cost=None,
                    )
                )

            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["succeeded"], 1)

    def test_run_eval_records_threshold_assertion(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp)
            (fixtures / "ok.json").write_text('{"status":"succeeded","tool_calls":10,"cost":0.20}', encoding="utf-8")
            (fixtures / "bad.json").write_text('{"status":"failed","tool_calls":20,"cost":0.30}', encoding="utf-8")

            report = run_eval(fixtures, thresholds=EvalThresholds(min_success_rate=0.8, max_average_tool_calls=12, max_total_cost=1.0))

            data = report.to_dict()
            self.assertTrue(data["regression"])
            self.assertEqual(data["thresholds"]["min_success_rate"], 0.8)
            self.assertEqual(len(data["threshold_failures"]), 2)

    def test_eval_report_can_compare_against_previous_run(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_dir = root / "previous"
            current_dir = root / "current"
            previous_dir.mkdir()
            current_dir.mkdir()
            (previous_dir / "python-bugfix.json").write_text('{"status":"failed"}', encoding="utf-8")
            (previous_dir / "python-cli.json").write_text('{"status":"succeeded","iterations":4}', encoding="utf-8")
            (current_dir / "python-bugfix.json").write_text('{"status":"succeeded","iterations":2}', encoding="utf-8")
            (current_dir / "python-cli.json").write_text('{"status":"failed","iterations":6}', encoding="utf-8")

            report = with_eval_comparison(run_eval(current_dir), run_eval(previous_dir))

            self.assertIsNotNone(report.comparison)
            assert report.comparison is not None
            self.assertEqual(report.comparison.previous_succeeded, 1)
            self.assertEqual(report.comparison.succeeded_delta, 0)
            self.assertEqual(report.comparison.newly_succeeded, ["python-bugfix"])
            self.assertEqual(report.comparison.newly_failed, ["python-cli"])
            self.assertIn("comparison", report.to_dict())

    def test_eval_matrix_summarizes_models_and_main_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fast_dir = root / "fast"
            strong_dir = root / "strong"
            fast_dir.mkdir()
            strong_dir.mkdir()
            (fast_dir / "ok.json").write_text('{"status":"succeeded","iterations":2,"tool_calls":4,"tokens":100}', encoding="utf-8")
            (fast_dir / "bad.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "iterations": 4,
                        "tool_calls": 8,
                        "tokens": 300,
                        "failure_class": "verifier_failed",
                        "failure_category": "verifier_plan_failed",
                    }
                ),
                encoding="utf-8",
            )
            (strong_dir / "ok.json").write_text('{"status":"succeeded","iterations":1,"tool_calls":2,"tokens":80}', encoding="utf-8")

            matrix = summarize_eval_matrix({"fast": run_eval(fast_dir), "strong": run_eval(strong_dir)})
            data = matrix.to_dict()

            fast = next(model for model in data["models"] if model["model"] == "fast")
            self.assertEqual(fast["success_rate"], 0.5)
            self.assertEqual(fast["avg_iterations"], 3.0)
            self.assertEqual(fast["avg_tool_calls"], 6.0)
            self.assertEqual(fast["avg_tokens"], 200.0)
            self.assertEqual(fast["main_failure"], "verifier_plan_failed")
            self.assertEqual(data["best_success_rate"], 1.0)

    def test_eval_run_command_exits_nonzero_when_threshold_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp) / "fixtures"
            fixtures.mkdir()
            report_path = Path(tmp) / "report.json"
            (fixtures / "bad.json").write_text('{"status":"failed"}', encoding="utf-8")

            with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
                run_eval_command(
                    Namespace(
                        fixtures_dir=str(fixtures),
                        report=str(report_path),
                        min_success_rate=0.8,
                        max_average_tool_calls=None,
                        max_total_cost=None,
                    )
                )

            self.assertEqual(raised.exception.code, 1)
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(data["regression"])

    def test_eval_assert_command_updates_existing_report(self) -> None:
        with TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "total": 2,
                        "succeeded": 1,
                        "failed": 1,
                        "success_rate": 0.5,
                        "iterations": 2,
                        "tool_calls": 8,
                        "tokens": 100,
                        "cost": 0.1,
                        "failure_reasons": {},
                        "verification_plan_failures": {},
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
                run_eval_assert_command(
                    Namespace(
                        report=str(report_path),
                        min_success_rate=0.8,
                        max_average_tool_calls=10,
                        max_total_cost=1.0,
                    )
                )

            self.assertEqual(raised.exception.code, 1)
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(data["regression"])
            self.assertIn("min_success_rate", data["thresholds"])

    def test_scaffold_eval_suite_creates_ten_small_repos_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"

            manifest = scaffold_eval_suite(root)

            self.assertEqual(len(manifest["cases"]), 10)
            names = {case["name"] for case in manifest["cases"]}
            self.assertEqual(
                names,
                {
                    "python-bugfix",
                    "python-cli",
                    "crawler",
                    "api-adapter",
                    "readme-only",
                    "js-bugfix",
                    "no-test-project",
                    "bad-web-source-repair",
                    "large-command-output",
                    "github-pr-artifact-export",
                },
            )
            crawler_repo = root / "repos" / "crawler"
            self.assertTrue((crawler_repo / "crawler.py").exists())
            self.assertTrue((root / "manifest.json").exists())
            bugfix = next(case for case in manifest["cases"] if case["name"] == "python-bugfix")
            self.assertEqual(bugfix["hints"]["target_files"], ["calculator.py"])
            self.assertIn("retry_count(3)", bugfix["hints"]["expected_behavior"])
            self.assertEqual(bugfix["hints"]["suggested_commands"], ["python3 -m unittest discover -s tests"])
            if shutil.which("git") is not None:
                self.assertTrue((crawler_repo / ".git").exists())
                self.assertTrue(all(case["git_initialized"] for case in manifest["cases"]))

    def test_eval_scaffold_command_writes_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "suite"

            with redirect_stdout(io.StringIO()):
                run_eval_scaffold_command(Namespace(output_dir=str(root), force=False))

            data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["cases"]), 10)

    def test_eval_case_result_from_job_extracts_metrics_from_steps(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="fix", status=JobStatus.SUCCEEDED, artifact_id="artifact-1")
        steps = [
            {"type": "llm_decision", "usage": {"total_tokens": 10, "cost": 0.01}},
            {"type": "tool_call", "tool": "run_tests"},
            {
                "passed": True,
                "reason": "ok",
                "required_fixes": [],
                "verification_plan": {"required_commands": ["related_test"]},
            },
        ]

        result = eval_case_result_from_job({"name": "python-bugfix", "instruction": "fix"}, job, steps)

        self.assertTrue(result["success"])
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["tokens"], 10)
        self.assertEqual(result["cost"], 0.01)
        self.assertEqual(result["artifact_id"], "artifact-1")
        self.assertEqual(result["verification"]["passed"], True)

    def test_eval_case_result_classifies_workspace_probe_failure(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="infrastructure_failed: workspace_inconsistent",
        )
        steps = [
            {
                "type": "workspace_probe",
                "passed": False,
                "category": "workspace_inconsistent",
                "diagnostics": {"file_api_exec_probe": {"exit_code": 1}},
            }
        ]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_class"], "infra_failed")
        self.assertEqual(result["failure_category"], "workspace_inconsistent")
        self.assertIn("workspace_probe", result["infra_diagnostics"])

    def test_run_eval_aggregates_failure_classes_and_categories(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp)
            (fixtures / "budget.json").write_text(
                json.dumps(
                    {
                        "name": "budget",
                        "status": "failed",
                        "failure_reason": "max_llm_tokens_exceeded",
                        "failure_class": "budget_exceeded",
                        "failure_category": "max_llm_tokens_exceeded",
                    }
                ),
                encoding="utf-8",
            )
            (fixtures / "infra.json").write_text(
                json.dumps(
                    {
                        "name": "infra",
                        "status": "failed",
                        "failure_reason": "infrastructure_failed: workspace_inconsistent",
                        "failure_class": "infra_failed",
                        "failure_category": "workspace_inconsistent",
                    }
                ),
                encoding="utf-8",
            )

            report = run_eval(fixtures)

            self.assertEqual(report.failure_classes, {"budget_exceeded": 1, "infra_failed": 1})
            self.assertEqual(report.failure_categories, {"max_llm_tokens_exceeded": 1, "workspace_inconsistent": 1})

    def test_eval_case_result_classifies_parser_failures_under_agent_failed(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="max_consecutive_failures_exceeded",
        )
        steps = [{"type": "llm_error", "detail": "unsupported decision type: run_command"}]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertEqual(result["failure_class"], "agent_failed")
        self.assertEqual(result["failure_category"], "parser_failed")

    def test_eval_case_result_classifies_llm_auth_failures_as_provider_auth(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="llm_auth_failed",
        )
        steps = [{"type": "llm_error", "reason": "llm_auth_failed", "detail": "401 Unauthorized"}]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertEqual(result["failure_class"], "model_unavailable")
        self.assertEqual(result["failure_category"], "provider_auth_failed")

    def test_eval_case_result_classifies_provider_5xx_as_model_unavailable(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="max_consecutive_failures_exceeded",
        )
        steps = [
            {
                "type": "llm_error",
                "reason": "llm_decision_failed",
                "detail": "Server error '503 Service Unavailable' for url 'http://localhost:8103/v1/chat/completions'; response={\"error\":{\"code\":\"no_upstream_capacity\"}}",
            }
        ]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertEqual(result["failure_class"], "model_unavailable")
        self.assertEqual(result["failure_category"], "provider_upstream_unavailable")

    def test_eval_case_result_classifies_provider_unavailable_reason(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="llm_provider_unavailable:provider_rate_limited",
        )
        steps = [{"type": "llm_error", "reason": "llm_provider_unavailable:provider_rate_limited", "detail": "429 Too Many Requests"}]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertEqual(result["failure_class"], "model_unavailable")
        self.assertEqual(result["failure_category"], "provider_rate_limited")

    def test_eval_case_result_ignores_artifact_export_network_error_for_model_availability(self) -> None:
        job = CodingJob(
            id=new_id("job"),
            user_id="u1",
            instruction="fix",
            status=JobStatus.FAILED,
            failure_reason="max_iterations_exceeded",
        )
        steps = [
            {"type": "llm_decision", "decision": {"tool": "edit_file"}},
            {
                "type": "workspace_archive_provider_failed",
                "error": "Server disconnected without sending a response.",
            },
        ]

        result = eval_case_result_from_job({"name": "python-cli", "instruction": "fix"}, job, steps)

        self.assertEqual(result["failure_class"], "budget_exceeded")
        self.assertEqual(result["failure_category"], "max_iterations_exceeded")

    def test_manifest_with_served_local_repos_rewrites_container_clone_url(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repos" / "python-bugfix"
            repo.mkdir(parents=True)
            manifest = {
                "cases": [
                    {
                        "name": "python-bugfix",
                        "repo_path": str(repo),
                        "repo_url": repo.resolve().as_uri(),
                    }
                ]
            }

            served = manifest_with_served_local_repos(manifest, base_path=root / "repos", host="host.docker.internal", port=9419)

            case = served["cases"][0]
            self.assertEqual(case["repo_url"], "git://host.docker.internal:9419/python-bugfix")
            self.assertEqual(case["local_repo_url"], repo.resolve().as_uri())
