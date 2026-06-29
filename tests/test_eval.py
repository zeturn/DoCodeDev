from __future__ import annotations

import json
import io
import shutil
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from docode.cli import run_eval_command, run_eval_scaffold_command
from docode.eval import run_eval, scaffold_eval_suite


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

    def test_eval_run_command_writes_report(self) -> None:
        with TemporaryDirectory() as tmp:
            fixtures = Path(tmp) / "fixtures"
            fixtures.mkdir()
            report_path = Path(tmp) / "report.json"
            (fixtures / "readme.json").write_text('{"status":"succeeded","iterations":1}', encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                run_eval_command(Namespace(fixtures_dir=str(fixtures), report=str(report_path)))

            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["total"], 1)
            self.assertEqual(data["succeeded"], 1)

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
