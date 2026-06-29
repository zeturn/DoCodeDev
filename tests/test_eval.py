from __future__ import annotations

import json
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from docode.cli import run_eval_command
from docode.eval import run_eval


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
