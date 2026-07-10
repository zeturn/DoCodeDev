from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tests.crawler_benchmark_v1.definitions import CASES, LEAKAGE_MARKERS
from tests.crawler_benchmark_v1.fixture_service import FixtureServer, response_for
from tests.crawler_benchmark_v1.harness import (
    materialize_workspace,
    metrics,
    reset,
    run_collector,
    sanitize,
    validate_controlled_artifact,
    validate_live_artifact,
    variant_source,
)
from tests.crawler_benchmark_v1.reference_solutions import SOLUTION_BY_CASE


class CrawlerFixtureContractTests(TestCase):
    def test_six_frozen_cases_have_varied_workspace_shapes_and_atomic_commands(self) -> None:
        self.assertEqual(len(CASES), 6)
        self.assertEqual(sum(case.scaffold == "" for case in CASES), 2)
        self.assertEqual(sum(case.scaffold not in {None, ""} for case in CASES), 2)
        self.assertEqual(sum(case.scaffold is None for case in CASES), 2)
        self.assertGreaterEqual(sum("<<'PY'" in case.required_commands[1] for case in CASES), 3)
        self.assertTrue(any(any(path.startswith("checks/") for path, _ in case.extra_files) for case in CASES))
        for case in CASES:
            with self.subTest(case=case.name), TemporaryDirectory() as tmp:
                workspace = materialize_workspace(case, Path(tmp) / case.name)
                self.assertTrue(any(workspace.rglob("*")))
                self.assertIn(case.required_commands[0], case.instruction)
                self.assertIn(case.required_commands[1], case.instruction)
                self.assertTrue(case.required_commands[1].rstrip().endswith("PY"))

    def test_fixture_responses_cover_counts_pagination_and_edge_shapes(self) -> None:
        expected = {
            "opal_canopy": ("/aurora/cards", b"data-glyph", b"text"),
            "flint_harbor": ("/kiln/observations", b"lithic-grid", b"text"),
            "marble_tide": ("/ledger/start", b"rel='next'", b"text"),
            "violet_prism": ("/prism/feed", b"xmlns:dc", b"application"),
            "copper_orbit": ("/orbit/measurements?cursor=", b"next_cursor", b"application"),
        }
        for name, (target, marker, content_prefix) in expected.items():
            with self.subTest(case=name):
                status, content_type, payload = response_for(name, target)
                self.assertEqual(status, 200)
                self.assertIn(marker, payload)
                self.assertTrue(content_type.encode().startswith(content_prefix))

    def test_production_contains_no_new_case_or_solution_markers(self) -> None:
        production_root = Path(__file__).resolve().parents[2] / "src" / "docode"
        source = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in production_root.rglob("*.py"))
        leaked = [marker for marker in LEAKAGE_MARKERS if marker in source]
        self.assertEqual(leaked, [], f"crawler benchmark markers leaked into production: {leaked}")

    def test_sanitizer_removes_environment_and_key_shaped_secrets(self) -> None:
        previous = os.environ.get("DOCODE_DEEPSEEK_API_KEY")
        os.environ["DOCODE_DEEPSEEK_API_KEY"] = "synthetic-secret-value"
        try:
            payload = sanitize({"trace": "synthetic-secret-value and sk-examplevalue123456789"})
        finally:
            if previous is None:
                os.environ.pop("DOCODE_DEEPSEEK_API_KEY", None)
            else:
                os.environ["DOCODE_DEEPSEEK_API_KEY"] = previous
        self.assertEqual(payload, {"trace": "[REDACTED] and [REDACTED]"})


class DeterministicCrawlerExecutionTests(TestCase):
    def test_all_six_reference_collectors_pass_base_and_hidden_variant_checks(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name), TemporaryDirectory() as tmp:
                workspace = materialize_workspace(case, Path(tmp) / "workspace")
                (workspace / case.target).write_text(SOLUTION_BY_CASE[case.name], encoding="utf-8")
                service_case = case.name if case.controlled else "violet_prism"
                with FixtureServer(service_case) as source:
                    if not case.controlled:
                        result = run_collector(workspace, case, source.base_url + "/prism/feed")
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(validate_live_artifact(workspace / case.output), [])
                        continue
                    for variant in (False, True):
                        reset(source.base_url)
                        source_url = variant_source(case, source.base_url, variant)
                        result = run_collector(workspace, case, source_url)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        failures = validate_controlled_artifact(
                            case,
                            workspace / case.output,
                            base_url=source.base_url,
                            variant=variant,
                            observed_metrics=metrics(source.base_url),
                        )
                        self.assertEqual(failures, [])

    def test_reset_and_metrics_endpoints_do_not_increment_source_count(self) -> None:
        with FixtureServer("opal_canopy") as source:
            reset(source.base_url)
            self.assertEqual(metrics(source.base_url), {"count": 0, "requests": []})
            with urllib.request.urlopen(source.base_url + "/aurora/cards", timeout=5) as response:
                response.read()
            self.assertEqual(metrics(source.base_url), {"count": 1, "requests": ["/aurora/cards"]})
            reset(source.base_url)
            self.assertEqual(metrics(source.base_url), {"count": 0, "requests": []})
