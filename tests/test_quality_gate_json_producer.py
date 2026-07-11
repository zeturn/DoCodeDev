from __future__ import annotations

from unittest import TestCase

from docode.agent.quality_gate import (
    JSON_PRODUCER_ISSUE_CODES,
    inspect_json_data,
    inspect_json_records,
    json_producer_source_path,
)
from docode.agent.task_contract import TaskContract


ARTIFACT_INSTRUCTION = """Build a feed collector in collector.py.

Each record must include: name, url, description.
The url must be absolute.
"""


class QualityGateJsonProducerTargetTests(TestCase):
    def test_json_records_empty_targets_producer_source_not_output_artifact(self) -> None:
        issues = inspect_json_records(
            [],
            "data/output.json",
            ARTIFACT_INSTRUCTION,
            producer_path="collector.py",
        )

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.code, "json_records_empty")
        self.assertEqual(issue.path, "collector.py")
        self.assertIn("data/output.json", issue.message)
        self.assertIn("Do not hand-edit the output JSON", issue.repair_hint or "")
        self.assertIn("parser/fetcher", issue.repair_hint or "")

    def test_json_absolute_url_invalid_targets_producer_source(self) -> None:
        issues = inspect_json_records(
            [
                {
                    "name": "repo",
                    "owner": "owner",
                    "repository": "owner/repo",
                    "url": "/owner/repo",
                }
            ],
            "data/output.json",
            ARTIFACT_INSTRUCTION,
            producer_path="collector.py",
        )

        self.assertTrue(any(issue.code == "json_url_invalid" for issue in issues))
        self.assertTrue(all(issue.path == "collector.py" for issue in issues))
        self.assertTrue(all("Do not hand-edit the JSON output" in (issue.repair_hint or "") for issue in issues))

    def test_unexpected_json_shape_targets_producer_source(self) -> None:
        issues = inspect_json_data(
            "not a structured artifact",
            "data/output.json",
            ARTIFACT_INSTRUCTION,
            producer_path="collector.py",
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "json_artifact_unexpected_shape")
        self.assertEqual(issues[0].path, "collector.py")

    def test_output_artifact_path_is_used_only_without_producer_source(self) -> None:
        issues = inspect_json_records([], "data/output.json", ARTIFACT_INSTRUCTION)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "json_records_empty")
        self.assertEqual(issues[0].path, "data/output.json")

    def test_json_producer_source_prefers_non_json_task_contract_target(self) -> None:
        contract = TaskContract(must_modify_files=["data/output.json", "collector.py"])

        self.assertEqual(json_producer_source_path(contract), "collector.py")

    def test_known_producer_codes_are_documented(self) -> None:
        self.assertIn("json_records_empty", JSON_PRODUCER_ISSUE_CODES)
        self.assertIn("json_url_invalid", JSON_PRODUCER_ISSUE_CODES)
