from __future__ import annotations

from unittest import TestCase

from docode.agent.quality_gate import (
    JSON_PRODUCER_ISSUE_CODES,
    inspect_json_data,
    inspect_json_records,
    json_producer_source_path,
)
from docode.agent.task_contract import TaskContract


GITHUB_TRENDS_INSTRUCTION = """Build a real GitHub Trending crawler in crawler.py.

Each record must include: name, owner, repository, url, description, language, stars, forks, stars_today.
The repository URL must be exactly https://github.com/<owner>/<repo>.
"""


class QualityGateJsonProducerTargetTests(TestCase):
    def test_json_records_empty_targets_producer_source_not_output_artifact(self) -> None:
        issues = inspect_json_records(
            [],
            "data/output.json",
            GITHUB_TRENDS_INSTRUCTION,
            producer_path="crawler.py",
        )

        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.code, "json_records_empty")
        self.assertEqual(issue.path, "crawler.py")
        self.assertIn("data/output.json", issue.message)
        self.assertIn("Do not hand-edit the output JSON", issue.repair_hint or "")
        self.assertIn("parser/fetcher", issue.repair_hint or "")

    def test_json_github_url_invalid_targets_producer_source(self) -> None:
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
            GITHUB_TRENDS_INSTRUCTION,
            producer_path="crawler.py",
        )

        self.assertTrue(any(issue.code == "json_github_url_invalid" for issue in issues))
        self.assertTrue(all(issue.path == "crawler.py" for issue in issues))
        self.assertTrue(all("Do not hand-edit the JSON output" in (issue.repair_hint or "") for issue in issues))

    def test_unexpected_json_shape_targets_producer_source(self) -> None:
        issues = inspect_json_data(
            "not a structured artifact",
            "data/output.json",
            GITHUB_TRENDS_INSTRUCTION,
            producer_path="crawler.py",
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "json_artifact_unexpected_shape")
        self.assertEqual(issues[0].path, "crawler.py")

    def test_output_artifact_path_is_used_only_without_producer_source(self) -> None:
        issues = inspect_json_records([], "data/output.json", GITHUB_TRENDS_INSTRUCTION)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "json_records_empty")
        self.assertEqual(issues[0].path, "data/output.json")

    def test_json_producer_source_prefers_non_json_task_contract_target(self) -> None:
        contract = TaskContract(must_modify_files=["data/output.json", "crawler.py"])

        self.assertEqual(json_producer_source_path(contract), "crawler.py")

    def test_known_producer_codes_are_documented(self) -> None:
        self.assertIn("json_records_empty", JSON_PRODUCER_ISSUE_CODES)
        self.assertIn("json_github_url_invalid", JSON_PRODUCER_ISSUE_CODES)
        self.assertIn("json_repository_url_mismatch", JSON_PRODUCER_ISSUE_CODES)
