from __future__ import annotations

import unittest

from docode.agent.profiles import select_task_profile
from docode.agent.prompts import DOCODE_SYSTEM_PROMPT


class TaskProfileTests(unittest.TestCase):
    def test_selects_crawler_from_semantics_without_site_assumptions(self) -> None:
        profile = select_task_profile("Collect an XML feed from https://example.test/feed.xml")
        self.assertEqual(profile.name, "crawler")
        self.assertTrue(profile.source_inspection_required)
        self.assertEqual(profile.allowed_source_schemes, ("http", "https"))

    def test_selects_repository_profile_for_cross_file_migration(self) -> None:
        profile = select_task_profile("Perform a multi-file configuration migration")
        self.assertEqual(profile.name, "repository_task")
        self.assertTrue(profile.context_policy.use_repository_index)

    def test_defaults_to_generic(self) -> None:
        profile = select_task_profile("Correct the arithmetic in calculator.py")
        self.assertEqual(profile.name, "generic")
        self.assertFalse(profile.source_inspection_required)

    def test_generic_prompt_contains_no_historical_crawler_schema(self) -> None:
        forbidden = ("parse_trending", "parse_repo_row", "stars_today", "GitHub Trending", "--preflight")
        for marker in forbidden:
            self.assertNotIn(marker, DOCODE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
