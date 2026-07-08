from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("crawler", ROOT / "crawler.py")
crawler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crawler)


class ParserTest(unittest.TestCase):
    def test_parse_fixture_records(self) -> None:
        html = (ROOT / "fixtures" / "sample.html").read_text(encoding="utf-8")
        records = crawler.parse_trending(html)
        self.assertGreaterEqual(len(records), 2)

        first = records[0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["owner"], "owner")
        self.assertEqual(first["repository"], "owner/repo")
        self.assertEqual(first["repository_name"], "repo")
        self.assertEqual(first["url"], "https://github.com/owner/repo")
        self.assertEqual(first["language"], "Python")
        self.assertEqual(first["stars_today"], 56)
        self.assertEqual(first["total_stars"], 1234)
        self.assertEqual(first["forks"], 78)

        second = records[1]
        self.assertEqual(second["rank"], 2)
        self.assertEqual(second["owner"], "acme")
        self.assertEqual(second["repository_name"], "tools")
        self.assertEqual(second["repository"], "acme/tools")

    def test_number_parser(self) -> None:
        self.assertEqual(crawler.number_from_text("1.2k"), 1200)
        self.assertEqual(crawler.number_from_text("56 stars today"), 56)


if __name__ == "__main__":
    unittest.main()
