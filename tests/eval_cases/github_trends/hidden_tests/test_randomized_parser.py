from __future__ import annotations

import importlib.util
import pathlib
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "repo"
SPEC = importlib.util.spec_from_file_location("crawler", ROOT / "crawler.py")
crawler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(crawler)


class RandomizedParserTest(unittest.TestCase):
    def test_parser_derives_values_from_html_not_hardcoded_fixture(self) -> None:
        html = textwrap.dedent(
            """
            <main>
              <article class="Box-row">
                <h2><a href="/alpha/project-x">alpha / project-x</a></h2>
                <p>Generated hidden fixture.</p>
                <span itemprop="programmingLanguage">Rust</span>
                <a class="Link--muted" href="/alpha/project-x/stargazers">9,876</a>
                <a class="Link--muted" href="/alpha/project-x/forks">3,210</a>
                <span>1.2k stars today</span>
              </article>
            </main>
            """
        )

        records = crawler.parse_trending(html)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["rank"], 1)
        self.assertEqual(record["owner"], "alpha")
        self.assertEqual(record["repository_name"], "project-x")
        self.assertEqual(record["repository"], "alpha/project-x")
        self.assertEqual(record["url"], "https://github.com/alpha/project-x")
        self.assertEqual(record["language"], "Rust")
        self.assertEqual(record["stars_today"], 1200)
        self.assertEqual(record["total_stars"], 9876)
        self.assertEqual(record["forks"], 3210)


if __name__ == "__main__":
    unittest.main()
