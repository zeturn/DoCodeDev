from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.runtime_v2_release_eval.definitions import CASES
from tests.runtime_v2_release_eval.hashing import sha256_tree, write_json


class ReleaseEvalFoundationTests(unittest.TestCase):
    def test_inventory_is_exactly_eight_crawlers_and_three_repositories(self) -> None:
        self.assertEqual(8, sum(case.category == "crawler" for case in CASES))
        self.assertEqual(3, sum(case.category == "repository" for case in CASES))
        self.assertEqual(len(CASES), len({case.name for case in CASES}))

    def test_tree_hash_is_stable_and_path_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").mkdir()
            (root / "a" / "fixture.txt").write_text("payload", encoding="utf-8")
            first = sha256_tree(root)
            self.assertEqual(first, sha256_tree(root))
            (root / "a" / "fixture.txt").rename(root / "fixture.txt")
            self.assertNotEqual(first, sha256_tree(root))

    def test_json_evidence_writer_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "evidence" / "result.json"
            write_json(target, {"status": "failed"})
            self.assertEqual('{\n  "status": "failed"\n}\n', target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
