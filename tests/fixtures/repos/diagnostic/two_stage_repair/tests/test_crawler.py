from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

from crawler import parse_records


SAMPLE = {"records": [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]}


class CrawlerTests(TestCase):
    def test_parse_records(self) -> None:
        self.assertEqual(parse_records(json.dumps(SAMPLE)), SAMPLE["records"])

    def test_cli_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            completed = subprocess.run(
                [sys.executable, "crawler.py", "sample.json", "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), SAMPLE["records"])
