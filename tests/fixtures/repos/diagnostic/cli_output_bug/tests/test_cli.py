from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

from cli import build_greeting


class CliTests(TestCase):
    def test_build_greeting(self) -> None:
        self.assertEqual(build_greeting("Ada"), {"greeting": "Hello, Ada!"})

    def test_cli_writes_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.json"
            completed = subprocess.run(
                [sys.executable, "cli.py", "--name", "Ada", "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"greeting": "Hello, Ada!"})
