from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import docode


class ArchitectureGuardTests(unittest.TestCase):
    def test_no_runtime_hotfix_or_import_side_effect_flags(self) -> None:
        root = Path(docode.__file__).parent
        self.assertFalse((root / "_runtime_hotfix_source_pipeline.py").exists())
        self.assertFalse(hasattr(docode, "__runtime_hotfix_applied__"))
        before = dict(vars(docode))
        importlib.reload(docode)
        self.assertEqual(docode.__version__, "0.2.0")
        self.assertEqual(before["__version__"], docode.__version__)

    def test_production_agent_has_no_historical_schema_leakage(self) -> None:
        root = Path(docode.__file__).parent
        targets = [root / "agent", root / "dobox"]
        forbidden = ("parse_trending", "parse_repo_row", "stars_today", "github trending", "--preflight", "cisa", "cis benchmark", "cis control", "security advisory")
        for target in targets:
            for path in target.rglob("*.py"):
                text = path.read_text(encoding="utf-8").lower()
                for marker in forbidden:
                    self.assertNotIn(marker, text, f"{marker!r} leaked into {path}")

    def test_production_does_not_import_tests_or_eval_fixtures(self) -> None:
        root = Path(docode.__file__).parent
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("from tests", text)
            self.assertNotIn("import tests", text)


if __name__ == "__main__":
    unittest.main()
