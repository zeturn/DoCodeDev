from __future__ import annotations

import sys
import unittest
from pathlib import Path


class RepositoryTestModuleIdentityTests(unittest.TestCase):
    def test_repository_test_files_have_one_module_identity(self) -> None:
        root = Path(__file__).resolve().parent
        by_file: dict[Path, list[tuple[str, int]]] = {}
        for name, module in list(sys.modules.items()):
            raw = getattr(module, "__file__", None)
            if not raw:
                continue
            path = Path(raw).resolve()
            if path != root and root not in path.parents:
                continue
            by_file.setdefault(path, []).append((name, id(module)))
        duplicates = {
            str(path): identities
            for path, identities in by_file.items()
            if len({module_id for _, module_id in identities}) > 1
        }
        self.assertEqual({}, duplicates, f"repository test modules loaded more than once: {duplicates}")


if __name__ == "__main__":
    unittest.main()
