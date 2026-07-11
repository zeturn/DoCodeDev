from __future__ import annotations

import unittest

from docode.agent.repository_index import IndexLimits, build_remote_repository_index
from docode.agent.workspace_reader import WorkspaceEntry


class MemoryReader:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.reads: list[str] = []

    async def list_files(self, path: str = ".") -> list[WorkspaceEntry]:
        return [WorkspaceEntry(name) for name in self.files]

    async def read_file(self, path: str, max_bytes: int = 256_000) -> str:
        self.reads.append(path)
        return self.files[path][:max_bytes]

    async def search(self, query: str, path: str = ".") -> list[object]:
        return []


class RemoteRepositoryIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_symbols_from_remote_reader_and_ignores_vendor(self) -> None:
        reader = MemoryReader({"src/service.py": "def load_config():\n    return 1\n", "web/client.ts": "export class Client {}", "vendor/copied.py": "def leaked(): pass", "pyproject.toml": "[project]"})
        context = await build_remote_repository_index(reader)
        self.assertEqual({item.symbol for item in context.symbols}, {"load_config", "Client"})
        self.assertNotIn("vendor/copied.py", reader.reads)
        self.assertEqual(context.repository_map.root, "/workspace")
        self.assertIn("pyproject.toml", context.repository_map.manifests)

    async def test_enforces_file_and_total_byte_budgets(self) -> None:
        reader = MemoryReader({f"src/f{i}.py": f"def f{i}(): pass\n" for i in range(10)})
        context = await build_remote_repository_index(reader, IndexLimits(maximum_files=3, maximum_total_bytes=100))
        self.assertLessEqual(len(context.files), 3)
        self.assertTrue(context.truncated)


if __name__ == "__main__":
    unittest.main()
