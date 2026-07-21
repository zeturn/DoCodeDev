import tempfile
import unittest
from pathlib import Path

from docode.agent.repository_index import RepositoryIndex
from docode.agent.repository_planner import RepositoryPlanner
from docode.agent.task_graph import TaskNode, TaskStatus


class RepositoryUnderstandingTests(unittest.TestCase):
    def test_indexes_multiple_languages_and_ranks_symbol_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[project]\nname='x'", encoding="utf-8")
            (root / "service.py").write_text("def migrate_config(value):\n    return value\n", encoding="utf-8")
            (root / "client.ts").write_text("export function callService() { return 1 }", encoding="utf-8")
            (root / "main.go").write_text("package main\nfunc StartServer() {}\n", encoding="utf-8")
            index = RepositoryIndex(root)
        self.assertEqual({record.symbol for record in index.symbols}, {"migrate_config", "callService", "StartServer"})
        self.assertEqual(index.rank_files("change migrate_config")[0][0], "service.py")
        self.assertIn("pyproject.toml", index.repository_map().manifests)

    def test_task_graph_updates_and_honors_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("def load_config(): pass", encoding="utf-8")
            graph = RepositoryPlanner(RepositoryIndex(root)).initial_graph("migrate config", ["python -m unittest"])
        self.assertEqual([node.id for node in graph.ready()], ["understand"])
        graph.set_status("understand", TaskStatus.DONE, reason="ranked repository target inspected", evidence_refs=["read:config.py"])
        self.assertEqual([node.id for node in graph.ready()], ["implement"])
        graph.add_or_update(TaskNode("impact", "Check references", dependencies=["implement"]))
        self.assertIn("impact", graph.nodes)


if __name__ == "__main__":
    unittest.main()
