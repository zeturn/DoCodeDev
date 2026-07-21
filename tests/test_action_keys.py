"""Unit tests for deterministic action-key builders."""

from unittest import TestCase

from docode.agent.action_keys import (
    final_candidate_action_key,
    rejected_action_key,
    tool_action_key,
)


class ToolActionKeyTests(TestCase):
    def test_read_file(self) -> None:
        k = tool_action_key("read_file", {"path": "src/a.py"})
        self.assertIn("read_file", k)
        self.assertIn("src/a.py", k)

    def test_run_command_normalised(self) -> None:
        a = tool_action_key("run_command", {"command": "python -m pytest  "})
        b = tool_action_key("run_command", {"command": " python -m pytest"})
        self.assertEqual(a, b)

    def test_write_file_uses_content_hash(self) -> None:
        a = tool_action_key("write_file", {"path": "f", "content": "aaa"})
        b = tool_action_key("write_file", {"path": "f", "content": "bbb"})
        self.assertNotEqual(a, b)
        c = tool_action_key("write_file", {"path": "f", "content": "aaa"})
        self.assertEqual(a, c)

    def test_edit_file_stable(self) -> None:
        a = tool_action_key("edit_file", {"path": "f", "old_text": "x", "new_text": "y"})
        b = tool_action_key("edit_file", {"path": "f", "old_text": "x", "new_text": "y"})
        c = tool_action_key("edit_file", {"path": "f", "old_text": "x", "new_text": "z"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_apply_patch_stable(self) -> None:
        a = tool_action_key("apply_patch", {"patch": "hello"})
        b = tool_action_key("apply_patch", {"patch": "hello"})
        c = tool_action_key("apply_patch", {"patch": "world"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_path_slash_normalisation(self) -> None:
        a = tool_action_key("read_file", {"path": "a\\b.py"})
        b = tool_action_key("read_file", {"path": "a/b.py"})
        self.assertEqual(a, b)

    def test_search_uses_query_hash(self) -> None:
        a = tool_action_key("search", {"query": "find me", "path": "."})
        b = tool_action_key("search", {"query": "find me", "path": "."})
        c = tool_action_key("search", {"query": "other", "path": "."})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class ControllerKeyTests(TestCase):
    def test_final_candidate_stable(self) -> None:
        a = final_candidate_action_key("summary text")
        b = final_candidate_action_key("summary text")
        c = final_candidate_action_key("different")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_rejected_action_key(self) -> None:
        k = rejected_action_key("reason", "read_file:src/a.py")
        self.assertIn("decision_rejected", k)
        self.assertIn("reason", k)
