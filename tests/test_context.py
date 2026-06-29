from __future__ import annotations

from unittest import TestCase

from docode.agent.context import ContextManager
from docode.agent.inspector import ProjectInspection
from docode.agent.task_contract import TaskContract
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class ContextManagerTests(TestCase):
    def test_context_pack_preserves_task_and_summarizes_long_history(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="Refactor the payment adapter without losing retries")
        inspection = ProjectInspection(
            listing="README.md\nsrc/payments.py\ntests/test_payments.py\n",
            important_files={"README.md": "# Payments\n", "src/payments.py": "def pay(): ...\n"},
            detected_commands={"test": "python3 -m unittest", "build": None, "lint": None},
            plan=["Inspect payment flow", "Patch retry behavior", "Run tests"],
            acceptance_criteria=["Retries are preserved", "Tests pass"],
        )
        messages = []
        for index in range(35):
            messages.append(
                {
                    "role": "tool",
                    "tool": "run_command",
                    "exit_code": 0,
                    "output": "line\n" * 2000,
                    "truncated": True,
                    "metadata": {
                        "command": f"python script_{index}.py",
                        "path": f"src/file_{index}.py",
                        "original_output_bytes": 50000,
                    },
                }
            )
        messages.append({"role": "tool", "tool": "run_tests", "exit_code": 1, "output": "AssertionError: retry missing", "metadata": {"command": "python3 -m unittest"}})
        messages.append({"role": "system", "kind": "feedback", "content": "Verification failed: restore retry behavior"})

        pack = ContextManager(recent_message_limit=5, section_bytes=4000).build_pack(
            job=job,
            inspection=inspection,
            messages=messages,
            git_status=ToolResult(tool="git_status", output=" M src/payments.py\n"),
            iteration=36,
            tool_calls_count=36,
            llm_tokens_used=1234,
            llm_cost_used=0.12,
            task_contract=TaskContract(must_modify_files=["src/payments.py"], must_run_commands=["python3 -m unittest"]),
        )
        rendered = pack.render()

        self.assertIn("Refactor the payment adapter without losing retries", rendered)
        self.assertIn("Current Plan", rendered)
        self.assertIn("Failed Steps / Repair Attempts", rendered)
        self.assertIn("AssertionError: retry missing", rendered)
        self.assertIn("restore retry behavior", rendered)
        self.assertIn("src/payments.py", rendered)
        self.assertIn("You must modify src/payments.py", rendered)
        self.assertIn("You must produce non-empty git diff before final_candidate", rendered)
        self.assertIn("Current Git Diff State", rendered)
        self.assertIn('"src/payments.py"', rendered)
        self.assertIn("final_candidate_allowed: yes after tests pass", rendered)
        self.assertLess(len(rendered.encode("utf-8")), 30_000)
        self.assertEqual(len(pack.recent_messages), 5)
        self.assertNotIn("line\n" * 1000, rendered)
