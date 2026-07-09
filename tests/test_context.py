from __future__ import annotations

from unittest import TestCase

from docode.agent.context import ContextManager
from docode.agent.inspector import ProjectInspection
from docode.agent.task_contract import TaskContract, task_contract_from_instruction
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class ContextManagerTests(TestCase):
    def test_task_contract_does_not_require_natural_language_git_diff_check(self) -> None:
        contract = task_contract_from_instruction(
            "Make a minimal code change.\n\n"
            "Evaluation hints:\n"
            "- target file: module.py\n"
            "- Verification commands:\n"
            "- git diff is non-empty\n"
            "- Semantic checks:\n"
            "- artifact_mode=pr"
        )

        self.assertEqual(contract.must_modify_files, ["module.py"])
        self.assertEqual(contract.must_run_commands, [])

    def test_task_contract_only_requires_explicit_edit_target_for_fixture_parser(self) -> None:
        contract = task_contract_from_instruction(
            "Implement crawler.py so it parses fixtures/products.html and writes product records to JSON.\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests\n"
            "2. python crawler.py fixtures/products.html --output out.json"
        )

        self.assertEqual(contract.must_modify_files, ["crawler.py"])
        self.assertEqual(
            contract.must_run_commands,
            [
                "python -m unittest discover -s tests",
                "python crawler.py fixtures/products.html --output out.json",
            ],
        )

    def test_task_contract_keeps_multiple_explicit_edit_targets(self) -> None:
        contract = task_contract_from_instruction("Update src/a.py and src/b.py to share the new parser behavior.")

        self.assertEqual(contract.must_modify_files, ["src/a.py", "src/b.py"])

    def test_task_contract_does_not_invent_cli_command_from_filename(self) -> None:
        contract = task_contract_from_instruction("Update cli.py so it can write JSON output.")

        self.assertEqual(contract.must_modify_files, ["cli.py"])
        self.assertEqual(contract.must_run_commands, [])

    def test_task_contract_does_not_invent_calculator_unittest_from_filename(self) -> None:
        contract = task_contract_from_instruction("Fix calculator.py so addition works.")

        self.assertEqual(contract.must_modify_files, ["calculator.py"])
        self.assertEqual(contract.must_run_commands, [])

    def test_task_contract_keeps_explicit_verification_commands_exactly(self) -> None:
        contract = task_contract_from_instruction(
            "Update cli.py so it writes output.\n\n"
            "Verification commands:\n"
            "1. python cli.py --name Ada --output out.json\n"
            "2. python -m json.tool out.json"
        )

        self.assertEqual(
            contract.must_run_commands,
            ["python cli.py --name Ada --output out.json", "python -m json.tool out.json"],
        )

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

    def test_crawler_context_adds_dependency_and_artifact_contract(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="Build a GitHub Trending crawler")

        text = ContextManager().task_contract(job, task_contract=TaskContract())

        self.assertIn("Crawler dependency policy", text)
        self.assertIn("dry-run must write the requested output artifact", text)
        self.assertIn("offline fixture mode", text)
