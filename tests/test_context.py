from __future__ import annotations

import json
from unittest import TestCase

from docode.agent.context import ContextManager
from docode.agent.inspector import ProjectInspection
from docode.agent.task_contract import TaskContract, task_contract_from_instruction
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class ContextManagerTests(TestCase):
    def test_source_inspection_body_is_shown_once_then_compacted(self) -> None:
        instruction = "Build a collector for source https://example.test/feed.xml and update collector.py."
        job = CodingJob(id=new_id("job"), user_id="u1", instruction=instruction)
        payload = {
            "requested_url": "https://example.test/feed.xml",
            "final_url": "https://example.test/feed.xml",
            "status_code": 200,
            "execution_scope": "sandbox",
            "mode": "raw",
            "body": "<rss><item><title>Observed title</title></item></rss>",
        }
        messages = [
            {
                "role": "tool",
                "tool": "inspect_source",
                "exit_code": 0,
                "output": json.dumps(payload),
                "metadata": {key: value for key, value in payload.items() if key != "body"},
            }
        ]
        manager = ContextManager()
        kwargs = {
            "job": job,
            "inspection": ProjectInspection(listing="collector.py\n"),
            "messages": messages,
            "git_status": ToolResult(tool="git_status", output=""),
            "iteration": 1,
            "tool_calls_count": 1,
            "llm_tokens_used": 0,
            "llm_cost_used": 0.0,
            "task_contract": task_contract_from_instruction(instruction),
        }

        first = manager.build_pack(**kwargs, include_source_body=True)
        later = manager.build_pack(**kwargs, include_source_body=False)

        self.assertIn("Raw source excerpt (shown once)", first.source_inspection)
        self.assertIn("Observed title", first.source_inspection)
        self.assertNotIn("Observed title", later.source_inspection)
        self.assertIn("raw body was already shown", later.source_inspection)
        self.assertEqual(first.recent_messages[0]["output"], "<source body represented in Source Inspection>")

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

    def test_context_adds_action_summary_for_inspected_clean_diff_candidate_targets(self) -> None:
        instruction = (
            "Fix the profile formatting bug across app.py and formatter.py.\n\n"
            "Target files: app.py, formatter.py\n\n"
            "Verification commands:\n"
            "1. python -m unittest discover -s tests"
        )
        job = CodingJob(id=new_id("job"), user_id="u1", instruction=instruction)
        messages = [
            {"role": "tool", "tool": "read_file", "exit_code": 0, "output": "def build_profile(): ...", "metadata": {"path": "app.py"}},
            {"role": "tool", "tool": "read_file", "exit_code": 0, "output": "def format_user(): ...", "metadata": {"path": "formatter.py"}},
            {"role": "tool", "tool": "read_file", "exit_code": 0, "output": "class AppTests: ...", "metadata": {"path": "tests/test_app.py"}},
        ]

        pack = ContextManager().build_pack(
            job=job,
            inspection=ProjectInspection(listing="app.py\nformatter.py\ntests/test_app.py\n"),
            messages=messages,
            git_status=ToolResult(tool="git_status", output=""),
            iteration=4,
            tool_calls_count=3,
            llm_tokens_used=0,
            llm_cost_used=0.0,
            task_contract=task_contract_from_instruction(instruction),
        )
        rendered = pack.render()

        self.assertIn("Already inspected:", rendered)
        self.assertIn("- app.py", rendered)
        self.assertIn("- formatter.py", rendered)
        self.assertIn("- tests/test_app.py", rendered)
        self.assertIn("Git diff is empty.", rendered)
        self.assertIn("Choose the most likely source file and edit it now.", rendered)
        self.assertIn("Repeated inspection warning", rendered)
        self.assertIn("Candidate target files: app.py, formatter.py", rendered)
        self.assertNotIn("You must modify app.py", rendered)
        self.assertNotIn("You must modify formatter.py", rendered)

    def test_context_keeps_strict_wording_for_explicit_modify_file(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="Modify formatter.py so display names are normalized.")

        text = ContextManager().task_contract(
            job,
            task_contract=task_contract_from_instruction(job.instruction),
        )

        self.assertIn("You must modify formatter.py", text)
        self.assertNotIn("Candidate target files: formatter.py", text)

    def test_crawler_context_adds_dependency_and_artifact_contract(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="u1", instruction="Build a GitHub Trending crawler")

        text = ContextManager().task_contract(job, task_contract=TaskContract())

        self.assertIn("Crawler dependency policy", text)
        self.assertIn("dry-run must write the requested output artifact", text)
        self.assertIn("offline fixture mode", text)
