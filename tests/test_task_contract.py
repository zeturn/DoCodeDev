from __future__ import annotations

from unittest import TestCase

from docode.agent.task_contract import task_contract_from_instruction, verification_commands_from_instruction


class TaskContractParserTests(TestCase):
    def test_inline_target_file(self) -> None:
        contract = task_contract_from_instruction("Target file: crawler.py")

        self.assertEqual(contract.must_modify_files, ["crawler.py"])

    def test_block_target_file(self) -> None:
        contract = task_contract_from_instruction("Target file:\ncrawler.py")

        self.assertEqual(contract.must_modify_files, ["crawler.py"])

    def test_multiple_block_target_files(self) -> None:
        contract = task_contract_from_instruction("Target files:\n- app.py\n- formatter.py")

        self.assertEqual(contract.must_modify_files, ["app.py", "formatter.py"])

    def test_edit_and_modify_target_headings(self) -> None:
        edit_contract = task_contract_from_instruction("Edit files:\n- src/a.py\n- src/b.py")
        modify_contract = task_contract_from_instruction("Modify file:\nsrc/main.py")

        self.assertEqual(edit_contract.must_modify_files, ["src/a.py", "src/b.py"])
        self.assertEqual(modify_contract.must_modify_files, ["src/main.py"])

    def test_explicit_target_prevents_command_file_fallback(self) -> None:
        contract = task_contract_from_instruction(
            "Target file:\n"
            "crawler.py\n\n"
            "Verification commands:\n"
            "1. python crawler.py --output data/output.json"
        )

        self.assertEqual(contract.must_modify_files, ["crawler.py"])

    def test_malformed_explicit_target_fails_without_fallback(self) -> None:
        contract = task_contract_from_instruction(
            "Target file:\n\n"
            "Verification commands:\n"
            "1. python crawler.py --output data/output.json"
        )

        self.assertEqual(contract.must_modify_files, [])

    def test_verification_heredoc_file_references_are_not_fallback_targets(self) -> None:
        contract = task_contract_from_instruction(
            "Build crawler.py.\n\n"
            "Verification commands:\n"
            "1. python - <<'CHECK'\n"
            "from pathlib import Path\n"
            "Path('data/output.json').read_text()\n"
            "CHECK"
        )

        self.assertEqual(contract.must_modify_files, ["crawler.py"])

    def test_quoted_and_unquoted_heredocs_are_atomic(self) -> None:
        for opener, closer in (("<<'CHECK'", "CHECK"), ('<<"CHECK"', "CHECK"), ("<<CHECK", "CHECK")):
            with self.subTest(opener=opener):
                commands = verification_commands_from_instruction(
                    "Verification commands:\n"
                    f"1. python - {opener}\n"
                    "print('one')\n"
                    f"{closer}\n"
                    "2. python -m unittest"
                )

                self.assertEqual(commands, [f"python - {opener}\nprint('one')\n{closer}", "python -m unittest"])

    def test_strip_tabs_heredoc_is_atomic(self) -> None:
        commands = verification_commands_from_instruction(
            "Verification commands:\n"
            "1. python - <<-'DONE'\n"
            "\tprint('one')\n"
            "\tDONE"
        )

        self.assertEqual(commands, ["python - <<-'DONE'\n\tprint('one')\n\tDONE"])

    def test_generic_heredoc_delimiter_is_supported(self) -> None:
        commands = verification_commands_from_instruction(
            "Verification commands:\n"
            "1. python - <<'END-CHECK'\n"
            "print('one')\n"
            "END-CHECK"
        )

        self.assertEqual(commands, ["python - <<'END-CHECK'\nprint('one')\nEND-CHECK"])

    def test_heredoc_body_commands_are_not_separate_commands(self) -> None:
        commands = verification_commands_from_instruction(
            "Verification commands:\n"
            "1. python - <<'DONE'\n"
            "python nested.py\n"
            "pytest nested_test.py\n"
            "DONE\n"
            "2. python final.py"
        )

        self.assertEqual(
            commands,
            ["python - <<'DONE'\npython nested.py\npytest nested_test.py\nDONE", "python final.py"],
        )

    def test_markdown_fences_around_commands_are_ignored(self) -> None:
        commands = verification_commands_from_instruction(
            "Verification commands:\n"
            "```bash\n"
            "1. python - <<'DONE'\n"
            "print('ok')\n"
            "DONE\n"
            "```"
        )

        self.assertEqual(commands, ["python - <<'DONE'\nprint('ok')\nDONE"])

    def test_unterminated_heredoc_does_not_emit_opener_or_body_commands(self) -> None:
        commands = verification_commands_from_instruction(
            "Verification commands:\n"
            "1. python - <<'DONE'\n"
            "python nested.py\n"
            "pytest nested_test.py"
        )

        self.assertEqual(commands, [])

    def test_exact_github_trends_instruction_preserves_target_and_validation(self) -> None:
        instruction = (
            "Build a real GitHub Trending crawler in crawler.py.\n\n"
            "Target URL:\n"
            "https://github.com/trending\n\n"
            "Target file:\n"
            "crawler.py\n\n"
            "Verification commands:\n"
            "1. python crawler.py --url https://github.com/trending --output data/output.json --dry-run\n"
            "2. python - <<'PY'\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "path = Path(\"data/output.json\")\n"
            "assert path.exists(), \"data/output.json missing\"\n"
            "records = json.loads(path.read_text())\n"
            "assert isinstance(records, list), \"output must be a list\"\n"
            "assert len(records) >= 5, f\"expected at least 5 records, got {len(records)}\"\n"
            "PY"
        )

        contract = task_contract_from_instruction(instruction)

        self.assertEqual(contract.must_modify_files, ["crawler.py"])
        self.assertEqual(len(contract.must_run_commands), 2)
        self.assertEqual(
            contract.must_run_commands[0],
            "python crawler.py --url https://github.com/trending --output data/output.json --dry-run",
        )
        self.assertTrue(contract.must_run_commands[1].startswith("python - <<'PY'\nimport json"))
        self.assertTrue(contract.must_run_commands[1].rstrip().endswith("\nPY"))
        self.assertIn("assert len(records) >= 5", contract.must_run_commands[1])
