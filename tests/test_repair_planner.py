from __future__ import annotations

from docode.agent.repair_planner import plan_repair_from_tool_result


def test_import_error_missing_symbol_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="ImportError: cannot import name 'parse_repositories' from 'crawler'",
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "import_error_missing_symbol"
    assert action.target_files == ["crawler.py"]
    assert "parse_repositories" in action.instruction
    assert "web_search" in action.forbidden_tools
    assert action.rerun_commands == ["python3 -m unittest discover -s tests"]
    assert action.initial_inspection_budget == 1


def test_import_error_did_you_mean_repair_hint() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "ImportError: cannot import name 'parse_trending_page' from 'crawler' "
            "(/workspace/crawler.py). Did you mean: 'parse_trending'?"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "import_error_missing_symbol"
    assert "parse_trending" in action.instruction
    assert "wrapper" in action.instruction


def test_no_tests_ran_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="Ran 0 tests in 0.000s\n\nNO TESTS RAN",
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "no_tests_ran"
    assert "tests/test_parser.py" in action.target_files


def test_fixture_missing_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="FileNotFoundError: [Errno 2] No such file or directory: '/workspace/tests/fixtures/github_trending.html'",
        metadata={"command": "python3 crawler.py --dry-run"},
    )

    assert action is not None
    assert action.category == "fixture_missing"
    assert action.target_files == ["tests/fixtures/github_trending.html"]
    assert action.rerun_commands == ["python3 crawler.py --dry-run"]


def test_fixture_missing_normalizes_parent_segments() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="FileNotFoundError: [Errno 2] No such file or directory: '/workspace/tests/../tests/fixtures/trending.html'",
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "fixture_missing"
    assert action.target_files == ["tests/fixtures/trending.html"]


def test_name_error_did_you_mean_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "Traceback (most recent call last):\n"
            '  File "/workspace/tests/test_parser.py", line 20, in test_parse\n'
            "    repos = parse_repositories(self.html_content)\n"
            '  File "/workspace/crawler.py", line 192, in parse_repositories\n'
            "    parser = _GitHubTrendingParser()\n"
            "NameError: name '_GitHubTrendingParser' is not defined. Did you mean: 'GitHubTrendingParser'?"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "name_error_did_you_mean"
    assert action.signature == "name_error_did_you_mean:_GitHubTrendingParser:GitHubTrendingParser"
    assert action.target_files == ["crawler.py"]
    assert "replace the undefined symbol `_GitHubTrendingParser` with `GitHubTrendingParser`" in action.instruction
    assert action.initial_inspection_budget == 1


def test_json_repository_semantic_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="json_repository_invalid_format: row 0 repository must look like owner/repo, got 'user/'",
        metadata={"command": "python3 crawler.py --dry-run"},
    )

    assert action is not None
    assert action.category == "json_semantic_failure"
    assert action.target_files == ["crawler.py"]


def test_key_error_missing_required_field_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "ERROR: test_parse\n"
            "Traceback (most recent call last):\n"
            '  File "/workspace/tests/test_parser.py", line 20, in test_parse\n'
            "    repo['forks']\n"
            "KeyError: 'forks'\n"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "missing_required_field"
    assert action.target_files == ["crawler.py"]
    assert "forks" in action.instruction
    assert "parser function itself" in action.instruction
    assert "forks: default 0" in action.instruction
    assert action.rerun_commands == ["python3 -m unittest discover -s tests"]


def test_assertion_not_found_missing_required_field_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output="AssertionError: 'language' not found in {'repository': 'owner/repo', 'url': 'https://github.com/owner/repo'}",
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "missing_required_field"
    assert "language" in action.instruction
    assert "language: default \"\"" in action.instruction


def test_missing_required_field_merges_multiple_fields() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "KeyError: 'forks'\n"
            "AssertionError: 'language' not found in {'repository': 'owner/repo'}\n"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "missing_required_field"
    assert action.signature == "missing_required_field:forks,language"
    assert "`forks`" in action.instruction
    assert "`language`" in action.instruction


def test_parser_records_empty_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "FAIL: test_parse_returns_list (test_parser.TestParser.test_parse_returns_list)\n"
            "Traceback (most recent call last):\n"
            '  File "/workspace/tests/test_parser.py", line 24, in test_parse_returns_list\n'
            "    self.assertGreaterEqual(len(results), 1)\n"
            "AssertionError: 0 not greater than or equal to 1\n"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "parser_records_empty"
    assert action.target_files == ["crawler.py"]
    assert "parser function returned 0 records" in action.instruction
    assert "dry-run JSON may already contain records" in action.instruction
    assert action.rerun_commands == ["python3 -m unittest discover -s tests"]


def test_parsed_value_mismatch_repair() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            "FAIL: test_parse_repositories_values (test_parser.TestParser.test_parse_repositories_values)\n"
            "Traceback (most recent call last):\n"
            '  File "/workspace/tests/test_parser.py", line 40, in test_parse_repositories_values\n'
            "    self.assertEqual(repo['repository'], 'user/repo')\n"
            "AssertionError: 'user1/repo1' != 'user/repo'\n"
            "- user1/repo1\n"
            "?     -     -\n"
            "+ user/repo\n"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "parsed_value_mismatch"
    assert action.target_files == ["crawler.py"]
    assert "Observed value: `user1/repo1`" in action.instruction
    assert "Expected value: `user/repo`" in action.instruction
    assert "fixture/test consistency" in action.instruction


def test_syntax_error_ignores_stdlib_traceback_target() -> None:
    action = plan_repair_from_tool_result(
        tool="run_command",
        output=(
            '  File "/usr/lib/python3.12/unittest/loader.py", line 394, in _find_test_path\n'
            "    module = self._get_module_from_name(name)\n"
            '  File "/workspace/tests/test_parser.py", line 8, in <module>\n'
            "    from crawler import parse_trending_page\n"
            '  File "/workspace/tests/../crawler.py", line 169\n'
            "    return\n"
            "    ^^^^^^\n"
            "IndentationError: expected an indented block after 'if' statement on line 166\n"
        ),
        metadata={"command": "python3 -m unittest discover -s tests"},
    )

    assert action is not None
    assert action.category == "syntax_error"
    assert action.target_files == ["crawler.py"]
    assert "usr/lib" not in action.instruction
