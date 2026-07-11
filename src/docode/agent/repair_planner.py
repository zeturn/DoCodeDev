from __future__ import annotations

import re
import posixpath
from dataclasses import dataclass, field
from typing import Any


TARGETED_REPAIR_ALLOWED_TOOLS = [
    "read_file",
    "edit_file",
    "write_file",
    "replace_in_file",
    "apply_patch",
    "run_command",
    "git_status",
    "git_diff",
]

TARGETED_REPAIR_FORBIDDEN_TOOLS = [
    "web_search",
    "fetch_url",
    "preview",
    "logs",
]


@dataclass(frozen=True, slots=True)
class RepairAction:
    category: str
    signature: str
    reason: str
    target_files: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    instruction: str = ""
    rerun_commands: list[str] = field(default_factory=list)
    exploration_forbidden: bool = True
    initial_inspection_budget: int = 2
    failure_class: str = ""
    producer_semantic_result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "signature": self.signature,
            "reason": self.reason,
            "target_files": self.target_files,
            "allowed_tools": self.allowed_tools,
            "forbidden_tools": self.forbidden_tools,
            "instruction": self.instruction,
            "rerun_commands": self.rerun_commands,
            "exploration_forbidden": self.exploration_forbidden,
            "initial_inspection_budget": self.initial_inspection_budget,
            "failure_class": self.failure_class,
            "producer_semantic_result": self.producer_semantic_result,
        }


IMPORT_NAME_RE = re.compile(
    r"cannot import name ['\"](?P<symbol>[A-Za-z_][A-Za-z0-9_]*)['\"] from ['\"](?P<module>[A-Za-z_][A-Za-z0-9_]*)"
)
DID_YOU_MEAN_RE = re.compile(r"Did you mean:\s*['\"]?(?P<suggestion>[A-Za-z_][A-Za-z0-9_]*)['\"]?")
MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError:\s+No module named ['\"](?P<module>[A-Za-z_][A-Za-z0-9_]*)['\"]")
NAME_ERROR_RE = re.compile(r"NameError:\s+name ['\"](?P<missing>[A-Za-z_][A-Za-z0-9_]*)['\"] is not defined")
UNBOUND_LOCAL_RE = re.compile(
    r"UnboundLocalError:\s+cannot access local variable ['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]"
)
KEY_ERROR_RE = re.compile(r"KeyError:\s+['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"]")
ASSERT_NOT_FOUND_RE = re.compile(r"AssertionError:\s+['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"] not found in")
ASSERT_MIN_RECORDS_RE = re.compile(
    r"AssertionError:\s+(?P<actual>\d+)\s+not greater than(?: or equal to)?\s+(?P<minimum>\d+)"
)
ASSERT_VALUE_MISMATCH_RE = re.compile(
    r"AssertionError:\s+(?P<actual>.+?)\s+!=\s+(?P<expected>.+?)(?:\n|$)"
)
FILE_NOT_FOUND_RE = re.compile(r"(?:FileNotFoundError|No such file or directory).*?['\"](?P<path>[^'\"]+(?:fixtures|fixture)[^'\"]*)['\"]")
SYNTAX_ERROR_FILE_RE = re.compile(r'File ["\'](?P<path>[^"\']+\.py)["\'], line (?P<line>\d+)')
TRACEBACK_FILE_RE = re.compile(r'File ["\'](?P<path>[^"\']+\.py)["\'], line \d+')
ARGPARSE_UNRECOGNIZED_RE = re.compile(r"(?P<script>[A-Za-z0-9_./-]+\.py):\s+error:\s+unrecognized arguments:\s+(?P<args>.+)")
INVALID_INT_LITERAL_RE = re.compile(
    r"ValueError:\s+invalid literal for int\(\) with base 10:\s+['\"](?P<value>[^'\"]+)['\"]"
)

FIELD_DEFAULT_HINTS = {
    "language": '""',
    "description": '""',
    "repository": '""',
    "url": '"https://..."',
    "stars": "0",
    "forks": "0",
    "stars_today": "0",
    "name": '""',
    "owner": '""',
    "id": '""',
    "title": '""',
    "status": '""',
}


def plan_repair_from_tool_result(*, tool: str, output: str, metadata: dict[str, Any] | None = None) -> RepairAction | None:
    metadata = metadata or {}
    command = str(metadata.get("command") or "")
    if tool != "run_command" and not command:
        return None

    for planner in (
        plan_import_error_missing_symbol,
        plan_name_error_did_you_mean,
        plan_unbound_local_error,
        plan_dependency_failure,
        plan_module_not_found,
        plan_missing_required_field,
        plan_parser_records_empty,
        plan_parsed_value_mismatch,
        plan_no_tests_ran,
        plan_fixture_missing,
        plan_json_semantic_failure,
        plan_cli_unrecognized_arguments,
        plan_invalid_number_literal,
        plan_syntax_error,
    ):
        action = planner(output=output, command=command)
        if action is not None:
            return action
    return None


def plan_unbound_local_error(*, output: str, command: str) -> RepairAction | None:
    match = UNBOUND_LOCAL_RE.search(output)
    if not match:
        return None
    name = match.group("name")
    target = inferred_source_targets(output, command)[0]
    rerun = command or "python3 -m unittest discover -s tests"
    return RepairAction(
        category="unbound_local_error",
        signature=f"unbound_local_error:{target}:{name}",
        reason=f"{target} references local variable `{name}` before it is assigned.",
        target_files=[target],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        instruction=(
            f"The command failed because `{name}` is treated as a local variable before assignment.\n\n"
            "Required fix:\n"
            f"1. Edit `{target}` so `{name}` is imported or assigned at module scope, or remove the later local assignment/import that shadows it.\n"
            "2. Keep the CLI behavior and tests intact.\n"
            f"3. Rerun exactly: `{rerun}`."
        ),
        rerun_commands=[rerun],
        exploration_forbidden=True,
        initial_inspection_budget=0,
    )


def plan_invalid_number_literal(*, output: str, command: str) -> RepairAction | None:
    match = INVALID_INT_LITERAL_RE.search(output)
    if not match:
        return None
    value = match.group("value")
    target = inferred_source_targets(output, command)[0]
    rerun = command or "python3 -m unittest discover -s tests"
    return RepairAction(
        category="number_parser_invalid_literal",
        signature=f"number_parser_invalid_literal:{stable_signature_fragment(value)}",
        reason=f"number_from_text tried to parse non-numeric text as int: {value!r}.",
        target_files=[target],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        instruction=(
            "The test failed because numeric parsing called int() on text that still contains words.\n\n"
            f"Input value: `{value}`\n\n"
            "Required fix:\n"
            "1. Update the numeric parser so it extracts the first numeric token from mixed text.\n"
            "2. Support commas and compact magnitude suffixes such as k/m.\n"
            "3. Do not weaken or delete parser tests.\n"
            f"4. Rerun exactly: `{rerun}`."
        ),
        rerun_commands=[rerun],
        exploration_forbidden=True,
        initial_inspection_budget=0,
    )


def plan_cli_unrecognized_arguments(*, output: str, command: str) -> RepairAction | None:
    match = ARGPARSE_UNRECOGNIZED_RE.search(output)
    if not match:
        return None
    script = normalize_workspace_relative_path(match.group("script"))
    if not script.endswith(".py"):
        return None
    args = " ".join(match.group("args").split())
    rerun = command or f"python3 {script}"
    return RepairAction(
        category="cli_unrecognized_arguments",
        signature=f"cli_unrecognized_arguments:{script}:{stable_signature_fragment(args)}",
        reason=f"{script} does not accept required CLI arguments: {args}",
        target_files=[script],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The required verification command failed because `{script}` rejects required CLI arguments.\n\n"
            f"Unrecognized arguments: `{args}`\n\n"
            "Required next action:\n"
            f"1. Edit `{script}` to add argparse support for those options.\n"
            "2. Preserve the existing working options and parser functions.\n"
            "3. If `--source` is present, read that local fixture path instead of fetching the network.\n"
            "4. If `--output` is present, write JSON output there, including in dry-run mode.\n"
            "5. Do not weaken or delete tests.\n"
            f"6. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=1,
    )


def plan_parser_records_empty(*, output: str, command: str) -> RepairAction | None:
    match = ASSERT_MIN_RECORDS_RE.search(output)
    if not match:
        return None
    actual = int(match.group("actual"))
    minimum = int(match.group("minimum"))
    if actual >= minimum:
        return None
    targets = inferred_source_targets(output, command)
    rerun = command or "python3 -m unittest discover -s tests"
    return RepairAction(
        category="parser_records_empty",
        signature=f"parser_records_empty:{actual}:min{minimum}",
        reason=f"Parser/function returned {actual} records but tests require at least {minimum}.",
        target_files=targets,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The required test command failed because the parser function returned {actual} records, "
            f"but the test requires at least {minimum}.\n\n"
            "Important:\n"
            "The dry-run JSON may already contain records because a later fallback, fixture path, or writer step works. "
            "That is not enough. The parser function called by the tests must parse the test fixture and return records directly.\n\n"
            "Required next action:\n"
            "1. Edit the parser/record-construction code in the target source file.\n"
            "2. Ensure the parser function used by tests returns at least one record for the fixture HTML.\n"
            "3. Align the parser function and dry-run path so they use the same parsing logic.\n"
            "4. Do not only make the CLI writer or dry-run fallback output JSON records.\n"
            "5. Do not weaken or delete the tests.\n"
            "6. Do not call web_search or fetch_url for this repair.\n"
            f"7. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=1,
    )


def plan_parsed_value_mismatch(*, output: str, command: str) -> RepairAction | None:
    match = ASSERT_VALUE_MISMATCH_RE.search(output)
    if not match:
        return None
    actual = clean_assertion_value(match.group("actual"))
    expected = clean_assertion_value(match.group("expected"))
    if actual is None or expected is None:
        return None
    field = assertion_field_name(output)
    targets = unique_preserving_order(
        [
            *inferred_source_targets(output, command),
            *inferred_test_targets(output, command),
            *infer_named_fixture_files(output, command),
        ]
    )
    rerun = command or "python3 -m unittest discover -s tests"
    field_line = f"Field under test: `{field}`\n" if field else ""
    return RepairAction(
        category="parsed_value_mismatch",
        signature=f"parsed_value_mismatch:{field or 'value'}:{stable_signature_fragment(actual)}:{stable_signature_fragment(expected)}",
        reason=f"Parser/function returned value {actual!r}, but tests expected {expected!r}.",
        target_files=targets,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The required test command failed because a parser-returned value was incorrect.\n\n"
            f"{field_line}"
            f"Observed value: `{actual}`\n"
            f"Expected value: `{expected}`\n\n"
            "Important:\n"
            "The final dry-run JSON may have the right shape, but the failing test checks exact values returned by the parser function. "
            "Fix parser logic or fixture/test consistency so the parser returns the expected value directly.\n\n"
            "Required next action:\n"
            "1. Inspect the failing assertion, the source function, and any fixture file used by the test.\n"
            "2. Fix source logic or fixture/test consistency based on evidence from those files.\n"
            "3. Do not only patch final JSON serialization.\n"
            "4. Do not weaken or delete the tests.\n"
            "5. Do not call web_search or fetch_url for this repair.\n"
            f"6. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=0 if field in {"stars_today", "stars", "forks", "total_stars", "owner", "repository", "repository_name"} else 1,
    )


def plan_name_error_did_you_mean(*, output: str, command: str) -> RepairAction | None:
    match = NAME_ERROR_RE.search(output)
    if not match:
        return None
    missing = match.group("missing")
    suggestion_match = DID_YOU_MEAN_RE.search(output)
    suggestion = suggestion_match.group("suggestion") if suggestion_match else ""
    targets = inferred_source_targets(output, command)
    rerun = command or "python3 -m unittest discover -s tests"
    if suggestion:
        category = "name_error_did_you_mean"
        signature = f"name_error_did_you_mean:{missing}:{suggestion}"
        reason = f"Undefined Python symbol: {missing}; suggested replacement: {suggestion}"
        instruction = (
            f"The command failed because `{missing}` is referenced but not defined. Python suggested `{suggestion}`.\n\n"
            "Required next action:\n"
            f"1. Edit the target source file and replace the undefined symbol `{missing}` with `{suggestion}` "
            "if the existing symbol is the intended one.\n"
            f"2. Prefer replacing references like `{missing}(` with `{suggestion}(` rather than defining a new alias, "
            "unless the missing symbol is intentionally required.\n"
            "3. Do not continue unrelated diagnosis.\n"
            "4. Do not call web_search or fetch_url.\n"
            f"5. Rerun exactly: `{rerun}`."
        )
    else:
        category = "name_error_undefined"
        signature = f"name_error_undefined:{missing}"
        reason = f"Undefined Python symbol: {missing}"
        instruction = (
            f"The command failed because `{missing}` is referenced but not defined.\n\n"
            "Required next action:\n"
            f"1. Edit the target source file to define `{missing}` or replace it with the correct existing symbol.\n"
            "2. Do not continue unrelated diagnosis.\n"
            "3. Do not call web_search or fetch_url.\n"
            f"4. Rerun exactly: `{rerun}`."
        )
    return RepairAction(
        category=category,
        signature=signature,
        reason=reason,
        target_files=targets,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=instruction,
        initial_inspection_budget=1,
    )


def plan_missing_required_field(*, output: str, command: str) -> RepairAction | None:
    fields: list[str] = []
    for match in KEY_ERROR_RE.finditer(output):
        field = match.group("field")
        if field not in fields:
            fields.append(field)
    for match in ASSERT_NOT_FOUND_RE.finditer(output):
        field = match.group("field")
        if field not in fields:
            fields.append(field)
    if not fields:
        return None

    targets = inferred_source_targets(output, command)
    rerun = command or "python3 -m unittest discover -s tests"
    field_list = ", ".join(f"`{field}`" for field in fields)
    defaults = default_hint_for_fields(fields)
    defaults_section = f"\nSuggested safe defaults:\n{defaults}\n" if defaults else ""
    return RepairAction(
        category="missing_required_field",
        signature=f"missing_required_field:{','.join(sorted(fields))}",
        reason=f"Parser/function returned records missing required field(s): {', '.join(fields)}.",
        target_files=targets,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The tests failed because parser-returned record dictionaries are missing required field(s): {field_list}.\n\n"
            "Important:\n"
            "The final dry-run JSON may already contain these fields because a later writer/defaulting step fills them in. "
            "That is not enough. The parser function itself must return records containing these keys.\n"
            f"{defaults_section}\n"
            "Required next action:\n"
            "1. Edit the parser/record-construction code in the target source file.\n"
            f"2. Ensure every record returned by the parser function itself includes these keys: {field_list}.\n"
            "3. Do not only add defaults in the JSON writer or final serialization step.\n"
            "4. Use safe default values if the source HTML does not contain a value.\n"
            "5. Do not weaken or delete the tests.\n"
            "6. Do not call web_search or fetch_url for this repair.\n"
            f"7. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=1,
    )


def plan_import_error_missing_symbol(*, output: str, command: str) -> RepairAction | None:
    match = IMPORT_NAME_RE.search(output)
    if not match:
        return None
    symbol = match.group("symbol")
    module = match.group("module")
    target = f"{module}.py"
    rerun = command or "python3 -m unittest discover -s tests"
    suggestion_match = DID_YOU_MEAN_RE.search(output)
    suggestion = suggestion_match.group("suggestion") if suggestion_match else ""
    suggestion_hint = ""
    if suggestion:
        suggestion_hint = (
            f"\nPython suggested an existing symbol `{suggestion}`. Prefer adding a small top-level wrapper "
            f"`{symbol}` that delegates to `{suggestion}` if their signatures are compatible, or rename/export "
            "the existing implementation if that is cleaner.\n"
        )
    return RepairAction(
        category="import_error_missing_symbol",
        signature=f"import_error_missing_symbol:{module}:{symbol}",
        reason=f"Tests import {symbol} from {module}, but {target} does not export it.",
        target_files=[target],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The test failure is caused by a missing exported symbol. The tests import `{symbol}` from `{module}`, "
            f"but `{target}` does not define/export it.\n\n"
            "Required next action:\n"
            f"1. Read `{target}` and the failing test only if needed.\n"
            f"2. Edit `{target}` to define a top-level `{symbol}` with the expected behavior/signature.\n"
            f"{suggestion_hint}"
            "3. Do not call web_search or fetch_url.\n"
            "4. Do not weaken or delete the tests unless they are clearly invalid.\n"
            f"5. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=1,
    )


def plan_module_not_found(*, output: str, command: str) -> RepairAction | None:
    match = MODULE_NOT_FOUND_RE.search(output)
    if not match:
        return None
    module = match.group("module")
    target = f"{module}.py"
    rerun = command or "python3 -m unittest discover -s tests"
    return RepairAction(
        category="module_not_found",
        signature=f"module_not_found:{module}",
        reason=f"Tests import module {module}, but {target} is missing or not importable.",
        target_files=[target],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The tests import module `{module}`, but `{target}` is missing or not importable.\n\n"
            "Required next action:\n"
            f"1. Ensure `{target}` exists at the workspace root unless the tests clearly expect a package.\n"
            "2. Move or rename the implementation if it was written to the wrong file.\n"
            "3. Do not continue unrelated diagnosis.\n"
            f"4. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=1,
    )


def plan_no_tests_ran(*, output: str, command: str) -> RepairAction | None:
    lowered = output.lower()
    if not any(
        pattern in lowered
        for pattern in (
            "no tests ran",
            "ran 0 tests",
            "collected 0 items",
            "start directory is not importable: 'tests'",
            'start directory is not importable: "tests"',
        )
    ):
        return None
    rerun = command or "python3 -m unittest discover -s tests"
    return RepairAction(
        category="no_tests_ran",
        signature=f"no_tests_ran:{rerun}",
        reason="The required test command did not discover importable tests.",
        target_files=inferred_test_targets(output, command),
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            "The required test command discovered zero tests.\n\n"
            "Required next action:\n"
            "1. Ensure tests live under `tests/`.\n"
            "2. Ensure test files are named `test_*.py`.\n"
            "3. If using unittest, define a `unittest.TestCase` subclass with methods beginning with `test_`.\n"
            "4. Do not continue unrelated diagnosis.\n"
            f"5. Rerun exactly: `{rerun}`."
        ),
        initial_inspection_budget=0,
    )


def plan_fixture_missing(*, output: str, command: str) -> RepairAction | None:
    match = FILE_NOT_FOUND_RE.search(output)
    if not match:
        return None
    path = normalize_workspace_relative_path(match.group("path"))
    return RepairAction(
        category="fixture_missing",
        signature=f"fixture_missing:{path}",
        reason=f"Required fixture file is missing: {path}",
        target_files=[path],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=[],
        rerun_commands=[command] if command else [],
        instruction=(
            f"The command failed because the fixture file `{path}` does not exist.\n\n"
            "Required next action:\n"
            f"1. Create `{path}` with representative offline sample content.\n"
            "2. Ensure the parser supports reading that fixture.\n"
            "3. If prior source evidence exists, use it or a minimal representative subset to create the fixture.\n"
            "4. Do not fetch more URLs unless no prior source evidence exists.\n"
            f"5. Rerun exactly: `{command}`."
        ),
        initial_inspection_budget=1,
    )


def plan_json_semantic_failure(*, output: str, command: str) -> RepairAction | None:
    lowered = output.lower()
    markers = (
        "json_required_field_empty",
        "json_url_invalid",
        "json_github_url_invalid",
        "json_records_empty",
        "json_repository_invalid",
        "json_repository_invalid_format",
        "json_repository_url_mismatch",
        "required field",
        "repository is empty",
        "repository has invalid format",
        "does not match url",
        "url is empty",
    )
    if not any(marker in lowered for marker in markers):
        return None
    commands = [command] if command else []
    return RepairAction(
        category="json_semantic_failure",
        signature="json_semantic_failure",
        reason="The output JSON exists but fails semantic quality checks.",
        target_files=inferred_source_targets(output, command),
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=commands,
        instruction=(
            "The JSON artifact exists but has semantically invalid records, such as empty required fields or invalid URLs.\n\n"
            "Required next action:\n"
            "1. Fix the parser/serializer so required fields are meaningful and non-empty.\n"
            "2. Do not weaken the semantic check.\n"
            "3. Derive related identifier and URL fields from the same source value when both are present.\n"
            "4. Rerun the failing command and semantic check."
        ),
        initial_inspection_budget=1,
    )


def plan_dependency_failure(*, output: str, command: str) -> RepairAction | None:
    lowered = output.lower()
    missing_third_party_import = any(
        marker in lowered
        for marker in (
            "modulenotfounderror: no module named 'bs4'",
            'modulenotfounderror: no module named "bs4"',
            "no module named 'requests'",
            'no module named "requests"',
        )
    )
    markers = (
        "moduleNotFoundError: No module named 'bs4'".lower(),
        'moduleNotFoundError: No module named "bs4"'.lower(),
        "no module named 'requests'",
        'no module named "requests"',
        "externally-managed-environment",
        "beautifulsoup4 package is required",
    )
    if not any(marker in lowered for marker in markers):
        return None
    if missing_third_party_import:
        target_files = inferred_source_targets(output, command)
        dependency_instruction = (
            "2. Remove the unavailable third-party import from the source file and use the Python standard library instead.\n"
            "3. Do not satisfy this repair by only adding `requirements.txt` or `pyproject.toml`; the required command reruns without installing dependencies.\n"
            "4. Rerun the failing command after editing the source file."
        )
    else:
        target_files = unique_preserving_order([*inferred_source_targets(output, command), "requirements.txt", "pyproject.toml"])
        dependency_instruction = (
            "2. Prefer removing third-party dependencies and using the Python standard library.\n"
            "3. If a third-party dependency is absolutely necessary, declare it and verify import in an isolated environment.\n"
            "4. Rerun the failing command after editing."
        )
    return RepairAction(
        category="dependency_unavailable",
        signature="dependency_unavailable",
        reason="The implementation depends on unavailable or undeclared third-party packages.",
        target_files=target_files,
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[command] if command else [],
        instruction=(
            "The implementation depends on unavailable or undeclared third-party packages.\n\n"
            "Required next action:\n"
            "1. Do not retry system pip install.\n"
            f"{dependency_instruction}"
        ),
        initial_inspection_budget=1,
    )


def plan_syntax_error(*, output: str, command: str) -> RepairAction | None:
    if "SyntaxError:" not in output and "IndentationError:" not in output:
        return None
    paths = infer_python_traceback_files(output, include_tests=True)
    match = SYNTAX_ERROR_FILE_RE.search(output)
    path = paths[-1] if paths else (normalize_workspace_relative_path(match.group("path")) if match else inferred_source_targets(output, command)[0])
    rerun = command or "python3 -m py_compile " + path
    return RepairAction(
        category="syntax_error",
        signature=f"syntax_error:{path}",
        reason=f"Python syntax failed in {path}.",
        target_files=[path],
        allowed_tools=TARGETED_REPAIR_ALLOWED_TOOLS,
        forbidden_tools=TARGETED_REPAIR_FORBIDDEN_TOOLS,
        rerun_commands=[rerun],
        instruction=(
            f"The command failed due to a syntax error in `{path}`.\n\n"
            "Required next action:\n"
            f"1. Read `{path}` around the reported line.\n"
            "2. Edit only the broken syntax or indentation.\n"
            "3. Do not continue unrelated diagnosis.\n"
            f"4. Rerun exactly: `{rerun}`."
        ),
    )


def format_repair_action(action: RepairAction, repeated_count: int = 1) -> str:
    lines = [
        f"Targeted repair required: {action.category}",
        f"Signature: {action.signature}",
        "",
        action.instruction,
    ]
    if action.failure_class:
        lines.append(f"Failure class: {action.failure_class}")
    if action.producer_semantic_result:
        lines.append(f"Producer semantic result: {action.producer_semantic_result}")
    if repeated_count >= 2:
        lines.extend(
            [
                "",
                "This same failure has occurred again.",
                "Do not continue diagnosis.",
                "Apply a direct minimal patch to the target file now.",
            ]
        )
    if repeated_count >= 3:
        lines.extend(
            [
                "",
                "Escalation: rewrite the relevant target file or function cleanly instead of applying another small patch.",
            ]
        )
    return "\n".join(lines)


def normalize_workspace_relative_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    elif "/workspace/" in normalized:
        normalized = normalized.rsplit("/workspace/", 1)[1]
    normalized = normalized.lstrip("./")
    normalized = posixpath.normpath(normalized)
    return "" if normalized == "." else normalized


def infer_python_traceback_files(output: str, *, include_tests: bool = False) -> list[str]:
    paths: list[str] = []
    for match in TRACEBACK_FILE_RE.finditer(output):
        raw_path = match.group("path")
        if raw_path.startswith("/") and not raw_path.startswith("/workspace/"):
            continue
        path = normalize_workspace_relative_path(raw_path)
        if path.startswith("tests/") and not include_tests:
            continue
        if path not in paths:
            paths.append(path)
    return paths


def inferred_source_targets(output: str, command: str) -> list[str]:
    candidates = infer_python_traceback_files(output)
    if not candidates:
        candidates = python_files_from_command(command)
    if not candidates:
        candidates = ["main.py"]
    return unique_preserving_order(candidates)


def inferred_test_targets(output: str, command: str) -> list[str]:
    candidates = [path for path in infer_python_traceback_files(output, include_tests=True) if path.startswith("tests/")]
    if candidates:
        return unique_preserving_order(candidates)
    if "discover -s tests" in command or " tests" in f" {command}":
        return ["tests/test_app.py"]
    return ["test_app.py"]


def infer_named_fixture_files(output: str, command: str) -> list[str]:
    combined = f"{output}\n{command}"
    paths: list[str] = []
    for match in re.finditer(r"\b[\w./-]*(?:fixtures?|samples?)/[\w./-]+\.(?:html|json|csv|txt|xml)\b", combined):
        path = normalize_workspace_relative_path(match.group(0))
        if path and path not in paths:
            paths.append(path)
    return paths


def python_files_from_command(command: str) -> list[str]:
    paths: list[str] = []
    for token in command.split():
        cleaned = token.strip("'\"")
        if cleaned.endswith(".py"):
            path = normalize_workspace_relative_path(cleaned)
            if path and path not in paths:
                paths.append(path)
    return paths


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def default_hint_for_fields(fields: list[str]) -> str:
    lines: list[str] = []
    for field in fields:
        default = FIELD_DEFAULT_HINTS.get(field)
        if default is not None:
            lines.append(f"- {field}: default {default}")
    return "\n".join(lines)


def clean_assertion_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def assertion_field_name(output: str) -> str:
    search_text = output.split("AssertionError:", 1)[0]
    for pattern in (
        r"\bfirst\[['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"]\]",
        r"\brepo\[['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"]\]",
        r"\brecord\[['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"]\]",
        r"\bitem\[['\"](?P<field>[A-Za-z_][A-Za-z0-9_]*)['\"]\]",
    ):
        matches = list(re.finditer(pattern, search_text))
        if matches:
            return matches[-1].group("field")
    return ""


def stable_signature_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80].strip("_")
    return fragment or "value"
