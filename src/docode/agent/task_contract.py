from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections.abc import Iterable


FILE_REF_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|go|rs|md|json|toml|yaml|yml)\b")


@dataclass(frozen=True, slots=True)
class TaskContract:
    must_modify_files: list[str] = field(default_factory=list)
    must_run_commands: list[str] = field(default_factory=list)
    forbidden_finish_conditions: list[str] = field(default_factory=list)


def task_contract_from_instruction(instruction: str) -> TaskContract:
    files = unique_preserving_order(match.group(0).strip("./") for match in FILE_REF_RE.finditer(instruction or ""))
    commands = suggested_commands(files)
    forbidden = [
        "Do not call final_candidate until git_status shows at least one modified file.",
        "Do not finish with a clean git status; produce a non-empty git diff first.",
    ]
    return TaskContract(must_modify_files=files, must_run_commands=commands, forbidden_finish_conditions=forbidden)


def suggested_commands(files: list[str]) -> list[str]:
    commands: list[str] = []
    file_names = {path.rsplit("/", 1)[-1] for path in files}
    if "calculator.py" in file_names:
        commands.append("python3 -m unittest discover -s tests")
    if "cli.py" in file_names:
        commands.append("python3 cli.py --name Ada")
    return commands


def unique_preserving_order(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
