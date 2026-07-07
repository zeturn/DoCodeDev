from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
import re


FILE_REF_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|go|rs|md|json|toml|yaml|yml|txt|csv|html)\b")
NUMBERED_COMMAND_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")


@dataclass(frozen=True, slots=True)
class TaskContract:
    must_modify_files: list[str] = field(default_factory=list)
    must_run_commands: list[str] = field(default_factory=list)
    forbidden_finish_conditions: list[str] = field(default_factory=list)


def task_contract_from_instruction(instruction: str) -> TaskContract:
    files = unique_preserving_order(
        path
        for match in FILE_REF_RE.finditer(instruction or "")
        if (path := normalize_contract_file(match.group(0))) and contract_file_allowed(path)
    )
    if is_crawler_instruction(instruction):
        files = unique_preserving_order(["crawler.py", *files])
    explicit_commands = verification_commands_from_instruction(instruction)
    commands = unique_preserving_order([*suggested_commands(files), *explicit_commands])
    if is_crawler_instruction(instruction) and not explicit_commands:
        crawler_defaults = [
            "python3 -m unittest discover -s tests",
            "python3 crawler.py --preflight",
            "python3 crawler.py --dry-run",
        ]
        commands = unique_preserving_order([*commands, *crawler_defaults])
    forbidden = [
        "Do not call final_candidate until git_status shows at least one modified file.",
        "Do not finish with a clean git status; produce a non-empty git diff first.",
    ]
    return TaskContract(must_modify_files=files, must_run_commands=commands, forbidden_finish_conditions=forbidden)


def normalize_contract_file(value: str) -> str:
    return value.strip().strip("`'\"").strip("./")


def contract_file_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    if not normalized:
        return False
    if normalized.startswith((".araneae/", "araneae/")):
        return False
    if normalized.endswith(".jsonl"):
        return False
    if normalized.startswith(("http:/", "https:/")):
        return False
    return True


def suggested_commands(files: list[str]) -> list[str]:
    commands: list[str] = []
    file_names = {path.rsplit("/", 1)[-1] for path in files}
    if "calculator.py" in file_names:
        commands.append("python3 -m unittest discover -s tests")
    if "cli.py" in file_names:
        commands.append("python3 cli.py --name Ada")
    return commands


def verification_commands_from_instruction(instruction: str) -> list[str]:
    commands: list[str] = []
    in_verification_block = False
    for raw_line in (instruction or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        heading = lowered.lstrip("- ").rstrip(":")
        if verification_heading(heading):
            in_verification_block = True
            continue
        if in_verification_block:
            command = normalize_command_line(line)
            if command and command_like(command):
                commands.append(command)
                continue
            if line and line.endswith(":"):
                in_verification_block = verification_heading(line.lower().lstrip("- ").rstrip(":"))
                continue
            if line and not command_like(command):
                in_verification_block = False
        if "verify with:" not in lowered and "suggested verification commands:" not in lowered:
            continue
        if lowered.startswith("semantic checks:") or lowered.startswith("- semantic checks:"):
            continue
        _, value = line.split(":", 1)
        command = normalize_command_line(value)
        if command and command_like(command):
            commands.append(command)
    return commands[:8]


def normalize_command_line(line: str) -> str:
    text = line.strip().strip("`")
    if text.startswith("- "):
        text = text[2:].strip().strip("`")
    numbered = NUMBERED_COMMAND_RE.match(text)
    if numbered:
        text = numbered.group(1).strip().strip("`")
    return text.strip(" -`")


def verification_heading(heading: str) -> bool:
    text = heading.strip().lower()
    if text in {"verification commands", "suggested verification commands"}:
        return True
    return "verification commands" in text and any(word in text for word in ("run", "exact", "success", "before", "pass"))


def command_like(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    parts = text.split()
    first = parts[0]
    if first == "git":
        if len(parts) >= 3 and parts[2] in {"is", "should", "must"}:
            return False
        return len(parts) >= 2 and parts[1] in {"status", "diff", "show", "log"}
    return first in {"python", "python3", "pytest", "npm", "node", "go", "cargo", "git", "ruff", "mypy", "make", "bash", "sh", "echo", "grep"}


def unique_preserving_order(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def is_crawler_instruction(instruction: str) -> bool:
    lowered = (instruction or "").lower()
    return any(keyword in lowered for keyword in ("crawler", "scraper", "scrape", "爬虫", "抓取", "采集", "trending"))
