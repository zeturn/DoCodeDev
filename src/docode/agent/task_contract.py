from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
import re


FILE_REF_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|go|rs|md|json|toml|yaml|yml|txt|csv|html)\b")
NUMBERED_COMMAND_RE = re.compile(r"^\s*\d+[.)]\s+(.+)$")
TARGET_HEADING_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:target|edit|modify)\s+files?\s*:\s*(?P<value>.*)$",
    flags=re.IGNORECASE,
)
HEREDOC_OPENER_RE = re.compile(
    r"<<(?P<strip_tabs>-)?\s*(?:'(?P<single>[^'\n]+)'|\"(?P<double>[^\"\n]+)\"|(?P<bare>[^\s;&|()<>]+))"
)
EDIT_TARGET_VERBS = {
    "add",
    "build",
    "change",
    "create",
    "edit",
    "fix",
    "implement",
    "modify",
    "refactor",
    "repair",
    "update",
}


@dataclass(frozen=True, slots=True)
class TaskContract:
    must_modify_files: list[str] = field(default_factory=list)
    must_run_commands: list[str] = field(default_factory=list)
    forbidden_finish_conditions: list[str] = field(default_factory=list)


def task_contract_from_instruction(instruction: str) -> TaskContract:
    all_files = unique_preserving_order(
        path
        for match in FILE_REF_RE.finditer(text_outside_verification_blocks(instruction))
        if (path := normalize_contract_file(match.group(0))) and contract_file_allowed(path)
    )
    files = target_files_from_instruction(instruction, all_files)
    explicit_commands = verification_commands_from_instruction(instruction)
    commands = unique_preserving_order(explicit_commands)
    forbidden = [
        "Do not call final_candidate until git_status shows at least one modified file.",
        "Do not finish with a clean git status; produce a non-empty git diff first.",
    ]
    return TaskContract(must_modify_files=files, must_run_commands=commands, forbidden_finish_conditions=forbidden)


def target_files_from_instruction(instruction: str, fallback_files: list[str]) -> list[str]:
    explicit, targets = explicit_target_block(instruction)
    if explicit:
        return targets

    targets: list[str] = []
    in_verification_block = False
    for raw_line in (instruction or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        heading = lowered.lstrip("- ").rstrip(":")
        if verification_heading(heading):
            in_verification_block = True
            continue
        if in_verification_block:
            command = normalize_command_line(line)
            if command and command_like(command):
                continue
            if line and line.endswith(":"):
                in_verification_block = verification_heading(line.lower().lstrip("- ").rstrip(":"))
                continue
            if line:
                in_verification_block = False
        matches = list(FILE_REF_RE.finditer(line))
        if not matches:
            continue
        if explicit_target_hint(lowered):
            targets.extend(
                path
                for match in matches
                if (path := normalize_contract_file(match.group(0))) and contract_file_allowed(path)
            )
            continue
        previous_target = False
        previous_end = 0
        for match in matches:
            path = normalize_contract_file(match.group(0))
            if not path or not contract_file_allowed(path):
                previous_target = False
                previous_end = match.end()
                continue
            between = line[previous_end : match.start()]
            if edit_verb_immediately_before(between) or (previous_target and conjunction_only(between)):
                targets.append(path)
                previous_target = True
            else:
                previous_target = False
            previous_end = match.end()
    return unique_preserving_order(targets or fallback_files)


def explicit_target_block(instruction: str) -> tuple[bool, list[str]]:
    """Return whether an explicit target heading exists and its declared paths."""

    lines = (instruction or "").splitlines()
    targets: list[str] = []
    explicit = False
    index = 0
    while index < len(lines):
        match = TARGET_HEADING_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        explicit = True
        inline = match.group("value").strip()
        if inline:
            targets.extend(target_paths_from_value(inline))
            index += 1
            continue

        index += 1
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.strip()
            if not line or section_heading(line) or TARGET_HEADING_RE.match(raw_line):
                break
            block_targets = target_paths_from_value(line, require_path_list=True)
            if not block_targets:
                break
            targets.extend(block_targets)
            index += 1
    return explicit, unique_preserving_order(targets)


def target_paths_from_value(value: str, *, require_path_list: bool = False) -> list[str]:
    text = value.strip().strip("`")
    text = re.sub(r"^[-*+]\s+", "", text)
    matches = list(FILE_REF_RE.finditer(text))
    if require_path_list:
        remainder = FILE_REF_RE.sub("", text)
        if not re.fullmatch(r"[\s,;`'\"()\[\]]*", remainder):
            return []
    return [
        path
        for match in matches
        if (path := normalize_contract_file(match.group(0))) and contract_file_allowed(path)
    ]


def section_heading(line: str) -> bool:
    text = line.strip().lstrip("- ").strip()
    if verification_heading(text.rstrip(":")):
        return True
    return bool(text.endswith(":"))


def explicit_target_hint(line: str) -> bool:
    return any(marker in line for marker in ("target file:", "target files:", "edit file:", "edit files:"))


def edit_verb_immediately_before(text: str) -> bool:
    words = re.findall(r"[a-zA-Z_]+", text.lower())
    return bool(words and words[-1] in EDIT_TARGET_VERBS)


def conjunction_only(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:,|and|or|\+)\s*", text, flags=re.IGNORECASE))


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
    _ = files
    return []


def verification_commands_from_instruction(instruction: str) -> list[str]:
    commands: list[str] = []
    lines = (instruction or "").splitlines()
    in_verification_block = False
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()
        lowered = line.lower()
        heading = lowered.lstrip("- ").rstrip(":")
        if verification_heading(heading):
            in_verification_block = True
            index += 1
            continue
        if in_verification_block:
            if markdown_fence(line) or not line:
                index += 1
                continue
            command = normalize_command_line(line)
            if command and command_like(command):
                delimiter = heredoc_delimiter_from_command(command)
                if delimiter is not None:
                    collected = collect_heredoc_command(lines, index, command, delimiter)
                    if collected is not None:
                        full_command, index = collected
                        commands.append(full_command)
                    else:
                        index = len(lines)
                    continue
                commands.append(command)
                index += 1
                continue
            if line and line.endswith(":"):
                in_verification_block = verification_heading(line.lower().lstrip("- ").rstrip(":"))
                index += 1
                continue
            if line and not command_like(command):
                in_verification_block = False
        if "verify with:" not in lowered and "suggested verification commands:" not in lowered:
            index += 1
            continue
        if lowered.startswith("semantic checks:") or lowered.startswith("- semantic checks:"):
            index += 1
            continue
        _, value = line.split(":", 1)
        command = normalize_command_line(value)
        if command and command_like(command):
            delimiter = heredoc_delimiter_from_command(command)
            if delimiter is not None:
                collected = collect_heredoc_command(lines, index, command, delimiter)
                if collected is not None:
                    full_command, index = collected
                    commands.append(full_command)
                    continue
                index = len(lines)
                continue
            else:
                commands.append(command)
        index += 1
    return commands[:8]


def heredoc_delimiter_from_command(command: str) -> tuple[str, bool] | None:
    match = HEREDOC_OPENER_RE.search(command)
    if match is None:
        return None
    delimiter = match.group("single") or match.group("double") or match.group("bare")
    return delimiter, bool(match.group("strip_tabs"))


def collect_heredoc_command(
    lines: list[str],
    start_index: int,
    first_line: str,
    delimiter: tuple[str, bool],
) -> tuple[str, int] | None:
    marker, strip_tabs = delimiter
    body: list[str] = [first_line]
    index = start_index + 1
    while index < len(lines):
        raw_line = lines[index]
        body.append(raw_line)
        candidate = raw_line.lstrip("\t") if strip_tabs else raw_line
        if candidate == marker:
            return "\n".join(body), index + 1
        index += 1
    return None


def strip_command_list_prefix(line: str) -> str:
    text = line.strip().strip("`")
    text = re.sub(r"^[-*+]\s+", "", text)
    numbered = NUMBERED_COMMAND_RE.match(text)
    if numbered:
        text = numbered.group(1).strip().strip("`")
    return text.strip(" -`")


def normalize_command_line(line: str) -> str:
    return strip_command_list_prefix(line)


def markdown_fence(line: str) -> bool:
    return bool(re.fullmatch(r"`{3,}(?:[A-Za-z0-9_+-]+)?", line.strip()))


def text_outside_verification_blocks(instruction: str) -> str:
    lines = (instruction or "").splitlines()
    kept: list[str] = []
    in_verification_block = False
    for raw_line in lines:
        line = raw_line.strip()
        heading = line.lower().lstrip("- ").rstrip(":")
        if verification_heading(heading):
            in_verification_block = True
            continue
        if in_verification_block:
            if line and line.endswith(":") and not verification_heading(heading):
                in_verification_block = False
                kept.append(raw_line)
            continue
        kept.append(raw_line)
    return "\n".join(kept)


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
