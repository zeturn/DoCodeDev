from __future__ import annotations

import difflib
import re


def format_numbered_lines(lines: list[str], start_line: int, end_line: int) -> str:
    if not lines:
        return "<empty file>"
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line))
    return "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1))


def read_line_range(content: str, start_line: int = 1, end_line: int = 120) -> tuple[str, dict[str, int]]:
    start = max(1, int_or_default(start_line, 1))
    end = max(start, int_or_default(end_line, start + 119))
    if end - start > 400:
        end = start + 400
    lines = content.splitlines()
    if not lines:
        return "<empty file>", {"start_line": start, "end_line": end, "total_lines": 0}
    actual_end = min(end, len(lines))
    return format_numbered_lines(lines, start, actual_end), {"start_line": start, "end_line": actual_end, "total_lines": len(lines)}


def read_python_symbol(content: str, symbol: str, context_lines: int = 5) -> tuple[str, dict[str, int | str]]:
    name = str(symbol or "").strip()
    if not name:
        return "symbol must be a non-empty string", {"symbol": name}
    lines = content.splitlines()
    match_index = find_python_symbol_line(lines, name)
    if match_index is None:
        candidates = find_symbol_candidates(lines, name)
        suffix = "\n\nClosest candidates:\n" + "\n".join(candidates[:10]) if candidates else ""
        return f"symbol not found: {name}{suffix}", {"symbol": name, "total_lines": len(lines)}
    body_end = find_python_block_end(lines, match_index)
    context = max(0, int_or_default(context_lines, 5))
    start = max(1, match_index + 1 - context)
    end = min(len(lines), body_end + context)
    return format_numbered_lines(lines, start, end), {
        "symbol": name,
        "definition_line": match_index + 1,
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
    }


def find_python_symbol_line(lines: list[str], symbol: str) -> int | None:
    names = [symbol]
    if "." in symbol:
        names.append(symbol.split(".")[-1])
    for name in names:
        pattern = re.compile(rf"^\s*(async\s+def|def|class)\s+{re.escape(name)}\s*[(:]")
        for index, line in enumerate(lines):
            if pattern.search(line):
                return index
    return None


def find_python_block_end(lines: list[str], start_index: int) -> int:
    if start_index < 0 or start_index >= len(lines):
        return 0
    start_line = lines[start_index]
    base_indent = len(start_line) - len(start_line.lstrip())
    last = start_index + 1
    for index in range(start_index + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            last = index + 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        last = index + 1
    return last


def find_symbol_candidates(lines: list[str], symbol: str) -> list[str]:
    pattern = re.compile(r"^\s*(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
    pairs: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            pairs.append((match.group(2), index + 1))
    names = [name for name, _ in pairs]
    close = set(difflib.get_close_matches(symbol.split(".")[-1], names, n=10, cutoff=0.2))
    candidates = [f"{line_no}: {name}" for name, line_no in pairs if name in close]
    return candidates or [f"{line_no}: {name}" for name, line_no in pairs[:10]]


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
