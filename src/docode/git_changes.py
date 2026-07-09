from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RUNTIME_CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
RUNTIME_CACHE_SUFFIXES = (".pyc", ".pyo")


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", str(value or ""))


def normalize_changed_path(path: str) -> str:
    normalized = strip_ansi(path).strip().replace("\\", "/")
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def ignored_runtime_artifact(path: str) -> bool:
    normalized = normalize_changed_path(path)
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    return any(part in RUNTIME_CACHE_DIRS for part in parts) or normalized.endswith(RUNTIME_CACHE_SUFFIXES)


def meaningful_change_path(path: str) -> bool:
    normalized = normalize_changed_path(path)
    return bool(
        normalized
        and normalized not in {".docode_probe", ".docode_probe_api"}
        and not normalized.startswith(".docode_probe")
        and not normalized.startswith(".git/")
        and not ignored_runtime_artifact(normalized)
    )


def parse_status_line(raw_line: str) -> tuple[str, str]:
    line = strip_ansi(raw_line).rstrip()
    if not line:
        return "", ""
    if line.startswith("?? "):
        return "??", normalize_changed_path(line[3:])
    if len(line) >= 4 and line[2] == " ":
        marker = line[:2]
        path = line[3:].strip()
    elif len(line) >= 3 and line[1] == " ":
        marker = line[:1]
        path = line[2:].strip()
    else:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return "", ""
        marker, path = parts[0], parts[1].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1].strip()
    return marker, normalize_changed_path(path)


def changed_paths_from_status(status: str) -> list[str]:
    paths: list[str] = []
    for raw_line in str(status or "").splitlines():
        marker, path = parse_status_line(raw_line)
        if path and (marker == "??" or marker.strip()) and meaningful_change_path(path) and path not in paths:
            paths.append(path)
    return paths


def filter_status_output(status: str) -> str:
    lines: list[str] = []
    for raw_line in str(status or "").splitlines():
        marker, path = parse_status_line(raw_line)
        if path and (marker == "??" or marker.strip()) and not meaningful_change_path(path):
            continue
        line = strip_ansi(raw_line).rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def filter_diff_output(diff: str) -> str:
    text = strip_ansi(diff)
    if "diff --git " not in text:
        return text
    kept: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            append_diff_block(kept, current)
            current = [line]
        elif current:
            current.append(line)
        else:
            kept.append(line)
    append_diff_block(kept, current)
    return "".join(kept)


def append_diff_block(kept: list[str], block: list[str]) -> None:
    if not block:
        return
    path = diff_block_path(block)
    if path and not meaningful_change_path(path):
        return
    kept.extend(block)


def diff_block_path(block: list[str]) -> str:
    for line in block:
        if line.startswith("diff --git "):
            tokens = line.strip().split()
            for token in reversed(tokens):
                if token.startswith("b/"):
                    return normalize_changed_path(token[2:])
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                return ""
            return normalize_changed_path(path.removeprefix("b/"))
    return ""
