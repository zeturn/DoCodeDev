from __future__ import annotations


def changed_files_from_diff(diff: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if meaningful_change_path(path) and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def meaningful_change_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    parts = normalized.split("/")
    return not (
        normalized in {".docode_probe", ".docode_probe_api"}
        or normalized.startswith(".docode_probe")
        or "__pycache__" in parts
        or normalized.endswith((".pyc", ".pyo"))
        or normalized.startswith(".git/")
    )
