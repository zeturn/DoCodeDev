from __future__ import annotations

import re
from pathlib import Path


def normalize_path(path: str) -> str:
    return str(path or "").strip().replace("\\", "/").lstrip("./")


def safe_workspace_path(workspace: Path, path: str) -> Path:
    normalized = normalize_path(path).lstrip("/")
    target = (workspace / normalized).resolve()
    root = workspace.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes workspace: {path}")
    return target


def python_portable_command(command: str) -> str:
    return re.sub(r"(?<![\w.-])python3(?=\s|$)", "python", command) if __import__("os").name == "nt" else command
