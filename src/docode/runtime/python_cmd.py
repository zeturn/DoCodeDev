from __future__ import annotations

import sys


def local_python_executable() -> str:
    return sys.executable or "python"


def local_python_command_args(*args: str) -> list[str]:
    return [local_python_executable(), *args]
