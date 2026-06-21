from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROMPT_OUTPUT_LINES = 300
DEFAULT_PROMPT_OUTPUT_BYTES = 60_000


@dataclass(frozen=True, slots=True)
class PromptOutput:
    text: str
    truncated: bool
    original_lines: int
    original_bytes: int


def prompt_safe_output(
    output: str,
    *,
    max_lines: int = DEFAULT_PROMPT_OUTPUT_LINES,
    max_bytes: int = DEFAULT_PROMPT_OUTPUT_BYTES,
) -> PromptOutput:
    original_bytes = len(output.encode("utf-8"))
    lines = output.splitlines(keepends=True)
    original_lines = len(lines)

    truncated = False
    if max_lines > 0 and len(lines) > max_lines:
        output = "".join(lines[:max_lines])
        truncated = True

    encoded = output.encode("utf-8")
    if max_bytes > 0 and len(encoded) > max_bytes:
        output = encoded[:max_bytes].decode("utf-8", errors="replace")
        truncated = True

    if truncated:
        output = output.rstrip("\n") + "\n<truncated>"

    return PromptOutput(
        text=output,
        truncated=truncated,
        original_lines=original_lines,
        original_bytes=original_bytes,
    )
