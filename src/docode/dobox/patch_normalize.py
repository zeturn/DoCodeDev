from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:diff|patch|text)?\s*\n(.*)\n```$", re.DOTALL | re.IGNORECASE)
_FENCE_LEADING_RE = re.compile(r"^```(?:diff|patch|text)?\s*\n", re.IGNORECASE)
_FENCE_TRAILING_RE = re.compile(r"\n```\s*$", re.IGNORECASE)
_BLOCK_RE = re.compile(r"^\*\*\*\s+(Begin Patch|End Patch|Update File|Add File|Delete File):?\s*(.*)$")


def strip_code_fences(text: str) -> str:
    """Remove markdown ```diff ... ``` wrappers a model often emits."""
    stripped = (text or "").strip()
    fence = _FENCE_RE.match(stripped)
    if fence:
        return fence.group(1).strip()
    if stripped.startswith("```"):
        stripped = _FENCE_LEADING_RE.sub("", stripped)
        stripped = _FENCE_TRAILING_RE.sub("", stripped)
        return stripped.strip()
    return stripped


def convert_aider_patch(text: str) -> str:
    """Convert an OpenAI/Codex/Aider ``*** Begin Patch`` envelope to a unified diff.

    The model (OpenAI-style patch tooling) frequently returns this envelope
    instead of a plain ``git apply``-compatible unified diff. The body uses
    ``@@`` hunk separators, ``-``/``+``/context lines, and absolute
    ``/workspace/...`` paths. We emit a zero-context unified diff (apply with
    ``git apply --unidiff-zero``) so the exact +/- lines are matched without
    requiring original line numbers, and we strip the ``/workspace/`` prefix.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    current_path = ""
    mode = ""

    def flush() -> None:
        nonlocal current, current_path, mode
        if current is None:
            return
        lines = current
        # Unified-diff hunk counts: context/`-` lines count toward the old side,
        # context/`+` lines toward the new side. With --unidiff-zero the start
        # line numbers are ignored, but the counts must still be correct.
        old_count = sum(1 for ln in lines if ln[:1] in " -")
        new_count = sum(1 for ln in lines if ln[:1] in " +")
        if mode == "Add File":
            # New files: ensure every body line is an addition.
            body = [("+" + ln) if not ln[:1] in "+-" else ln for ln in lines]
            header = (
                f"diff --git a/{current_path} b/{current_path}\n"
                f"--- /dev/null\n+++ b/{current_path}\n"
                f"@@ -0,0 +1,{new_count} @@\n"
            )
            blocks.append(header + "\n".join(body) + ("\n" if body else ""))
        elif mode == "Delete File":
            header = (
                f"diff --git a/{current_path} b/{current_path}\n"
                f"--- a/{current_path}\n+++ /dev/null\n"
                f"@@ -1,{old_count} +0,0 @@\n"
            )
            blocks.append(header + "\n".join(lines) + ("\n" if lines else ""))
        elif mode == "Update File":
            header = (
                f"diff --git a/{current_path} b/{current_path}\n"
                f"--- a/{current_path}\n+++ b/{current_path}\n"
                f"@@ -1,{old_count} +1,{new_count} @@\n"
            )
            blocks.append(header + "\n".join(lines) + ("\n" if lines else ""))
        current = None

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m = _BLOCK_RE.match(line)
        if m:
            kind = m.group(1)
            if kind == "Begin Patch":
                continue
            if kind == "End Patch":
                flush()
                continue
            flush()
            mode = kind
            p = m.group(2).strip()
            if p.startswith("/workspace/"):
                p = p[len("/workspace/"):]
            elif p == "/workspace":
                p = ""
            current_path = p
            current = []
            continue
        if current is not None:
            if line.startswith("@@"):
                # aider/OpenAI hunk separator marker; not a content line.
                continue
            current.append(line)
    flush()
    return "\n".join(blocks)


def normalize_model_patch(patch: str) -> str:
    """Normalize a model-produced patch into something ``git apply`` can parse.

    Handles markdown code fences and OpenAI/Codex ``*** Begin Patch`` envelopes.
    Plain unified diffs pass through unchanged.
    """
    text = strip_code_fences(patch or "")
    if "*** Begin Patch" in text:
        text = convert_aider_patch(text)
    return text


__all__ = ["normalize_model_patch", "strip_code_fences", "convert_aider_patch"]
