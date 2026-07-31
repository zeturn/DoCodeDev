"""Stable, comparison-safe action keys for every agent action.

Every key must be deterministic, platform-portable, and free of
timestamps/UUIDs/iteration numbers.  Identical twin actions MUST
produce identical keys so that the no-progress tracker can reliably
detect repeated non-progress loops.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from docode.agent.workflow import normalize_command


# ── helpers ──────────────────────────────────────────────────────────────

def _sha256_hex(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_path(path: str) -> str:
    return str(path).strip().replace("\\", "/")


# ── tool action keys ────────────────────────────────────────────────────


def tool_action_key(tool_name: str, args: dict[str, object]) -> str:
    """Build a stable action key for any tool call."""
    path = _normalize_path(str(args.get("path", "")))
    path_s = _normalize_path(str(args.get("start_path", "")))
    symbol = str(args.get("symbol", "")).strip()

    if tool_name == "read_file":
        return f"read_file:{path}"

    if tool_name == "read_file_range":
        start = _int_or(args.get("start_line"), 1)
        end = _int_or(args.get("end_line"), 120)
        return f"read_file_range:{path}:{start}:{end}"

    if tool_name == "read_symbol":
        return f"read_symbol:{path}:{symbol}" if symbol else f"read_symbol:{path}"

    if tool_name == "search":
        query = str(args.get("query", "")).strip()
        return f"search:{path}:{_sha256_hex(query)}"

    if tool_name == "list_files":
        return f"list_files:{path}"

    if tool_name == "run_command":
        command = normalize_command(str(args.get("command", "")))
        return f"run_command:{command}"

    if tool_name == "write_file":
        content = str(args.get("content", ""))
        return f"write_file:{path}:{_sha256_hex(content)}"

    if tool_name == "edit_file":
        return _edit_style_key(
            "edit_file", path,
            str(args.get("old_text", "")),
            str(args.get("new_text", "")),
        )

    if tool_name == "replace_in_file":
        return _edit_style_key(
            "replace_in_file", path,
            str(args.get("find", "")),
            str(args.get("replace", "")),
        )

    if tool_name == "apply_patch":
        patch = str(args.get("patch", ""))
        return f"apply_patch:{_sha256_hex(patch)}"

    # catch-all: avoid accidentally making two different calls look the same
    args_repr = json.dumps(
        {k: str(v) for k, v in sorted(args.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{tool_name}:{_sha256_hex(args_repr)}"


# ── controller / synthetic keys ──────────────────────────────────────────


def final_candidate_action_key(
    summary: str,
    no_test_reason: str | None = None,
) -> str:
    """Stable key for a final-candidate submission."""
    payload = {"summary": summary.strip()}
    if no_test_reason and no_test_reason.strip():
        payload["no_test_reason"] = no_test_reason.strip()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"final_candidate:{_sha256_hex(encoded)}"


def rejected_action_key(reason: str, requested_action_key: str) -> str:
    """Stable key for a decision-rejection."""
    return f"decision_rejected:{reason.strip()}:{requested_action_key}"


# ── internal helpers ─────────────────────────────────────────────────────


def _edit_style_key(
    tool: str,
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    change = f"{old_text}→{new_text}"
    return f"{tool}:{path}:{_sha256_hex(change)}"


def _int_or(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
