from __future__ import annotations

from typing import Any


def redacted_step_content(content: dict[str, Any]) -> dict[str, object]:
    payload = dict(content)
    if payload.get("type") == "tool_result":
        payload.pop("output", None)
    if "git_diff" in payload:
        git_diff = str(payload.pop("git_diff") or "")
        payload["git_diff_bytes"] = len(git_diff.encode("utf-8"))
        payload["git_diff_lines"] = len(git_diff.splitlines())
    if "git_status" in payload:
        git_status = str(payload.pop("git_status") or "")
        payload["git_status_bytes"] = len(git_status.encode("utf-8"))
        payload["git_status_lines"] = len(git_status.splitlines())
    for key in ("status", "test", "build", "lint"):
        if isinstance(payload.get(key), dict):
            check = dict(payload[key])
            output = str(check.pop("output", "") or "")
            if output:
                check["output_bytes"] = len(output.encode("utf-8"))
                check["output_lines"] = len(output.splitlines())
            payload[key] = check
    return payload
