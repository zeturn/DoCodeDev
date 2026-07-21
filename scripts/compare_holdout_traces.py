"""Compare a passing holdout trace against a failing holdout trace.

Locates the first semantic divergence between two deterministic holdout runs. Volatile
fields (timestamps, PIDs, temporary paths, random job/artifact ids) are ignored.

Comparison priority (first divergence wins):
    scripted action -> ToolResult exit code -> ToolResult metadata -> changed paths
    -> edit epoch -> TaskGraph evidence -> command execution -> scheduler freshness
    -> finalization blocker -> consecutive failure count

Usage:
    python scripts/compare_holdout_traces.py passing-trace.json failing-trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


VOLATILE_KEYS = {"timestamp", "pid", "job_id", "artifact_id", "workspace_hash_before", "workspace_hash_after", "output"}


def _norm_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./").lstrip("/")


def _event_signature(event: dict[str, Any]) -> dict[str, Any]:
    et = event.get("event_type")
    if et == "tool_started":
        args = event.get("args_normalized") or {}
        return {"event_type": et, "tool": event.get("tool"), "args": {k: _norm_path(v) if isinstance(v, str) else v for k, v in args.items()}}
    if et == "tool_completed":
        meta = event.get("metadata") or {}
        return {
            "event_type": et,
            "tool": event.get("tool"),
            "exit_code": event.get("exit_code"),
            "changed_paths_after": sorted(_norm_path(p) for p in (event.get("changed_paths_after") or [])),
            "meta_paths": sorted(_norm_path(p) for p in (meta.get("paths") or [])),
            "meta_path": _norm_path(meta.get("path")) if meta.get("path") else None,
        }
    if et == "tool_exception":
        return {"event_type": et, "tool": event.get("tool"), "exception_type": event.get("exception_type")}
    if et == "agent_state_after_tool":
        after = event.get("after") or {}
        nodes = {k: v.get("status") for k, v in (after.get("task_graph_nodes") or {}).items()}
        transitions = [
            {"node_id": t.get("node_id"), "to": t.get("to"), "evidence_refs": t.get("evidence_refs")}
            for t in (event.get("task_graph_transitions") or [])
        ]
        return {
            "event_type": et,
            "tool": event.get("tool"),
            "edit_epoch": after.get("edit_epoch"),
            "consecutive_failures": after.get("consecutive_failures"),
            "task_graph_nodes": nodes,
            "task_graph_transitions": transitions,
        }
    return {"event_type": et}


def _load_steps(trace_path: Path) -> list[dict[str, Any]]:
    candidate = trace_path.parent / "steps.json"
    if not candidate.exists():
        return []
    return json.loads(candidate.read_text(encoding="utf-8"))


def _finalization_signatures(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in steps:
        content = step.get("content") or {}
        etype = content.get("type")
        if etype in ("finalization_attempt", "pre_export_finalization", "finalization_attempt_result"):
            sig = {k: content.get(k) for k in content.keys()}
            sig["_step_type"] = etype
            out.append(sig)
    return out


def compare_events(passing: list[dict[str, Any]], failing: list[dict[str, Any]]) -> dict[str, Any] | None:
    length = min(len(passing), len(failing))
    for i in range(length):
        ps = _event_signature(passing[i])
        fs = _event_signature(failing[i])
        if ps != fs:
            return {
                "location": "events",
                "sequence": i,
                "passing": ps,
                "failing": fs,
            }
    if len(passing) != len(failing):
        return {
            "location": "events",
            "sequence": length,
            "passing": _event_signature(passing[length]) if len(passing) > length else None,
            "failing": _event_signature(failing[length]) if len(failing) > length else None,
            "note": f"event list length differs (passing={len(passing)}, failing={len(failing)})",
        }
    return None


def compare_finalization(passing_steps: list[dict[str, Any]], failing_steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    pf = _finalization_signatures(passing_steps)
    ff = _finalization_signatures(failing_steps)
    length = min(len(pf), len(ff))
    for i in range(length):
        if pf[i] != ff[i]:
            return {
                "location": "finalization",
                "sequence": i,
                "passing": pf[i],
                "failing": ff[i],
            }
    if len(pf) != len(ff):
        return {
            "location": "finalization",
            "sequence": length,
            "passing": pf[length] if len(pf) > length else None,
            "failing": ff[length] if len(ff) > length else None,
            "note": f"finalization step count differs (passing={len(pf)}, failing={len(ff)})",
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare passing and failing holdout traces.")
    parser.add_argument("passing")
    parser.add_argument("failing")
    parser.add_argument("--passing-steps")
    parser.add_argument("--failing-steps")
    args = parser.parse_args()

    passing_trace = json.loads(Path(args.passing).read_text(encoding="utf-8"))
    failing_trace = json.loads(Path(args.failing).read_text(encoding="utf-8"))

    passing_events = passing_trace.get("events") or []
    failing_events = failing_trace.get("events") or []

    passing_steps = json.loads(Path(args.passing_steps).read_text(encoding="utf-8")) if args.passing_steps else _load_steps(Path(args.passing))
    failing_steps = json.loads(Path(args.failing_steps).read_text(encoding="utf-8")) if args.failing_steps else _load_steps(Path(args.failing))

    event_div = compare_events(passing_events, failing_events)
    final_div = compare_finalization(passing_steps, failing_steps)

    result: dict[str, Any] = {
        "passing_status": passing_trace.get("result_status"),
        "failing_status": failing_trace.get("result_status"),
        "event_divergence": event_div,
        "finalization_divergence": final_div,
    }
    if event_div is not None:
        result["first_divergence_sequence"] = event_div["sequence"]
        result["first_divergence_location"] = "events"
    elif final_div is not None:
        result["first_divergence_sequence"] = final_div["sequence"]
        result["first_divergence_location"] = "finalization"
    else:
        result["first_divergence_sequence"] = None
        result["first_divergence_location"] = None

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
