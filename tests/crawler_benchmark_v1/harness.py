from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from tests.crawler_benchmark_v1.definitions import CrawlerCase, expected_rows


EDIT_TOOLS = {"write_file", "edit_file", "replace_in_file", "apply_patch"}
READ_TOOLS = {"read_file", "read_file_range", "read_symbol", "search", "list_files", "fetch_url"}


def materialize_workspace(case: CrawlerCase, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    if case.scaffold is not None:
        target = destination / case.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(case.scaffold, encoding="utf-8")
    for relative, content in case.extra_files:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return destination


def metrics(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url + "/__metrics", timeout=5) as response:
        return json.load(response)


def reset(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/__reset", timeout=5) as response:
        response.read()


def variant_source(case: CrawlerCase, base_url: str, variant: bool) -> str:
    if not case.controlled:
        return case.source_path
    separator = "&" if "?" in case.source_path else "?"
    suffix = f"{separator}variant=2" if variant else ""
    return base_url + case.source_path + suffix


def expected_requests(case: CrawlerCase, *, variant: bool) -> list[str]:
    suffix = "?variant=2" if variant else ""
    if case.name == "opal_canopy":
        return ["/aurora/cards" + suffix]
    if case.name == "flint_harbor":
        return ["/kiln/observations" + suffix]
    if case.name == "marble_tide":
        return ["/ledger/start" + suffix, "/ledger/next" + suffix]
    if case.name == "violet_prism":
        return ["/prism/feed" + suffix]
    if case.name == "copper_orbit":
        cursor = "phase-violet-2" if variant else "phase-amber-2"
        tail = "&variant=2" if variant else ""
        return [f"/orbit/measurements?cursor={tail}", f"/orbit/measurements?cursor={cursor}{tail}"]
    return []


def validate_controlled_artifact(
    case: CrawlerCase,
    artifact: Path,
    *,
    base_url: str,
    variant: bool,
    observed_metrics: dict[str, Any],
) -> list[str]:
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"artifact is not valid JSON: {exc}"]
    return validate_controlled_payload(
        case,
        payload,
        base_url=base_url,
        variant=variant,
        observed_metrics=observed_metrics,
    )


def validate_controlled_payload(
    case: CrawlerCase,
    payload: Any,
    *,
    base_url: str,
    variant: bool,
    observed_metrics: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected = expected_rows(case, variant=variant, base_url=base_url)
    if payload != expected:
        failures.append(f"payload mismatch: expected {expected!r}, got {payload!r}")
    requests = expected_requests(case, variant=variant)
    if observed_metrics.get("requests") != requests:
        failures.append(f"request sequence mismatch: expected {requests!r}, got {observed_metrics.get('requests')!r}")
    if observed_metrics.get("count") != len(requests):
        failures.append(f"request count mismatch: expected {len(requests)}, got {observed_metrics.get('count')!r}")
    return failures


def validate_live_payload(payload: Any) -> list[str]:
    if not isinstance(payload, list) or len(payload) < 5:
        return [f"expected at least five records, got {len(payload) if isinstance(payload, list) else type(payload).__name__}"]
    fields = {"headline", "link", "published", "source", "summary"}
    failures = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or set(row) != fields:
            failures.append(f"record {index} has wrong schema")
            continue
        if not all(isinstance(row[key], str) and row[key].strip() for key in ("headline", "link", "published", "source")):
            failures.append(f"record {index} has empty required values")
        if not row["link"].startswith(("http://", "https://")):
            failures.append(f"record {index} link is not absolute")
        if row["summary"] is not None and not isinstance(row["summary"], str):
            failures.append(f"record {index} summary has wrong type")
    return failures


def validate_live_artifact(artifact: Path) -> list[str]:
    try:
        return validate_live_payload(json.loads(artifact.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"artifact is not valid JSON: {exc}"]


def run_collector(workspace: Path, case: CrawlerCase, source_url: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, case.target, source_url, case.output],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def summarize_steps(steps: list[Any], initial_files: set[str]) -> dict[str, Any]:
    contents = [step.content for step in steps]
    tool_results = [content for content in contents if content.get("type") == "tool_result"]
    edits = [content for content in tool_results if content.get("tool") in EDIT_TOOLS and content.get("exit_code") == 0]
    reads = [content for content in tool_results if content.get("tool") in READ_TOOLS and content.get("exit_code") == 0]
    commands = [content for content in tool_results if content.get("tool") == "run_command"]
    first_edit = min((step.step_index for step in steps if step.content in edits), default=None)
    first_read = min((step.step_index for step in steps if step.content in reads), default=None)
    whole_file_rewrite = any(
        content.get("tool") == "write_file"
        and str((content.get("metadata") or {}).get("path", "")).replace("\\", "/") in initial_files
        for content in edits
    )
    return {
        "iterations": len([content for content in contents if content.get("type") == "llm_decision"]),
        "llm_decisions": len([content for content in contents if content.get("type") == "llm_decision"]),
        "tool_calls": len([content for content in contents if content.get("type") == "tool_call"]),
        "commands_run": len(commands),
        "successful_commands": len([content for content in commands if content.get("exit_code") == 0]),
        "repair_actions": len([content for content in contents if content.get("type") == "repair_action"]),
        "successful_edits": len(edits),
        "successful_reads": len(reads),
        "final_candidate_attempted": any(
            content.get("type") == "auto_final_candidate"
            or (content.get("type") == "llm_decision" and content.get("decision_type") == "final_candidate")
            for content in contents
        ),
        "read_before_edit": first_read is not None and (first_edit is None or first_read < first_edit),
        "whole_file_rewrite": whole_file_rewrite,
    }


def command_results(steps: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "command": str((step.content.get("metadata") or {}).get("command") or ""),
            "exit_code": step.content.get("exit_code"),
            "summary": str(step.content.get("summary") or step.content.get("output") or "")[-800:],
        }
        for step in steps
        if step.content.get("type") == "tool_result" and step.content.get("tool") == "run_command"
    ]


def secret_values() -> set[str]:
    values = set()
    for name in ("DOCODE_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY", "DOCODE_DOBOX_TOKEN", "DOCODE_APICRED_TOKEN"):
        if os.getenv(name):
            values.add(os.environ[name])
    return values


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secret_values():
            result = result.replace(secret, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", result)
    return value
