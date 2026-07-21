"""Aggregation and suite output for the evaluation harness.

Produces ``manifest.json``, ``summary.json``, ``results.jsonl`` and
``report.md``. Every JSON artifact carries ``schema_version``,
``generated_at`` and ``suite_run_id``. No secrets are written: provider base
URLs are redacted and request bodies are never serialized.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docode.eval.models import (
    STRICT_PASSING,
    EXPECTED_OUTCOME_PASSING,
    Outcome,
    RunResult,
)

SUITE_SCHEMA_VERSION = 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def aggregate_results(
    run_results: list[RunResult],
    *,
    suite_run_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or _utcnow()
    total = len(run_results)
    strict_passed = sum(1 for r in run_results if r.outcome in STRICT_PASSING)
    expected_passed = sum(1 for r in run_results if r.outcome in EXPECTED_OUTCOME_PASSING)

    def _count(outcome_value: str) -> int:
        return sum(1 for r in run_results if r.outcome == outcome_value)

    by_outcome: dict[str, int] = {}
    for value in Outcome.__members__.values():
        c = _count(value.value)
        if c:
            by_outcome[value.value] = c

    false_success = sum(1 for r in run_results if r.false_success)
    false_failure = sum(1 for r in run_results if r.false_failure)

    case_rows: dict[str, dict[str, Any]] = {}
    for r in run_results:
        row = case_rows.setdefault(
            r.case_id,
            {
                "case_id": r.case_id,
                "runs": 0,
                "strict_passed": 0,
                "expected_outcome_passed": 0,
                "outcomes": {},
            },
        )
        row["runs"] += 1
        if r.outcome in STRICT_PASSING:
            row["strict_passed"] += 1
        if r.outcome in EXPECTED_OUTCOME_PASSING:
            row["expected_outcome_passed"] += 1
        row["outcomes"][r.outcome or "unknown"] = row["outcomes"].get(r.outcome or "unknown", 0) + 1

    avg_iterations = (sum(r.iterations for r in run_results) / total) if total else 0.0
    avg_tool_calls = (sum(r.tool_call_count for r in run_results) / total) if total else 0.0
    avg_elapsed = (sum(r.elapsed_seconds for r in run_results) / total) if total else 0.0
    total_tokens = sum(r.total_tokens for r in run_results)
    total_cost = sum(r.estimated_cost for r in run_results)

    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "suite_run_id": suite_run_id,
        "total_runs": total,
        "strict_pass_rate": (strict_passed / total) if total else 0.0,
        "expected_adjusted_pass_rate": (expected_passed / total) if total else 0.0,
        "strict_passed": strict_passed,
        "expected_outcome_passed": expected_passed,
        "by_outcome": by_outcome,
        "false_success_count": false_success,
        "false_failure_count": false_failure,
        "agent_failure_count": _count(Outcome.AGENT_FAILURE.value),
        "checker_failure_count": _count(Outcome.CHECKER_FAILURE.value),
        "provider_failure_count": _count(Outcome.PROVIDER_FAILURE.value),
        "decision_parse_failure_count": _count(Outcome.DECISION_PARSE_FAILURE.value),
        "transport_failure_count": _count(Outcome.DOBOX_TRANSPORT_FAILURE.value),
        "infrastructure_failure_count": _count(Outcome.INFRASTRUCTURE_FAILURE.value),
        "harness_failure_count": _count(Outcome.HARNESS_FAILURE.value),
        "no_progress_count": _count(Outcome.NO_PROGRESS.value),
        "budget_exceeded_count": _count(Outcome.BUDGET_EXCEEDED.value),
        "average_iterations": avg_iterations,
        "average_tool_calls": avg_tool_calls,
        "average_elapsed_seconds": avg_elapsed,
        "total_tokens": total_tokens,
        "estimated_total_cost": total_cost,
        "cases": case_rows,
    }


def suite_exit_code(summary: dict[str, Any], *, harness_valid: bool, infra_fail_closed: bool) -> int:
    """Map aggregate results to the runner exit code.

    0 = every case met its expected outcome
    1 = at least one genuine Agent/Runtime failure
    2 = configuration / infrastructure fail-closed
    3 = harness or fixture invalid
    """
    if not harness_valid:
        return 3
    if infra_fail_closed:
        return 2
    # A genuine (non-expected) failure means at least one strict-failing run.
    strict_passed = summary.get("strict_passed", 0)
    expected_passed = summary.get("expected_outcome_passed", 0)
    total = summary.get("total_runs", 0)
    if total > 0 and (strict_passed + (expected_passed - strict_passed)) == total:
        return 0
    # Distinguish infra/provider failures (exit 2 semantics) from agent failures.
    non_agent = (
        summary.get("provider_failure_count", 0)
        + summary.get("decision_parse_failure_count", 0)
        + summary.get("transport_failure_count", 0)
        + summary.get("infrastructure_failure_count", 0)
        + summary.get("harness_failure_count", 0)
    )
    if non_agent > 0 and strict_passed + expected_passed == 0:
        return 2
    return 1


def write_suite_outputs(
    output_dir: Path,
    *,
    manifests: dict[str, Any],
    run_results: list[RunResult],
    suite_run_id: str,
    generated_at: str | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or _utcnow()
    summary = aggregate_results(run_results, suite_run_id=suite_run_id, generated_at=generated_at)

    manifest_doc: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "suite_run_id": suite_run_id,
        "cases": [
            {
                "id": m.id,
                "title": m.title,
                "category": m.category,
                "language": m.language,
                "expected_terminal": m.expected_terminal,
                "tags": list(m.tags),
            }
            for m in manifests.values()
        ],
    }

    paths: dict[str, Path] = {}
    (output_dir / "manifest.json").write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["manifest"] = output_dir / "manifest.json"
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["summary"] = output_dir / "summary.json"

    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in run_results:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    paths["results"] = output_dir / "results.jsonl"

    report = _render_report(summary, run_results, manifests)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    paths["report"] = output_dir / "report.md"
    return paths


def _render_report(summary: dict[str, Any], run_results: list[RunResult], manifests: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Release Evaluation Harness V1 — Report")
    lines.append("")
    lines.append(f"- suite_run_id: `{summary['suite_run_id']}`")
    lines.append(f"- generated_at: {summary['generated_at']}")
    lines.append(f"- total runs: {summary['total_runs']}")
    lines.append(f"- strict pass rate: {summary['strict_pass_rate']:.3f}")
    lines.append(f"- expected-adjusted pass rate: {summary['expected_adjusted_pass_rate']:.3f}")
    lines.append("")
    lines.append("## Outcome distribution")
    for outcome, count in summary.get("by_outcome", {}).items():
        lines.append(f"- {outcome}: {count}")
    lines.append("")
    lines.append("## Case results")
    for case_id, row in summary.get("cases", {}).items():
        title = manifests.get(case_id)
        title = title.title if title else case_id
        lines.append(f"### {case_id} — {title}")
        lines.append(f"- runs: {row['runs']}")
        lines.append(f"- strict passed: {row['strict_passed']}")
        lines.append(f"- expected-outcome passed: {row['expected_outcome_passed']}")
        for outcome, count in row.get("outcomes", {}).items():
            lines.append(f"  - {outcome}: {count}")
    lines.append("")
    lines.append("## Per-run detail")
    for r in run_results:
        lines.append(
            f"- {r.case_id}#{r.run_index}: outcome={r.outcome} "
            f"job={r.job_id} project={r.project_id} artifact={r.artifact_id} "
            f"iters={r.iterations} tools={r.tool_call_count} tokens={r.total_tokens} "
            f"elapsed={r.elapsed_seconds:.1f}s"
        )
        if r.failure_reason:
            lines.append(f"    - failure: {r.failure_reason}")
    lines.append("")
    return "\n".join(lines)
