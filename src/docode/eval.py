from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    name: str
    status: str
    success: bool
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    failure_reason: str | None = None
    verification_plan_failures: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalReport:
    total: int
    succeeded: int
    failed: int
    success_rate: float
    iterations: int
    tool_calls: int
    tokens: int
    cost: float
    failure_reasons: dict[str, int]
    verification_plan_failures: dict[str, int]
    cases: list[EvalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "cost": self.cost,
            "failure_reasons": self.failure_reasons,
            "verification_plan_failures": self.verification_plan_failures,
            "cases": [asdict(case) for case in self.cases],
        }


def run_eval(fixtures_dir: Path) -> EvalReport:
    cases = [load_eval_case(path) for path in sorted(fixtures_dir.glob("*.json"))]
    total = len(cases)
    succeeded = sum(1 for case in cases if case.success)
    failed = total - succeeded
    return EvalReport(
        total=total,
        succeeded=succeeded,
        failed=failed,
        success_rate=(succeeded / total) if total else 0.0,
        iterations=sum(case.iterations for case in cases),
        tool_calls=sum(case.tool_calls for case in cases),
        tokens=sum(case.tokens for case in cases),
        cost=sum(case.cost for case in cases),
        failure_reasons=count_values(case.failure_reason for case in cases if case.failure_reason),
        verification_plan_failures=count_values(failure for case in cases for failure in case.verification_plan_failures),
        cases=cases,
    )


def write_eval_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_eval_case(path: Path) -> EvalCaseResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"eval case must be an object: {path}")
    status = str(data.get("status") or "")
    success = bool(data.get("success", status.lower() in {"succeeded", "success", "passed"}))
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    verification = data.get("verification") if isinstance(data.get("verification"), dict) else {}
    return EvalCaseResult(
        name=str(data.get("name") or path.stem),
        status=status or ("succeeded" if success else "failed"),
        success=success,
        iterations=int_or_zero(data.get("iterations")),
        tool_calls=int_or_zero(data.get("tool_calls")),
        tokens=int_or_zero(data.get("tokens") or usage.get("total_tokens") or usage.get("tokens")),
        cost=float_or_zero(data.get("cost") or usage.get("cost")),
        failure_reason=str(data.get("failure_reason") or "") or None,
        verification_plan_failures=verification_failures(verification),
    )


def verification_failures(verification: dict[str, Any]) -> list[str]:
    fixes = verification.get("required_fixes") or verification.get("verification_plan_failures") or []
    if isinstance(fixes, str):
        return [fixes]
    if isinstance(fixes, list):
        return [str(fix) for fix in fixes if str(fix)]
    return []


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
