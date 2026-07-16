"""Core data models for the deterministic holdout evaluation harness (V1).

This module is intentionally dependency-light: it only depends on the standard
library and the existing ``docode`` storage models. The outcome taxonomy and
the run/aggregate dataclasses are the single source of truth used by the
manifest loader, the fixture validator, the runner, and the metrics aggregator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ── Outcome taxonomy ───────────────────────────────────────────────────────
# These are the canonical classification labels emitted by the harness. They
# are deliberately more expressive than a single "passed/failed" so that a
# safe failure (an unsatisfiable task handled correctly) is not counted as an
# ordinary Agent failure.


class Outcome(str, Enum):
    PASSED = "passed"
    EXPECTED_OUTCOME_PASS = "expected_outcome_pass"
    AGENT_FAILURE = "agent_failure"
    CHECKER_FAILURE = "checker_failure"
    PROVIDER_FAILURE = "provider_failure"
    DECISION_PARSE_FAILURE = "decision_parse_failure"
    DOBOX_TRANSPORT_FAILURE = "dobox_transport_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_PROGRESS = "no_progress"
    HARNESS_FAILURE = "harness_failure"


# Human-readable labels for reports.
OUTCOME_LABELS: dict[str, str] = {
    Outcome.PASSED.value: "passed",
    Outcome.EXPECTED_OUTCOME_PASS.value: "expected outcome pass (safe failure)",
    Outcome.AGENT_FAILURE.value: "agent failure",
    Outcome.CHECKER_FAILURE.value: "checker failure (false success)",
    Outcome.PROVIDER_FAILURE.value: "provider failure",
    Outcome.DECISION_PARSE_FAILURE.value: "decision parse failure",
    Outcome.DOBOX_TRANSPORT_FAILURE.value: "DoBox transport failure",
    Outcome.INFRASTRUCTURE_FAILURE.value: "infrastructure failure",
    Outcome.BUDGET_EXCEEDED.value: "budget exceeded",
    Outcome.NO_PROGRESS.value: "no progress",
    Outcome.HARNESS_FAILURE.value: "harness failure",
}

# Outcomes that count toward an "expected outcome" adjusted pass rate.
EXPECTED_OUTCOME_PASSING = frozenset({Outcome.PASSED.value, Outcome.EXPECTED_OUTCOME_PASS.value})

# Outcomes that represent a genuine Agent capability failure (strict pass rate).
STRICT_PASSING = frozenset({Outcome.PASSED.value})

# Outcomes that must never be attributed to Agent capability.
NON_AGENT_OUTCOMES = frozenset(
    {
        Outcome.PROVIDER_FAILURE.value,
        Outcome.DECISION_PARSE_FAILURE.value,
        Outcome.DOBOX_TRANSPORT_FAILURE.value,
        Outcome.INFRASTRUCTURE_FAILURE.value,
        Outcome.HARNESS_FAILURE.value,
    }
)


VALID_EXPECTED_TERMINALS = frozenset({"succeeded", "failed", "blocked"})


class FixtureManifestError(ValueError):
    """Raised when a fixture manifest fails strict validation."""


# ── Hidden checker result ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    details: str = ""


@dataclass(frozen=True, slots=True)
class CheckResult:
    passed: bool
    checks: list[Check] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [asdict(check) for check in self.checks],
        }


# ── Per-run result (everything the metrics layer needs) ───────────────────


@dataclass(slots=True)
class RunResult:
    suite_run_id: str
    case_id: str
    run_index: int
    job_id: str | None = None
    project_id: str | None = None
    sandbox_id: str | None = None
    agent_session_id: str | None = None
    artifact_id: str | None = None
    provider: str | None = None
    model: str | None = None
    terminal_status: str | None = None
    outcome: str | None = None
    checker_passed: bool | None = None
    expected_terminal: str = "succeeded"
    required_commands: list[str] = field(default_factory=list)
    iterations: int = 0
    llm_decision_count: int = 0
    tool_call_count: int = 0
    tool_calls_by_type: dict[str, int] = field(default_factory=dict)
    edit_count: int = 0
    command_count: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    failure_reason: str | None = None
    no_progress_count: int = 0
    transport_errors: int = 0
    decision_parse_errors: int = 0
    # Set True when the checker passed although the job terminal was a failure
    # (implementation was actually correct) -> a false negative.
    false_failure: bool = False
    # Set True when the job succeeded but the checker failed -> false success.
    false_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Outcome classification ─────────────────────────────────────────────────


def classify_run_outcome(
    *,
    terminal_status: str | None,
    checker_passed: bool | None,
    expected_terminal: str,
    failure_reason: str | None = None,
    failure_class: str | None = None,
    failure_category: str | None = None,
    harness_error: bool = False,
) -> str:
    """Map raw run signals to a single canonical Outcome label.

    The rules follow the harness contract:

    * a harness/infrastructure error is never an Agent capability failure;
    * a job that reached the expected terminal with a passing checker is a
      success (or an expected-outcome pass for the unsatisfiable case);
    * a job that succeeded but failed the checker is a checker failure
      (false success);
    * a job that failed but whose checker proves the implementation correct is
      a false failure (still an agent failure in strict terms).
    """
    status = (terminal_status or "").lower()

    if harness_error:
        return Outcome.HARNESS_FAILURE.value

    if failure_class in ("infra_failed",) or failure_category in (
        "workspace_inconsistent",
        "provider_call_failed",
        "provider_auth_failed",
    ):
        if failure_category in ("provider_auth_failed", "provider_call_failed"):
            return Outcome.INFRASTRUCTURE_FAILURE.value
        return Outcome.INFRASTRUCTURE_FAILURE.value

    if failure_class == "model_unavailable":
        return Outcome.PROVIDER_FAILURE.value

    # Decision/parser failures surface through the runtime as parser_failed or
    # unsupported decision types.
    reason = (failure_reason or "").lower()
    if "unsupported decision type" in reason or failure_class == "parser_failed":
        return Outcome.DECISION_PARSE_FAILURE.value

    # Transport failures raised by the DoBox client are classified separately.
    if failure_class == "transport_failed" or "dobox_transport" in reason or "server disconnected" in reason:
        return Outcome.DOBOX_TRANSPORT_FAILURE.value

    if failure_class == "budget_exceeded":
        return Outcome.BUDGET_EXCEEDED.value

    if failure_class == "no_progress":
        return Outcome.NO_PROGRESS.value

    succeeded = status == "succeeded"
    checker_ok = bool(checker_passed)

    if expected_terminal == "succeeded":
        if succeeded and checker_ok:
            return Outcome.PASSED.value
        if succeeded and not checker_ok:
            return Outcome.CHECKER_FAILURE.value
        # job failed; checker may still validate the implementation.
        if (not succeeded) and checker_ok:
            return Outcome.AGENT_FAILURE.value  # counted as false_failure by caller
        return Outcome.AGENT_FAILURE.value

    # Unsatisfiable / safe-failure case.
    if checker_ok and not succeeded:
        return Outcome.EXPECTED_OUTCOME_PASS.value
    if succeeded and not checker_ok:
        # The agent "succeeded" but should not have; the checker correctly
        # rejects the fabricated success.
        return Outcome.CHECKER_FAILURE.value
    return Outcome.AGENT_FAILURE.value


def derive_false_flags(outcome: str, *, terminal_status: str | None = None, checker_passed: bool | None = None) -> tuple[bool, bool]:
    # Derived from the canonical outcome, not just terminal/checker, so the
    # expected safe-failure (unsatisfiable) case is NOT counted as a false
    # failure. A false success means the job succeeded but the checker rejected
    # the implementation (CHECKER_FAILURE). A false failure means the job failed
    # yet the checker accepted the implementation (an AGENT_FAILURE whose checker
    # actually passed).
    false_success = outcome == Outcome.CHECKER_FAILURE.value
    false_failure = (outcome == Outcome.AGENT_FAILURE.value) and bool(checker_passed)
    return false_failure, false_success
