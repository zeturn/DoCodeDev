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


# Keywords that indicate a genuine transport/protocol failure (connect/read/
# write timeout or protocol error) raised by the DoBox client. A structured
# HTTP 500 from workspace provisioning is NOT a transport failure.
_TRANSPORT_KEYWORDS = (
    "doboxtransporterror",
    "remoteprotocolerror",
    "connecterror",
    "readerror",
    "writeerror",
    "server disconnected",
)

# Keywords that indicate a provider-side failure (auth, model availability,
# provider HTTP errors) before a valid decision was produced.
_PROVIDER_KEYWORDS = (
    "provider auth",
    "provider authentication",
    "model unavailable",
    "apicred_authorize_failed",
    "authentication failed",
    "unauthorized",
    "401 ",
    "403 ",
)


@dataclass(frozen=True)
class FailureSignals:
    """Centralized structured failure signals extracted from a finished run.

    ``run_case`` must build this once via :func:`extract_failure_signals` rather
    than scattering ad-hoc string checks across the runner. Structured signals
    take priority over free-text ``failure_reason`` matching.
    """

    failure_class: str | None = None
    failure_category: str | None = None
    failure_stage: str | None = None
    exception_type: str | None = None
    workspace_created: bool = False
    llm_started: bool = False
    tool_execution_started: bool = False
    transport_error: bool = False
    provider_error: bool = False
    decision_parse_error: bool = False


def _summarize_steps(steps: list[Any]) -> tuple[bool, bool, bool, bool]:
    """Return (llm_started, tool_execution_started, transport_error, decision_parse_error)."""
    llm_started = False
    tool_execution_started = False
    transport_error = False
    decision_parse_error = False
    for step in steps or []:
        content = step.content if hasattr(step, "content") else (step.get("content", {}) if isinstance(step, dict) else {})
        kind = step.kind if hasattr(step, "kind") else (step.get("kind") if isinstance(step, dict) else None)
        if not isinstance(content, dict):
            content = {}
        stype = str(content.get("type") or "")
        if stype == "llm_decision":
            llm_started = True
        if kind == "tool" and stype in ("tool_call", "tool_result"):
            tool_execution_started = True
        if stype == "transport_error" or "transport" in str(content.get("error") or "").lower():
            transport_error = True
        if "unsupported decision type" in str(content).lower():
            decision_parse_error = True
    return llm_started, tool_execution_started, transport_error, decision_parse_error


def extract_failure_signals(
    job: Any,
    steps: list[Any],
    *,
    harness_error: bool = False,
    harness_exception_type: str | None = None,
) -> FailureSignals:
    """Extract structured failure signals from a finished job + its steps.

    This is the single place that interprets the job's structured fields
    (``terminal_result.category``, ``dobox_project_id``) and the step stream.
    """
    project_id = getattr(job, "dobox_project_id", None)
    workspace_created = bool(project_id)
    llm_started, tool_execution_started, transport_error, decision_parse_error = _summarize_steps(steps)

    terminal = getattr(job, "terminal_result", None)
    failure_category = None
    if isinstance(terminal, dict):
        cat = terminal.get("category")
        failure_category = cat.value if hasattr(cat, "value") else cat
    elif terminal is not None:
        cat = getattr(terminal, "category", None)
        if hasattr(cat, "value"):
            failure_category = cat.value
        elif isinstance(cat, str):
            failure_category = cat

    failure_reason = getattr(job, "failure_reason", None) or ""
    reason_l = failure_reason.lower()

    provider_error = failure_category == "provider_failure" or any(k in reason_l for k in _PROVIDER_KEYWORDS)
    # Transport only for genuine transport/protocol errors, never for a
    # structured HTTP 500 workspace-provisioning error.
    transport_error = transport_error or any(k in reason_l for k in _TRANSPORT_KEYWORDS)

    if not workspace_created and not llm_started and not tool_execution_started:
        failure_stage = "provisioning"
    elif not llm_started and not tool_execution_started:
        failure_stage = "pre_decision"
    elif not tool_execution_started:
        failure_stage = "decision_only"
    else:
        failure_stage = "execution"

    return FailureSignals(
        failure_class=None,
        failure_category=failure_category,
        failure_stage=failure_stage,
        exception_type=harness_exception_type if harness_error else None,
        workspace_created=workspace_created,
        llm_started=llm_started,
        tool_execution_started=tool_execution_started,
        transport_error=transport_error,
        provider_error=provider_error,
        decision_parse_error=decision_parse_error,
    )


def classify_run_outcome(
    *,
    terminal_status: str | None,
    checker_passed: bool | None,
    expected_terminal: str,
    failure_reason: str | None = None,
    signals: FailureSignals | None = None,
    harness_error: bool = False,
    failure_class: str | None = None,
    failure_category: str | None = None,
) -> str:
    """Map raw run signals to a single canonical Outcome label.

    Classification priority:
      1. harness_failure
      2. infrastructure_failure
      3. provider_failure
      4. decision_parse_failure
      5. dobox_transport_failure
      6. budget_exceeded
      7. no_progress
      8. expected outcome / passed / checker failure / agent failure

    Structured ``signals`` take priority over free-text ``failure_reason``.
    The legacy ``failure_class``/``failure_category`` kwargs are retained for
    backward compatibility and are only consulted when ``signals`` is omitted.
    """
    if signals is None:
        # Backward-compatible path: synthesize minimal signals. We deliberately
        # do NOT assume a provisioning failure here (the generic provisioning
        # rule requires real structured signals via ``failure_stage``).
        signals = FailureSignals(
            failure_class=failure_class,
            failure_category=failure_category,
            workspace_created=bool(
                failure_category == "workspace_inconsistent" or failure_class == "infra_failed"
            ),
        )

    status = (terminal_status or "").lower()
    reason_l = (failure_reason or "").lower()

    # 1. Harness failure: only genuine harness/runtime exceptions.
    if harness_error or signals.failure_category == "harness_failure":
        return Outcome.HARNESS_FAILURE.value

    # 2. Infrastructure failure: workspace/project provisioning failed before
    #    any LLM decision or Agent tool call, or an explicit infra signal.
    #    A provider/transport/parser signal that merely happened to occur
    #    pre-workspace is classified under its own (higher-specificity) bucket,
    #    not as a generic infrastructure failure.
    provisioning_failed = (
        (signals.failure_class == "infra_failed" or signals.failure_category in ("workspace_inconsistent",) or signals.failure_stage == "provisioning")
        and not signals.provider_error
        and not signals.decision_parse_error
        and not signals.transport_error
    )
    if provisioning_failed:
        return Outcome.INFRASTRUCTURE_FAILURE.value

    # 3. Provider failure: auth, model unavailable, provider HTTP failure
    #    before a valid decision was produced.
    if signals.failure_category == "provider_failure" or signals.provider_error or signals.failure_class == "model_unavailable":
        return Outcome.PROVIDER_FAILURE.value

    # 4. Decision/parser failure.
    if signals.decision_parse_error or signals.failure_class == "parser_failed" or "unsupported decision type" in reason_l:
        return Outcome.DECISION_PARSE_FAILURE.value

    # 5. DoBox transport failure (genuine transport/protocol errors only).
    if signals.transport_error or signals.failure_class == "transport_failed" or "dobox_transport" in reason_l or "server disconnected" in reason_l:
        return Outcome.DOBOX_TRANSPORT_FAILURE.value

    # 6. Budget exceeded.
    if signals.failure_class == "budget_exceeded" or "budget" in reason_l:
        return Outcome.BUDGET_EXCEEDED.value

    # 7. No progress.
    if signals.failure_class == "no_progress" or "no_progress" in reason_l or "non_convergent" in reason_l:
        return Outcome.NO_PROGRESS.value

    # 8. Terminal-based classification.
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
