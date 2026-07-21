from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class FailureCategory(str, Enum):
    RUNTIME_FAILURE = "runtime_failure"
    PROVIDER_FAILURE = "provider_failure"
    SANDBOX_FAILURE = "sandbox_failure"
    TOOL_FAILURE = "tool_failure"
    POLICY_FAILURE = "policy_failure"
    REPOSITORY_UNDERSTANDING_FAILURE = "repository_understanding_failure"
    CODE_GENERATION_FAILURE = "code_generation_failure"
    VERIFICATION_FAILURE = "verification_failure"
    SEMANTIC_FAILURE = "semantic_failure"
    REPAIR_NON_CONVERGENT = "repair_non_convergent"
    VERIFIER_FALSE_NEGATIVE = "verifier_false_negative"
    FINALIZATION_FAILURE = "finalization_failure"
    HARNESS_FAILURE = "harness_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class TerminalResult:
    status: str
    category: FailureCategory
    failure_reason: str = ""
    functionally_correct: bool | None = None
    strict_success: bool = False
    harness_valid: bool = True

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["category"] = self.category.value
        return value


def category_for_reason(reason: str) -> FailureCategory:
    lowered = (reason or "").lower()
    if "non_convergent" in lowered or "repeated_zero_record" in lowered:
        return FailureCategory.REPAIR_NON_CONVERGENT
    if any(token in lowered for token in ("llm", "provider", "apicred", "auth_failed")):
        return FailureCategory.PROVIDER_FAILURE
    if any(token in lowered for token in ("sandbox", "dobox")):
        return FailureCategory.SANDBOX_FAILURE
    if "semantic" in lowered or "quality" in lowered:
        return FailureCategory.SEMANTIC_FAILURE
    if "verif" in lowered or "required_command" in lowered:
        return FailureCategory.VERIFICATION_FAILURE
    if "finalization" in lowered or "export" in lowered:
        return FailureCategory.FINALIZATION_FAILURE
    if "environment" in lowered or "workspace_inconsistent" in lowered:
        return FailureCategory.ENVIRONMENT_FAILURE
    if "source" in lowered and "unavailable" in lowered:
        return FailureCategory.SOURCE_UNAVAILABLE
    return FailureCategory.RUNTIME_FAILURE


def failed_terminal_result(reason: str) -> dict[str, object]:
    return TerminalResult("failed", category_for_reason(reason), reason).to_dict()
