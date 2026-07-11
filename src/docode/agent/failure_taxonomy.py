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
