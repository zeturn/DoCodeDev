from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .state import AgentState


@dataclass(frozen=True, slots=True)
class StopDecision:
    should_stop: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StopPolicy:
    max_iterations: int = 50
    max_runtime_seconds: int = 1800
    max_consecutive_failures: int = 5
    max_tool_calls: int = 100
    max_llm_tokens: int | None = None
    max_llm_cost: float | None = None

    def evaluate(self, state: AgentState) -> StopDecision:
        if state.iteration >= self.max_iterations:
            return StopDecision(True, "max_iterations_exceeded")
        if monotonic() - state.started_monotonic >= self.max_runtime_seconds:
            return StopDecision(True, "max_runtime_exceeded")
        if state.consecutive_failures >= self.max_consecutive_failures:
            return StopDecision(True, "max_consecutive_failures_exceeded")
        if state.tool_calls_count >= self.max_tool_calls:
            return StopDecision(True, "max_tool_calls_exceeded")
        if self.max_llm_tokens is not None and state.llm_tokens_used >= self.max_llm_tokens:
            return StopDecision(True, "max_llm_tokens_exceeded")
        if self.max_llm_cost is not None and state.llm_cost_used >= self.max_llm_cost:
            return StopDecision(True, "max_llm_cost_exceeded")
        return StopDecision(False)
