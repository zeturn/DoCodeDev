from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMUsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    estimated: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_text_call(self, *, prompt: str, response: str, cost: float = 0.0) -> None:
        self.calls += 1
        self.prompt_tokens += estimate_tokens(prompt)
        self.completion_tokens += estimate_tokens(response)
        self.cost += cost

    def record_provider_call(self, *, prompt: str, result: Any) -> None:
        if result.prompt_tokens is None and result.completion_tokens is None and result.total_tokens is None:
            self.record_text_call(prompt=prompt, response=result.text, cost=result.cost or 0.0)
            return

        prompt_tokens = result.prompt_tokens if result.prompt_tokens is not None else (0 if result.total_tokens is not None else estimate_tokens(prompt))
        if result.completion_tokens is not None:
            completion_tokens = result.completion_tokens
        elif result.total_tokens is not None:
            completion_tokens = max(0, result.total_tokens - prompt_tokens)
        else:
            completion_tokens = estimate_tokens(result.text)

        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost += result.cost or 0.0
        self.estimated = False

    def snapshot(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "estimated": self.estimated,
        }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative provider-agnostic estimate for billing guardrails when the
    # provider adapter does not expose structured usage metadata.
    return max(1, (len(text.encode("utf-8")) + 3) // 4)
