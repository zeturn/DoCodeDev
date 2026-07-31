"""No-progress detection and escalation for agent loops.

Tracks repeated non-progress actions and repeated identical blockers
across agent iterations.  Escalates: guide → block exact repeat → stop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from docode.agent.outcome import RequiredAction, StepOutcome

_MAX_OBSERVED_KEYS = 200
_MAX_ACTION_COUNTS = 50
_MAX_BLOCKER_COUNTS = 30
_MAX_RECENT_ACTIONS = 20


class NoProgressEscalation(str, Enum):
    NONE = "none"
    GUIDE = "guide"
    BLOCK_REPEAT = "block_repeat"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class NoProgressPolicy:
    guide_after: int = 2
    block_repeat_after: int = 3
    stop_after: int = 5
    repeated_blocker_stop_after: int = 4


@dataclass(frozen=True, slots=True)
class NoProgressAssessment:
    no_progress: bool
    streak: int
    repeated_action_count: int
    repeated_blocker_count: int
    escalation: NoProgressEscalation
    reason: str
    blocked_action_key: str | None = None
    required_action: RequiredAction = RequiredAction.NONE

    def to_dict(self) -> dict[str, object]:
        return {
            "no_progress": self.no_progress,
            "streak": self.streak,
            "repeated_action_count": self.repeated_action_count,
            "repeated_blocker_count": self.repeated_blocker_count,
            "escalation": self.escalation.value,
            "reason": self.reason,
            "blocked_action_key": self.blocked_action_key,
            "required_action": self.required_action.value,
        }


class NoProgressTracker:
    """Observes outcomes and escalates when the agent makes no real progress."""

    def __init__(self, policy: NoProgressPolicy | None = None) -> None:
        self.policy = policy or NoProgressPolicy()

        self._streak: int = 0
        self._last_progress_outcome: StepOutcome | None = None

        self._observed_read_keys: dict[str, int] = OrderedDict()
        self._action_counts: OrderedDict[str, int] = OrderedDict()
        self._blocker_counts: OrderedDict[str, int] = OrderedDict()
        self._recent_actions: list[str] = []

        self._blocked_keys: set[str] = set()
        self._done_once: bool = False

    # -- public API --------------------------------------------------------

    def observe(self, outcome: StepOutcome) -> NoProgressAssessment:
        self._done_once = True

        if outcome.progress:
            return self._handle_progress(outcome)
        return self._handle_no_progress(outcome)

    def should_block(self, action_key: str) -> bool:
        return action_key in self._blocked_keys

    def reset(self) -> None:
        self._streak = 0
        self._last_progress_outcome = None
        self._observed_read_keys.clear()
        self._action_counts.clear()
        self._blocker_counts.clear()
        self._recent_actions.clear()
        self._blocked_keys.clear()

    def reset_action_blocks(self) -> None:
        """Clear the set of blocked action keys (called on edit-epoch change)."""
        self._blocked_keys.clear()

    # -- internal ----------------------------------------------------------

    def _handle_progress(self, outcome: StepOutcome) -> NoProgressAssessment:
        self._streak = 0
        self._last_progress_outcome = outcome
        self._action_counts.pop(outcome.action_key, None)
        self._blocked_keys.discard(outcome.action_key)

        bl = outcome.primary_blocker()
        if bl is not None:
            self._blocker_counts.pop(bl.fingerprint(), None)

        return NoProgressAssessment(
            no_progress=False,
            streak=0,
            repeated_action_count=0,
            repeated_blocker_count=0,
            escalation=NoProgressEscalation.NONE,
            reason="progress",
        )

    def _handle_no_progress(self, outcome: StepOutcome) -> NoProgressAssessment:
        self._streak += 1
        self._record_action(outcome.action_key)

        blocker = outcome.primary_blocker()
        blocker_fp = blocker.fingerprint() if blocker else None
        repeated_blocker = 0
        if blocker_fp:
            count = self._blocker_counts.get(blocker_fp, 0) + 1
            self._blocker_counts[blocker_fp] = count
            repeated_blocker = count
            _trim_ordered_dict(self._blocker_counts, _MAX_BLOCKER_COUNTS)

        repeated_action = self._action_counts.get(outcome.action_key, 0)
        escalation = self._compute_escalation(repeated_action, repeated_blocker)

        reason = self._build_reason(outcome, escalation)
        required = self._compute_required_action(outcome, escalation)

        blocked_action = None
        if escalation == NoProgressEscalation.BLOCK_REPEAT:
            self._blocked_keys.add(outcome.action_key)
            blocked_action = outcome.action_key

        return NoProgressAssessment(
            no_progress=True,
            streak=self._streak,
            repeated_action_count=repeated_action,
            repeated_blocker_count=repeated_blocker,
            escalation=escalation,
            reason=reason,
            blocked_action_key=blocked_action,
            required_action=required,
        )

    def _record_action(self, key: str) -> None:
        count = self._action_counts.get(key, 0) + 1
        self._action_counts[key] = count
        _trim_ordered_dict(self._action_counts, _MAX_ACTION_COUNTS)

        self._recent_actions.append(key)
        if len(self._recent_actions) > _MAX_RECENT_ACTIONS:
            self._recent_actions = self._recent_actions[-_MAX_RECENT_ACTIONS:]

    def _compute_escalation(
        self,
        repeated_action: int,
        repeated_blocker: int,
    ) -> NoProgressEscalation:
        if repeated_blocker >= self.policy.repeated_blocker_stop_after:
            return NoProgressEscalation.STOP
        if self._streak >= self.policy.stop_after:
            return NoProgressEscalation.STOP
        if repeated_action >= self.policy.block_repeat_after:
            return NoProgressEscalation.BLOCK_REPEAT
        if self._streak >= self.policy.guide_after:
            return NoProgressEscalation.GUIDE
        return NoProgressEscalation.NONE

    def _build_reason(
        self,
        outcome: StepOutcome,
        escalation: NoProgressEscalation,
    ) -> str:
        blocker = outcome.primary_blocker()
        if escalation == NoProgressEscalation.STOP:
            if blocker:
                return (
                    f"no_progress_non_convergent:repeated_blocker:"
                    f"{blocker.code}"
                )
            return (
                f"no_progress_non_convergent:repeated_action:"
                f"{outcome.action_key[:64]}"
            )
        if escalation == NoProgressEscalation.BLOCK_REPEAT:
            return (
                f"repeated_action_blocked:{outcome.action_key[:64]}"
            )
        if escalation == NoProgressEscalation.GUIDE:
            return (
                f"no_progress_guidance:streak={self._streak}"
            )
        return f"no_progress_streak={self._streak}"

    def _compute_required_action(
        self,
        outcome: StepOutcome,
        escalation: NoProgressEscalation,
    ) -> RequiredAction:
        if escalation == NoProgressEscalation.STOP:
            return RequiredAction.STOP_NON_CONVERGENT
        if escalation == NoProgressEscalation.BLOCK_REPEAT:
            return RequiredAction.CHOOSE_DIFFERENT_ACTION
        return outcome.effective_required_action()


def _trim_ordered_dict(d: OrderedDict, limit: int) -> None:
    while len(d) > limit:
        d.popitem(last=False)
