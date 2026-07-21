"""Unit tests for NoProgressTracker escalation and detection."""

from unittest import TestCase

from docode.agent.no_progress import (
    NoProgressAssessment,
    NoProgressEscalation,
    NoProgressPolicy,
    NoProgressTracker,
)
from docode.agent.outcome import (
    BlockerSource,
    FinalizationBlocker,
    OutcomeKind,
    RequiredAction,
    StepOutcome,
)


def _out(progress: bool = False, ak: str = "a", **kw) -> StepOutcome:
    kwargs = {
        "kind": OutcomeKind.TOOL,
        "action_key": ak,
        "success": not progress or True,
        "progress": progress,
    }
    kwargs.update(kw)
    return StepOutcome(**kwargs)


def _out_with_blocker(code: str, source: BlockerSource = BlockerSource.FINALIZATION,
                      action: RequiredAction = RequiredAction.EDIT_TARGET,
                      ak: str = "final_candidate", **kw) -> StepOutcome:
    blocker = FinalizationBlocker(
        code=code, source=source,
        message=code, required_action=action,
    )
    kwargs = {
        "kind": OutcomeKind.FINALIZATION,
        "action_key": ak,
        "success": False,
        "progress": False,
        "blockers": (blocker,),
    }
    kwargs.update(kw)
    return StepOutcome(**kwargs)


class ProgressionTests(TestCase):
    def test_progress_resets_streak(self) -> None:
        tracker = NoProgressTracker()
        tracker.observe(_out(progress=False, ak="read_file:x"))
        tracker.observe(_out(progress=False, ak="read_file:x"))
        ass = tracker.observe(_out(progress=True, ak="edit_file:y"))
        self.assertEqual(ass.streak, 0)
        self.assertFalse(ass.no_progress)

    def test_repeated_no_progress_accumulates_streak(self) -> None:
        tracker = NoProgressTracker()
        for _ in range(3):
            tracker.observe(_out(progress=False, ak="read_file:x"))
        ass = tracker.observe(_out(progress=False, ak="read_file:x"))
        self.assertEqual(ass.streak, 4)
        self.assertTrue(ass.no_progress)

    def test_different_actions_do_not_reset_streak(self) -> None:
        tracker = NoProgressTracker()
        tracker.observe(_out(progress=False, ak="read_file:x"))
        tracker.observe(_out(progress=False, ak="read_file:y"))
        ass = tracker.observe(_out(progress=False, ak="read_file:z"))
        self.assertEqual(ass.streak, 3)

    def test_block_repeat_after_threshold(self) -> None:
        tracker = NoProgressTracker()
        for _ in range(3):
            tracker.observe(_out(progress=False, ak="read_file:x"))
        ass = tracker.observe(_out(progress=False, ak="read_file:x"))
        self.assertEqual(ass.escalation, NoProgressEscalation.BLOCK_REPEAT)
        self.assertTrue(tracker.should_block("read_file:x"))

    def test_should_not_block_different_keys(self) -> None:
        tracker = NoProgressTracker()
        for _ in range(3):
            tracker.observe(_out(progress=False, ak="read_file:x"))
        self.assertTrue(tracker.should_block("read_file:x"))
        self.assertFalse(tracker.should_block("read_file:y"))

    def test_stop_after_threshold(self) -> None:
        tracker = NoProgressTracker()
        for i in range(5):
            ass = tracker.observe(_out(progress=False, ak=f"read_file:{i}"))
        self.assertEqual(ass.escalation, NoProgressEscalation.STOP)
        self.assertIn("no_progress_non_convergent", ass.reason)

    def test_repeated_blocker_stops(self) -> None:
        policy = NoProgressPolicy(repeated_blocker_stop_after=3)
        tracker = NoProgressTracker(policy)
        for i in range(3):
            ak = f"action_{i}"
            ass = tracker.observe(_out_with_blocker("diff_empty", ak=ak))
        self.assertEqual(ass.escalation, NoProgressEscalation.STOP)

    def test_rotation_cannot_evade_blocker_detection(self) -> None:
        policy = NoProgressPolicy(repeated_blocker_stop_after=3)
        tracker = NoProgressTracker(policy)
        actions = ["final_candidate", "git_status", "git_diff", "list_files"]
        for i, ak in enumerate(actions):
            ass = tracker.observe(_out_with_blocker("diff_empty", ak=ak))
            if i >= 2:
                self.assertEqual(ass.escalation, NoProgressEscalation.STOP)

    def test_reset_clears_all(self) -> None:
        tracker = NoProgressTracker()
        for _ in range(4):
            tracker.observe(_out(progress=False, ak="read_file:x"))
        tracker.reset()
        ass = tracker.observe(_out(progress=False, ak="read_file:x"))
        self.assertEqual(ass.streak, 1)
        self.assertFalse(tracker.should_block("read_file:x"))
