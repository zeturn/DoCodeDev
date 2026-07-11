import unittest

from docode.agent.failure_taxonomy import FailureCategory, TerminalResult
from docode.agent.finalization_controller import FinalizationController, FinalizationState
from docode.agent.repair_coordinator import RepairAction, RepairCoordinator, RepairPhase
from docode.agent.verification_scheduler import VerificationScheduler


class RuntimeV2ControllerTests(unittest.TestCase):
    def test_edit_epoch_invalidates_producer_and_validator(self) -> None:
        scheduler = VerificationScheduler.from_explicit_commands(["python collect.py --output out.json", "python validate.py out.json"])
        scheduler.record("python collect.py --output out.json", True)
        scheduler.record("python validate.py out.json", True)
        self.assertIsNone(scheduler.next_command())
        scheduler.mark_edit()
        self.assertEqual(scheduler.next_command(), "python collect.py --output out.json")

    def test_failed_producer_prevents_validator(self) -> None:
        scheduler = VerificationScheduler.from_explicit_commands(["python generate.py out.json", "python check.py out.json"])
        scheduler.record("python generate.py out.json", False)
        self.assertEqual(scheduler.next_command(), "python generate.py out.json")

    def test_third_identical_repair_is_non_convergent(self) -> None:
        coordinator = RepairCoordinator()
        action = RepairAction("semantic_failure", "same", ["producer.py"], ["out.json"])
        self.assertEqual(coordinator.activate(action), RepairPhase.EDIT_REQUIRED)
        self.assertEqual(coordinator.activate(action), RepairPhase.EDIT_REQUIRED)
        self.assertEqual(coordinator.activate(action), RepairPhase.NON_CONVERGENT)

    def test_terminal_result_keeps_harness_and_strict_separate(self) -> None:
        result = TerminalResult("failed", FailureCategory.HARNESS_FAILURE, "checker crashed", functionally_correct=None, harness_valid=False)
        self.assertFalse(result.to_dict()["harness_valid"])
        self.assertFalse(result.strict_success)

    def test_finalization_rejects_stale_or_placeholder_patch(self) -> None:
        state = FinalizationState(("src/app.py",), diff="+ # TODO placeholder", explicit_commands_fresh=False, summary="done", exporter_succeeded=True)
        decision = FinalizationController().evaluate(state)
        self.assertFalse(decision.ready)
        self.assertIn("placeholder_or_debug_marker", decision.failures)
        self.assertIn("explicit_commands_stale", decision.failures)

    def test_finalization_accepts_fresh_reviewed_patch(self) -> None:
        state = FinalizationState(("src/app.py",), ("src/app.py",), "+ return value", True, True, True, False, "Implemented and verified", True)
        self.assertTrue(FinalizationController().evaluate(state).ready)


if __name__ == "__main__":
    unittest.main()
