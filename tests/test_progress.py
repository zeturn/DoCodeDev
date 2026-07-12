"""Unit tests for state-progress fingerprint stability."""

from unittest import TestCase
from unittest.mock import MagicMock

from docode.agent.progress import state_progress_fingerprint, state_progress_snapshot
from docode.agent.repair_coordinator import RepairCoordinator, RepairPhase
from docode.agent.state import AgentState
from docode.storage.models import CodingJob


def _state(**overrides) -> AgentState:
    job = CodingJob(
        id="j", user_id="u",
        instruction="test",
        max_iterations=36,
        max_runtime_seconds=900,
        max_consecutive_failures=10,
        max_tool_calls=80,
    )
    s = AgentState(job=job)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class FingerprintExclusionTests(TestCase):
    def test_iteration_change_does_not_alter_fingerprint(self) -> None:
        a = _state(iteration=0)
        b = _state(iteration=100)
        self.assertEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_tool_calls_change_does_not_alter_fingerprint(self) -> None:
        a = _state(tool_calls_count=0)
        b = _state(tool_calls_count=500)
        self.assertEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_message_count_does_not_alter_fingerprint(self) -> None:
        a = _state(messages=[])
        b = _state(messages=[{"role": "user", "content": "hi"}] * 20)
        self.assertEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )


class FingerprintChangeTests(TestCase):
    def test_edit_epoch_change_alters_fingerprint(self) -> None:
        a = _state(edit_epoch=0)
        b = _state(edit_epoch=1)
        self.assertNotEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_quality_gate_change_alters_fingerprint(self) -> None:
        a = _state(quality_gate_passed=False)
        b = _state(quality_gate_passed=True)
        self.assertNotEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_repair_mode_change_alters_fingerprint(self) -> None:
        a = _state(repair_mode=None)
        b = _state(repair_mode="must_edit")
        self.assertNotEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_repair_phase_change_alters_fingerprint(self) -> None:
        rc_a = RepairCoordinator()
        rc_a.phase = RepairPhase.INSPECT_ALLOWED
        rc_b = RepairCoordinator()
        rc_b.phase = RepairPhase.EDIT_REQUIRED
        a = _state(repair_coordinator=rc_a)
        b = _state(repair_coordinator=rc_b)
        self.assertNotEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )

    def test_terminal_repair_reason_change_alters_fingerprint(self) -> None:
        a = _state(terminal_repair_reason=None)
        b = _state(terminal_repair_reason="no_progress")
        self.assertNotEqual(
            state_progress_fingerprint(a),
            state_progress_fingerprint(b),
        )


class SnapshotStructureTests(TestCase):
    def test_snapshot_contains_expected_keys(self) -> None:
        snap = state_progress_snapshot(_state())
        for key in (
            "edit_epoch", "changed_paths", "task_graph_nodes",
            "scheduler", "repair", "quality_gate_passed",
            "active_blocker_fingerprint",
        ):
            self.assertIn(key, snap)
