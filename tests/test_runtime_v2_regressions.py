from __future__ import annotations

import unittest

from docode.agent.loop import repair_action_from_quality_gate
from docode.agent.quality_gate import QualityGateResult, QualityIssue
from docode.agent.repair_coordinator import RepairCoordinator, RepairPhase


class RuntimeV2RegressionTests(unittest.TestCase):
    def test_generated_artifact_failure_targets_producer_not_json(self) -> None:
        issue = QualityIssue("blocker", "artifact_semantic_failure", "wrong count", "out.json", producer_targets=("producer.py",), artifact_ownership="generated")
        action = repair_action_from_quality_gate(QualityGateResult(False, [issue]))
        self.assertEqual(["producer.py"], action.target_files)
        self.assertNotIn("out.json", action.target_files)

    def test_unknown_generated_artifact_producer_enters_locate_phase(self) -> None:
        issue = QualityIssue("blocker", "artifact_semantic_failure", "wrong count", "out.json", artifact_ownership="generated", evidence_refs=("artifact:out.json",))
        action = repair_action_from_quality_gate(QualityGateResult(False, [issue]))
        self.assertEqual([], action.target_files)
        self.assertNotIn("edit_file", action.allowed_tools)
        coordinator = RepairCoordinator()
        self.assertEqual(RepairPhase.LOCATE_PRODUCER, coordinator.activate(action))

    def test_source_owned_artifact_can_be_direct_target(self) -> None:
        issue = QualityIssue("blocker", "artifact_semantic_failure", "wrong count", "catalog.json", artifact_ownership="source_owned")
        action = repair_action_from_quality_gate(QualityGateResult(False, [issue]))
        self.assertEqual(["catalog.json"], action.target_files)


if __name__ == "__main__":
    unittest.main()
