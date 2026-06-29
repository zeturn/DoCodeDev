from __future__ import annotations

from unittest import TestCase

from docode.agent.state import AgentState
from docode.agent.stuck import StuckDetector
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class StuckDetectorTests(TestCase):
    def test_detects_clean_status_after_many_iterations_without_edit(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="fix calculator.py"), iteration=6)

        signal = StuckDetector().evaluate(state=state, latest_git_status="")

        self.assertTrue(signal.stuck)
        self.assertEqual(signal.reason, "no_diff_after_multiple_iterations")
        self.assertIn("next action must be an edit_file", signal.repair_instruction)

    def test_does_not_detect_after_edit_tool_called(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction="fix calculator.py"), iteration=6)
        state.add_tool_result(ToolResult(tool="edit_file", output="diff", exit_code=0))

        signal = StuckDetector().evaluate(state=state, latest_git_status="")

        self.assertFalse(signal.stuck)
