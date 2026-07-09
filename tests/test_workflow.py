from __future__ import annotations

from unittest import TestCase

from docode.agent.state import AgentState
from docode.agent.task_contract import TaskContract
from docode.agent.workflow import command_was_run, commands_equivalent
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class WorkflowCommandEquivalenceTests(TestCase):
    def test_harmless_stderr_redirection_satisfies_required_command(self) -> None:
        self.assertTrue(commands_equivalent("python -m unittest discover -s tests 2>&1", "python -m unittest discover -s tests"))
        self.assertTrue(commands_equivalent("python crawler.py sample.json --output out.json 2>&1", "python crawler.py sample.json --output out.json"))

    def test_successful_and_compound_can_satisfy_multiple_required_segments(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(
            must_run_commands=[
                "python -m unittest discover -s tests",
                "python crawler.py sample.json --output out.json",
            ]
        )
        state.add_tool_result(
            ToolResult(
                tool="run_command",
                output="OK",
                exit_code=0,
                metadata={
                    "command": "python -m unittest discover -s tests && python crawler.py sample.json --output out.json && cat out.json"
                },
            )
        )

        self.assertTrue(command_was_run(state, "python -m unittest discover -s tests"))
        self.assertTrue(command_was_run(state, "python crawler.py sample.json --output out.json"))

    def test_failing_compound_does_not_satisfy_required_segment(self) -> None:
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_run_commands=["python -m unittest discover -s tests"])
        state.add_tool_result(
            ToolResult(
                tool="run_command",
                output="FAIL",
                exit_code=1,
                metadata={"command": "python -m unittest discover -s tests && python crawler.py sample.json --output out.json"},
            )
        )

        self.assertFalse(command_was_run(state, "python -m unittest discover -s tests"))
