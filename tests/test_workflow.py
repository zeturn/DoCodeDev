from __future__ import annotations

from unittest import TestCase

from docode.agent.state import AgentState
from docode.agent.task_contract import TaskContract
from docode.agent.workflow import command_was_run, commands_equivalent, display_command, workflow_snapshot
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class WorkflowCommandEquivalenceTests(TestCase):
    def test_identical_multiline_heredoc_is_recognized_as_run(self) -> None:
        command = "python - <<'PY'\nprint('ok')\nPY"
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.task_contract = TaskContract(must_run_commands=[command])
        state.add_tool_result(ToolResult(tool="run_command", output="ok", metadata={"command": command}))

        self.assertTrue(command_was_run(state, command))

    def test_multiline_command_normalizes_crlf_without_losing_body_identity(self) -> None:
        expected = "python - <<'PY'\nprint('ok')\nPY"
        observed = expected.replace("\n", "\r\n")

        self.assertTrue(commands_equivalent(observed, expected))

    def test_same_heredoc_opener_with_different_bodies_is_not_equivalent(self) -> None:
        first = "python - <<'PY'\nprint('one')\nPY"
        second = "python - <<'PY'\nprint('two')\nPY"

        self.assertFalse(commands_equivalent(first, second))
        self.assertFalse(commands_equivalent("python - <<'PY'", first))

    def test_failed_multiline_command_does_not_satisfy_final_gate(self) -> None:
        command = "python - <<'PY'\nraise AssertionError('no')\nPY"
        state = AgentState(job=CodingJob(id=new_id("job"), user_id="u1", instruction=""))
        state.inspection = object()
        state.task_contract = TaskContract(must_run_commands=[command])
        state.messages.extend(
            [
                {"role": "tool", "tool": "write_file", "exit_code": 0, "metadata": {"path": "app.py"}},
                {"role": "tool", "tool": "run_command", "exit_code": 1, "metadata": {"command": command}},
            ]
        )

        snapshot = workflow_snapshot(state, " M app.py\n")

        self.assertFalse(snapshot.final_allowed)
        self.assertIn("multiline verification command", snapshot.required_action)

    def test_display_command_summarizes_multiline_only(self) -> None:
        self.assertEqual(display_command("python app.py"), "python app.py")
        self.assertEqual(
            display_command("python - <<'PY'\nprint('ok')\nPY"),
            "python - <<'PY' [multiline verification command, 3 lines]",
        )

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
