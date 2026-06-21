from __future__ import annotations

import json
from unittest import TestCase

from docode.agent.output import prompt_safe_output
from docode.agent.state import AgentState
from docode.dobox.types import ToolResult
from docode.llm.runtime import tool_result_for_prompt
from docode.storage.models import CodingJob, new_id


class PromptOutputTests(TestCase):
    def test_prompt_safe_output_keeps_first_300_lines(self) -> None:
        output = "".join(f"line {index}\n" for index in range(305))
        prompt_output = prompt_safe_output(output)

        self.assertTrue(prompt_output.truncated)
        self.assertEqual(prompt_output.original_lines, 305)
        self.assertIn("line 0\n", prompt_output.text)
        self.assertIn("line 299", prompt_output.text)
        self.assertNotIn("line 300", prompt_output.text)
        self.assertTrue(prompt_output.text.endswith("<truncated>"))

    def test_agent_state_stores_prompt_safe_tool_output(self) -> None:
        state = AgentState(CodingJob(id=new_id("job"), user_id="u1", instruction="inspect logs"))
        output = "".join(f"line {index}\n" for index in range(305))

        state.add_tool_result(ToolResult(tool="run_command", output=output, exit_code=1))

        message = state.messages[-1]
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["exit_code"], 1)
        self.assertTrue(message["truncated"])
        self.assertNotIn("line 300", str(message["output"]))
        self.assertEqual(message["metadata"]["original_output_lines"], 305)
        self.assertTrue(message["metadata"]["prompt_output_truncated"])

    def test_verifier_prompt_tool_result_is_prompt_safe_json(self) -> None:
        output = "".join(f"line {index}\n" for index in range(305))

        payload = json.loads(tool_result_for_prompt(ToolResult(tool="run_tests", output=output, metadata={"command": "pytest", "detected": True})))

        self.assertEqual(payload["tool"], "run_tests")
        self.assertEqual(payload["command"], "pytest")
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["original_output_lines"], 305)
        self.assertIn("line 299", payload["output"])
        self.assertNotIn("line 300", payload["output"])
