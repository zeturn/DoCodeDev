from __future__ import annotations

from unittest import TestCase

from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob, new_id


class StopPolicyTests(TestCase):
    def test_stops_when_tool_call_budget_is_exhausted(self) -> None:
        state = AgentState(CodingJob(id=new_id("job"), user_id="u1", instruction="do work"))
        state.add_tool_result(ToolResult(tool="list_files", output=""))
        decision = StopPolicy(max_tool_calls=1).evaluate(state)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "max_tool_calls_exceeded")

    def test_stops_when_llm_token_budget_is_exhausted(self) -> None:
        state = AgentState(CodingJob(id=new_id("job"), user_id="u1", instruction="do work"))
        state.llm_tokens_used = 42
        decision = StopPolicy(max_llm_tokens=42).evaluate(state)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "max_llm_tokens_exceeded")

    def test_stops_when_llm_cost_budget_is_exhausted(self) -> None:
        state = AgentState(CodingJob(id=new_id("job"), user_id="u1", instruction="do work"))
        state.llm_cost_used = 0.25
        decision = StopPolicy(max_llm_cost=0.25).evaluate(state)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "max_llm_cost_exceeded")
