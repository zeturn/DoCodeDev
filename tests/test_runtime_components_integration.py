from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from docode.agent.loop import CodingAgentLoop
from docode.agent.state import AgentState
from docode.agent.stop_policy import StopPolicy
from docode.agent.task_graph import TaskStatus
from docode.dobox.types import ToolResult
from docode.storage.models import CodingJob


class RuntimeComponentsIntegrationTests(IsolatedAsyncioTestCase):
    async def test_real_loop_bootstrap_wires_one_component_set_into_state(self) -> None:
        job = CodingJob(id="runtime-v2-wire", user_id="test", instruction="Build a JSON feed collector from https://example.test/feed with name, url fields.\n\nVerification commands:\n1. python produce.py --output out.json\n2. python validate.py out.json")
        loop = CodingAgentLoop(
            llm=SimpleNamespace(), tools=SimpleNamespace(), verifier=SimpleNamespace(),
            repository=SimpleNamespace(add_step=AsyncMock()), exporter=SimpleNamespace(), stop_policy=StopPolicy(),
            inspector=SimpleNamespace(inspect=AsyncMock(return_value=SimpleNamespace(
                listing="producer.py", important_files=["producer.py"], detected_commands={},
                explicit_commands=[], plan=[], acceptance_criteria=[]
            ))),
        )
        state = AgentState(job)
        await loop.bootstrap(state)
        self.assertIs(state.profile, loop.runtime_components.profile)
        self.assertEqual(state.profile.name, "crawler")
        self.assertIs(state.verification_scheduler, loop.runtime_components.verification_scheduler)
        self.assertIs(state.repair_coordinator, loop.runtime_components.repair_coordinator)
        self.assertEqual(set(state.task_graph.nodes), {"understand", "plan", "implement", "verify", "review"})
        bootstrap = loop.repository.add_step.await_args.args[2]
        self.assertEqual(bootstrap["runtime_components"]["profile"], "crawler")

        state.add_tool_result(ToolResult("write_file", "ok", metadata={"path": "producer.py"}))
        self.assertEqual(state.edit_epoch, 1)
        self.assertEqual(state.verification_scheduler.edit_epoch, 1)
        self.assertEqual(state.task_graph.nodes["implement"].status, TaskStatus.DONE)
        command = state.verification_scheduler.next_command()
        state.add_tool_result(ToolResult("run_command", "ok", metadata={"command": command}))
        self.assertTrue(state.verification_scheduler.is_fresh_success(command))
        validator = state.verification_scheduler.next_command()
        self.assertEqual(validator, "python validate.py out.json")


if __name__ == "__main__":
    import unittest
    unittest.main()
