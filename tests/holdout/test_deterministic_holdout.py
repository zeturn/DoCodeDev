from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase

from docode.agent.loop import CodingAgentLoop
from docode.agent.quality_gate import QualityGate
from docode.agent.stop_policy import StopPolicy
from docode.agent.verifier import CodingVerifier
from docode.artifacts.exporter import ArtifactExporter
from docode.storage.models import CodingJob, JobStatus, new_id

from tests.holdout.definitions import CASES, LEAKAGE_MARKERS
from tests.holdout.harness import HoldoutLocalTools, ScriptedHoldoutLLM, materialize_fixture, summarize_steps, validate_workspace
from tests.test_smoke_readme_job import RecordingRepository


class HoldoutFixtureTests(TestCase):
    def test_all_eight_fixtures_seed_and_begin_unsolved(self) -> None:
        self.assertEqual(len(CASES), 8)
        for case in CASES:
            with self.subTest(case=case.name), TemporaryDirectory() as tmp:
                workspace = materialize_fixture(case, Path(tmp) / case.name)
                self.assertTrue(any(workspace.rglob("*")))
                if case.name == "ivory_quill":
                    self.assertEqual([path.name for path in workspace.iterdir()], ["README.md"])
                if case.name == "amber_depth":
                    self.assertGreater((workspace / "zephyr_lattice.py").stat().st_size, 80_000)

    def test_production_contains_no_holdout_leakage(self) -> None:
        production_root = Path(__file__).resolve().parents[2] / "src" / "docode"
        source = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in production_root.rglob("*.py")
        )
        leaked = [marker for marker in LEAKAGE_MARKERS if marker in source]
        self.assertEqual(leaked, [], f"holdout markers leaked into production: {leaked}")

    def test_holdout_filenames_are_absent_from_production(self) -> None:
        production_root = Path(__file__).resolve().parents[2] / "src" / "docode"
        source = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in production_root.rglob("*.py"))
        filenames = ["morrow_mesh.mjs", "lumen_quota.py", "zephyr_lattice.py", "harvest_adapter.py", "quiet_core.py", "contract-check.mjs"]
        self.assertEqual([name for name in filenames if name in source], [])


class DeterministicHoldoutLoopTests(IsolatedAsyncioTestCase):
    async def test_all_cases_reach_final_gates_and_export_artifacts(self) -> None:
        for case in CASES:
            with self.subTest(case=case.name), TemporaryDirectory() as tmp:
                root = Path(tmp)
                workspace = materialize_fixture(case, root / "workspace")
                repo = RecordingRepository()
                job = await repo.create_job(
                    CodingJob(
                        id=new_id("job"),
                        user_id="holdout-deterministic",
                        instruction=case.instruction,
                        max_iterations=36,
                        max_runtime_seconds=900,
                        max_consecutive_failures=10,
                        max_tool_calls=80,
                    )
                )
                tools = HoldoutLocalTools(workspace, test_command=case.required_commands[0])
                loop = CodingAgentLoop(
                    llm=ScriptedHoldoutLLM(case),
                    tools=tools,
                    verifier=CodingVerifier(),
                    repository=repo,
                    exporter=ArtifactExporter(root / "artifacts", repo, workspace_file_reader=tools.read_file),
                    stop_policy=StopPolicy(max_iterations=36, max_runtime_seconds=900, max_consecutive_failures=10, max_tool_calls=80),
                    quality_gate=QualityGate(),
                )

                result = await loop.run(job)
                steps = await repo.list_steps(job.id)
                summary = summarize_steps(steps)
                failures = validate_workspace(case, workspace)

                self.assertEqual(result.status, JobStatus.SUCCEEDED, f"{case.name}: {result.failure_reason}; {failures}")
                self.assertEqual(failures, [])
                self.assertTrue(summary["final_candidate_attempted"])
                self.assertTrue(summary["verifier_result"])
                self.assertTrue(summary["quality_gate_result"])
                self.assertTrue(summary["read_before_edit"])
                artifacts = await repo.list_artifacts(job.id)
                self.assertTrue(artifacts)
                self.assertIsNotNone(result.artifact_id)
                if case.name == "amber_depth":
                    self.assertFalse(summary["whole_file_rewrite"])
                    ranged = [step for step in steps if step.content.get("type") == "tool_result" and step.content.get("tool") == "read_file_range"]
                    self.assertTrue(ranged)
                if case.name == "indigo_block":
                    heredoc_calls = [
                        step.content for step in steps
                        if step.content.get("type") == "tool_call"
                        and step.content.get("tool") == "run_command"
                        and "<<'NODE'" in str(step.content.get("args", {}).get("command", ""))
                    ]
                    self.assertEqual(len(heredoc_calls), 1)
                    self.assertTrue(str(heredoc_calls[0]["args"]["command"]).rstrip().endswith("NODE"))
                if case.name == "crimson_ladder":
                    command_results = [
                        step.content for step in steps
                        if step.content.get("type") == "tool_result" and step.content.get("tool") == "run_command"
                    ]
                    self.assertTrue(any(item.get("exit_code") != 0 for item in command_results))
                    self.assertTrue(any(item.get("exit_code") == 0 for item in command_results))
