from __future__ import annotations

import json
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.api.events import event_stream, load_result_payload, step_event_payload
from docode.api.artifact_actions import artifact_descriptor
from docode.artifacts.terminal import export_stopped_artifacts
from docode.api.job_actions import cancel_existing_job
from docode.config import DocodeConfig
from docode.storage.models import CodingJob, JobStatus, new_id, public_job_dict
from docode.storage.repository import InMemoryJobRepository


class RecordingQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)


class JobRoutesTests(IsolatedAsyncioTestCase):
    async def test_public_job_dict_hides_apicred_access_token(self) -> None:
        job = CodingJob(id=new_id("job"), user_id="user-1", instruction="run tests", apicred_access_token="bp_xat_secret")

        payload = public_job_dict(job)

        self.assertNotIn("apicred_access_token", payload)
        self.assertNotIn("bp_xat_secret", repr(payload))

    async def test_event_stream_emits_status_step_and_done_events(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="user-1",
                instruction="cancel me",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
            )
        )
        await repo.add_step(job.id, "system", {"type": "cancelled", "reason": "user_requested_cancel"})

        stream = event_stream(repo, job.id)
        first = await anext(stream)
        second = await anext(stream)
        third = await anext(stream)

        self.assertIn("event: status", first)
        self.assertIn('"status": "stopped"', first)
        self.assertIn("event: step", second)
        self.assertIn('"type": "cancelled"', second)
        self.assertIn('"reason": "user_requested_cancel"', second)
        self.assertIn("event: done", third)
        self.assertIn('"failure_reason": "cancelled"', third)

    async def test_event_stream_waits_for_stopped_sandbox_artifact_finalization(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="user-1",
                instruction="cancel me after sandbox work",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
                dobox_project_id="project-1",
                dobox_sandbox_id="sandbox-1",
            )
        )
        await repo.add_step(job.id, "system", {"type": "cancelled", "reason": "user_requested_cancel"})
        sleeps = 0

        async def finalize_on_sleep(_: float) -> None:
            nonlocal sleeps
            sleeps += 1
            await repo.update_job(job.id, artifact_id="art_final")

        stream = event_stream(repo, job.id)
        with patch("docode.api.events.asyncio.sleep", finalize_on_sleep):
            first = await anext(stream)
            second = await anext(stream)
            third = await anext(stream)

        self.assertIn("event: status", first)
        self.assertIn('"status": "stopped"', first)
        self.assertIn("event: step", second)
        self.assertIn("event: done", third)
        self.assertIn('"artifact_id": "art_final"', third)
        self.assertEqual(sleeps, 1)

    async def test_load_result_payload_reads_result_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps({"status": "succeeded", "changed_files": ["README.md"]}), encoding="utf-8")
            await repo.add_artifact(job.id, "result", str(result_path), result_path.stat().st_size)

            payload = await load_result_payload(repo, job.id)

            self.assertEqual(payload["status"], "succeeded")
            self.assertEqual(payload["changed_files"], ["README.md"])

    async def test_artifact_listing_payload_hides_local_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            path = Path(tmp) / "patch.diff"
            path.write_text("diff", encoding="utf-8")
            artifact = await repo.add_artifact(job.id, "patch", str(path), path.stat().st_size)

            payload = [artifact_descriptor(item) for item in await repo.list_artifacts(job.id)]

            self.assertEqual(payload[0]["id"], artifact.id)
            self.assertEqual(payload[0]["filename"], "patch.diff")
            self.assertEqual(payload[0]["download_url"], f"/v1/artifacts/{artifact.id}/download")
            self.assertNotIn("path", payload[0])
            self.assertNotIn(str(tmp), repr(payload[0]))

    async def test_cancel_export_helper_writes_stopped_result_payload(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="cancel queued job"))
            await repo.add_step(job.id, "system", {"type": "cancelled", "reason": "user_requested_cancel"})

            artifact_id = await export_stopped_artifacts(repo, Path(tmp), job, "cancelled")
            await repo.update_job(job.id, status=JobStatus.STOPPED, failure_reason="cancelled", artifact_id=artifact_id)

            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertIsNotNone(completed.artifact_id)
            artifacts = await repo.list_artifacts(job.id)
            self.assertEqual({artifact.kind for artifact in artifacts}, {"report", "log", "result", "zip"})
            self.assertEqual(completed.artifact_id, next(artifact.id for artifact in artifacts if artifact.kind == "result"))
            payload = await load_result_payload(repo, job.id)
            assert payload is not None
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["stopped_reason"], "cancelled")

    async def test_cancel_sandbox_job_defers_artifact_export_to_worker(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            queue = RecordingQueue()
            job = await repo.create_job(
                CodingJob(
                    id=new_id("job"),
                    user_id="user-1",
                    instruction="cancel running job",
                    status=JobStatus.RUNNING,
                    dobox_project_id="project-1",
                    dobox_sandbox_id="sandbox-1",
                    dobox_agent_session_id="7",
                )
            )

            response = await cancel_existing_job(repo, queue, DocodeConfig(artifact_dir=Path(tmp)), job)  # type: ignore[arg-type]

            self.assertEqual(response, {"job_id": job.id, "status": "stopped"})
            completed = await repo.get_job(job.id)
            assert completed is not None
            self.assertEqual(completed.status, JobStatus.STOPPED)
            self.assertEqual(completed.failure_reason, "cancelled")
            self.assertIsNone(completed.artifact_id)
            self.assertEqual(queue.enqueued, [job.id])
            self.assertEqual(await repo.list_artifacts(job.id), [])

    async def test_event_stream_flattens_tool_call_and_result_steps(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(
            CodingJob(id=new_id("job"), user_id="user-1", instruction="run tests", status=JobStatus.SUCCEEDED, artifact_id="art_1")
        )
        await repo.add_step(job.id, "tool", {"type": "tool_call", "tool": "run_command", "args": {"command": "npm test"}})
        await repo.add_step(
            job.id,
            "tool",
            {
                "type": "tool_result",
                "tool": "run_command",
                "exit_code": 1,
                "summary": "3 tests failed",
                "output": "full failure output\nSECRET_TOKEN=do-not-stream",
                "truncated": False,
            },
        )

        stream = event_stream(repo, job.id)
        _ = await anext(stream)
        call = await anext(stream)
        result = await anext(stream)
        done = await anext(stream)

        self.assertIn('"type": "tool_call"', call)
        self.assertIn('"command": "npm test"', call)
        self.assertIn('"type": "tool_result"', result)
        self.assertIn('"summary": "3 tests failed"', result)
        self.assertNotIn("SECRET_TOKEN", result)
        self.assertNotIn('"output"', result)
        self.assertIn('"artifact_id": "art_1"', done)

    async def test_event_stream_hides_verifier_diff_and_check_output(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(
            CodingJob(id=new_id("job"), user_id="user-1", instruction="run tests", status=JobStatus.SUCCEEDED, artifact_id="art_1")
        )
        await repo.add_step(
            job.id,
            "verifier",
            {
                "passed": False,
                "confidence": 0.4,
                "reason": "tests failed",
                "required_fixes": ["fix tests"],
                "git_status": " M SECRET_TOKEN_FILE\n",
                "git_diff": "diff --git a/a b/a\n+SECRET_TOKEN=do-not-stream\n",
                "status": {"tool": "git_status", "exit_code": 0, "output": " M SECRET_TOKEN_FILE\n"},
                "test": {"tool": "run_tests", "exit_code": 1, "output": "failed\nSECRET_TOKEN=do-not-stream"},
                "build": {"tool": "run_build", "exit_code": 0, "output": "ok"},
                "lint": {"tool": "run_lint", "exit_code": 0, "output": ""},
            },
        )

        stream = event_stream(repo, job.id)
        _ = await anext(stream)
        verifier = await anext(stream)

        self.assertIn('"kind": "verifier"', verifier)
        self.assertIn('"git_status_lines": 1', verifier)
        self.assertIn('"git_diff_lines": 2', verifier)
        self.assertIn('"output_lines": 2', verifier)
        self.assertNotIn("SECRET_TOKEN", verifier)
        self.assertNotIn('"git_status":', verifier)
        self.assertNotIn('"git_diff"', verifier)
        self.assertNotIn('"output"', verifier)

    async def test_step_listing_payload_hides_full_tool_output(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="run tests"))
        step = await repo.add_step(
            job.id,
            "tool",
            {
                "type": "tool_result",
                "tool": "run_command",
                "exit_code": 1,
                "summary": "failed",
                "output": "full output\nSECRET_TOKEN=do-not-list",
                "truncated": False,
            },
        )

        payload = step_event_payload(step)

        self.assertEqual(payload["summary"], "failed")
        self.assertEqual(payload["exit_code"], 1)
        self.assertNotIn("output", payload)
        self.assertNotIn("SECRET_TOKEN", repr(payload))

    async def test_step_listing_payload_hides_verifier_diff_and_check_output(self) -> None:
        repo = InMemoryJobRepository()
        job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="run tests"))
        step = await repo.add_step(
            job.id,
            "verifier",
            {
                "passed": False,
                "confidence": 0.4,
                "reason": "tests failed",
                "required_fixes": ["fix tests"],
                "git_status": " M SECRET_TOKEN_FILE\n",
                "git_diff": "diff --git a/a b/a\n+SECRET_TOKEN=do-not-list\n",
                "status": {"tool": "git_status", "exit_code": 0, "output": " M SECRET_TOKEN_FILE\n"},
                "test": {"tool": "run_tests", "exit_code": 1, "output": "failed\nSECRET_TOKEN=do-not-list"},
                "build": {"tool": "run_build", "exit_code": 0, "output": "ok"},
                "lint": {"tool": "run_lint", "exit_code": 0, "output": ""},
            },
        )

        payload = step_event_payload(step)

        self.assertEqual(payload["passed"], False)
        self.assertEqual(payload["git_status_lines"], 1)
        self.assertGreater(payload["git_status_bytes"], 0)
        self.assertEqual(payload["git_diff_lines"], 2)
        self.assertGreater(payload["git_diff_bytes"], 0)
        self.assertEqual(payload["status"]["output_lines"], 1)
        self.assertGreater(payload["status"]["output_bytes"], 0)
        self.assertEqual(payload["test"]["output_lines"], 2)
        self.assertGreater(payload["test"]["output_bytes"], 0)
        self.assertNotIn("git_status", payload)
        self.assertNotIn("git_diff", payload)
        self.assertNotIn("output", payload["status"])
        self.assertNotIn("output", payload["test"])
        self.assertNotIn("output", payload["build"])
        self.assertNotIn("SECRET_TOKEN", repr(payload))
