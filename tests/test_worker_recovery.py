from __future__ import annotations

from datetime import timedelta
from unittest import IsolatedAsyncioTestCase

from docode.storage.models import CodingJob, JobStatus, new_id, utcnow
from docode.storage.repository import InMemoryJobRepository
from docode.worker.recovery import recover_interrupted_jobs, recover_stale_active_jobs


class WorkerRecoveryTests(IsolatedAsyncioTestCase):
    async def test_requeues_interrupted_nonterminal_jobs(self) -> None:
        repo = InMemoryJobRepository()
        queued = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="queued"))
        running = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="running", status=JobStatus.RUNNING))
        verifying = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="verifying", status=JobStatus.VERIFYING))
        succeeded = await repo.create_job(CodingJob(id=new_id("job"), user_id="u1", instruction="done", status=JobStatus.SUCCEEDED))

        recovered = await recover_interrupted_jobs(repo)

        self.assertEqual(set(recovered), {running.id, verifying.id})
        self.assertEqual((await repo.get_job(queued.id)).status, JobStatus.QUEUED)
        self.assertEqual((await repo.get_job(running.id)).status, JobStatus.QUEUED)
        self.assertEqual((await repo.get_job(verifying.id)).status, JobStatus.QUEUED)
        self.assertEqual((await repo.get_job(succeeded.id)).status, JobStatus.SUCCEEDED)

        running_steps = await repo.list_steps(running.id)
        self.assertEqual(running_steps[0].content["type"], "worker_recovered_after_restart")
        self.assertEqual(running_steps[0].content["previous_status"], "running")

    async def test_requeues_stopped_sandbox_jobs_missing_artifacts_for_finalization(self) -> None:
        repo = InMemoryJobRepository()
        stopped_pending = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="cancelled while running",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
                dobox_project_id="project-1",
                dobox_sandbox_id="sandbox-1",
                dobox_agent_session_id="7",
            )
        )
        stopped_without_sandbox = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="cancelled before sandbox",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
            )
        )
        stopped_finalized = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="cancelled and finalized",
                status=JobStatus.STOPPED,
                failure_reason="cancelled",
                dobox_project_id="project-2",
                artifact_id="art_1",
            )
        )

        recovered = await recover_interrupted_jobs(repo)

        self.assertEqual(recovered, [stopped_pending.id])
        self.assertEqual((await repo.get_job(stopped_pending.id)).status, JobStatus.STOPPED)
        self.assertEqual((await repo.get_job(stopped_without_sandbox.id)).status, JobStatus.STOPPED)
        self.assertEqual((await repo.get_job(stopped_finalized.id)).status, JobStatus.STOPPED)
        stopped_steps = await repo.list_steps(stopped_pending.id)
        self.assertEqual(stopped_steps[0].content["type"], "worker_recovered_stopped_finalization")
        self.assertEqual(stopped_steps[0].content["previous_status"], "stopped")

    async def test_requeues_stale_active_jobs_without_process_restart(self) -> None:
        repo = InMemoryJobRepository()
        stale = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="stale running",
                status=JobStatus.RUNNING,
                updated_at=utcnow() - timedelta(seconds=180),
            )
        )
        fresh = await repo.create_job(
            CodingJob(
                id=new_id("job"),
                user_id="u1",
                instruction="fresh running",
                status=JobStatus.RUNNING,
            )
        )

        recovered = await recover_stale_active_jobs(repo, 90)

        self.assertEqual(recovered, [stale.id])
        self.assertEqual((await repo.get_job(stale.id)).status, JobStatus.QUEUED)
        self.assertEqual((await repo.get_job(fresh.id)).status, JobStatus.RUNNING)
        stale_steps = await repo.list_steps(stale.id)
        self.assertEqual(stale_steps[0].content["type"], "worker_recovered_stale_job")
        self.assertEqual(stale_steps[0].content["previous_status"], "running")
