from __future__ import annotations

from docode.storage.models import JobStatus
from docode.storage.repository import JobRepository


INTERRUPTED_STATUSES = {JobStatus.PREPARING, JobStatus.RUNNING, JobStatus.VERIFYING}


async def recover_interrupted_jobs(repository: JobRepository) -> list[str]:
    """Find jobs that need worker attention after process restart.

    Active interrupted jobs are moved back to queued. Stopped jobs with an
    existing sandbox but no artifact stay stopped and are re-enqueued so the
    worker can export final sandbox evidence and apply cleanup policy.
    """
    recovered: list[str] = []
    for job in await repository.list_jobs(INTERRUPTED_STATUSES):
        await repository.update_job(job.id, status=JobStatus.QUEUED)
        await repository.add_step(
            job.id,
            "system",
            {"type": "worker_recovered_after_restart", "previous_status": job.status.value},
        )
        recovered.append(job.id)
    for job in await repository.list_jobs({JobStatus.STOPPED}):
        if not job.dobox_project_id or job.artifact_id:
            continue
        await repository.add_step(
            job.id,
            "system",
            {"type": "worker_recovered_stopped_finalization", "previous_status": job.status.value},
        )
        recovered.append(job.id)
    return recovered
