from __future__ import annotations

from datetime import timedelta

from docode.storage.models import JobStatus, utcnow
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


async def recover_stale_active_jobs(repository: JobRepository, stale_after_seconds: int) -> list[str]:
    if stale_after_seconds <= 0:
        return []
    recovered: list[str] = []
    active_jobs = await repository.list_jobs(INTERRUPTED_STATUSES)
    if not active_jobs:
        return recovered
    now = utcnow()
    threshold = timedelta(seconds=stale_after_seconds)
    for job in active_jobs:
        if now - job.updated_at < threshold:
            continue
        await repository.update_job(job.id, status=JobStatus.QUEUED)
        await repository.add_step(
            job.id,
            "system",
            {
                "type": "worker_recovered_stale_job",
                "previous_status": job.status.value,
                "stale_after_seconds": stale_after_seconds,
                "last_updated_at": job.updated_at.isoformat(),
            },
        )
        recovered.append(job.id)
    return recovered
