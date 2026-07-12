from __future__ import annotations

from docode.storage.models import CodingJob, JobStatus
from docode.storage.repository import InMemoryJobRepository


class RecordingRepository(InMemoryJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.status_updates: list[JobStatus] = []

    async def update_job(self, job_id: str, **changes: object) -> CodingJob:
        updated = await super().update_job(job_id, **changes)
        if "status" in changes:
            self.status_updates.append(updated.status)
        return updated
