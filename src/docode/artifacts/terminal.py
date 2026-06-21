from __future__ import annotations

from pathlib import Path

from docode.artifacts.exporter import ArtifactExporter, terminal_artifact_id
from docode.storage.models import CodingJob
from docode.storage.repository import JobRepository


async def export_stopped_artifacts(repository: JobRepository, artifact_dir: Path, job: CodingJob, reason: str) -> str | None:
    try:
        artifacts = await ArtifactExporter(artifact_dir, repository).export_stopped(
            job,
            reason,
            steps=await repository.list_steps(job.id),
        )
    except Exception as exc:
        await repository.add_step(job.id, "system", {"type": "stopped_export_failed", "reason": reason, "error": str(exc)})
        return None
    artifact_id = terminal_artifact_id(artifacts)
    await repository.add_step(job.id, "system", {"type": "stopped_artifacts_exported", "reason": reason, "artifact_id": artifact_id})
    return artifact_id
