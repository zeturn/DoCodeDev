from __future__ import annotations

from pathlib import Path

from docode.storage.models import DocodeArtifact
from docode.storage.repository import JobRepository


class ArtifactDownloadError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


async def resolve_artifact_download_path(repository: JobRepository, artifact_id: str, user_id: str) -> Path:
    artifact = await repository.get_artifact(artifact_id)
    if artifact is None:
        raise ArtifactDownloadError(404, "artifact not found")
    job = await repository.get_job(artifact.job_id)
    if job is None or job.user_id != user_id:
        raise ArtifactDownloadError(404, "artifact not found")
    path = Path(artifact.path)
    if not path.exists():
        raise ArtifactDownloadError(404, "artifact file missing")
    return path


def artifact_descriptor(artifact: DocodeArtifact) -> dict[str, object]:
    return {
        "id": artifact.id,
        "job_id": artifact.job_id,
        "kind": artifact.kind,
        "filename": Path(artifact.path).name,
        "size_bytes": artifact.size_bytes,
        "created_at": artifact.created_at,
        "download_url": f"/v1/artifacts/{artifact.id}/download",
    }
