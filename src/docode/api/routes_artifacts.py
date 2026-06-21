from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from docode.api.artifact_actions import ArtifactDownloadError, resolve_artifact_download_path
from docode.api.auth import UserContext, get_user_context
from docode.storage.repository import JobRepository


def make_artifacts_router(repository: JobRepository, user_dependency=get_user_context) -> APIRouter:
    router = APIRouter(prefix="/v1/artifacts", tags=["artifacts"])

    @router.get("/{artifact_id}/download")
    async def download_artifact(artifact_id: str, user: UserContext = Depends(user_dependency)) -> FileResponse:
        try:
            path = await resolve_artifact_download_path(repository, artifact_id, user.user_id)
        except ArtifactDownloadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        return FileResponse(path)

    return router
