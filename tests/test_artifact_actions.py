from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from docode.api.artifact_actions import ArtifactDownloadError, artifact_descriptor, resolve_artifact_download_path
from docode.storage.models import CodingJob, new_id
from docode.storage.repository import InMemoryJobRepository


class ArtifactActionTests(IsolatedAsyncioTestCase):
    async def test_resolves_owned_artifact_path(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            path = Path(tmp) / "result.json"
            path.write_text("{}", encoding="utf-8")
            artifact = await repo.add_artifact(job.id, "result", str(path), path.stat().st_size)

            resolved = await resolve_artifact_download_path(repo, artifact.id, "user-1")

            self.assertEqual(resolved, path)

    async def test_hides_artifact_from_wrong_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            path = Path(tmp) / "result.json"
            path.write_text("{}", encoding="utf-8")
            artifact = await repo.add_artifact(job.id, "result", str(path), path.stat().st_size)

            with self.assertRaises(ArtifactDownloadError) as raised:
                await resolve_artifact_download_path(repo, artifact.id, "user-2")

            self.assertEqual(raised.exception.status_code, 404)
            self.assertEqual(raised.exception.detail, "artifact not found")

    async def test_rejects_missing_artifact_record(self) -> None:
        repo = InMemoryJobRepository()

        with self.assertRaises(ArtifactDownloadError) as raised:
            await resolve_artifact_download_path(repo, "art_missing", "user-1")

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail, "artifact not found")

    async def test_rejects_missing_artifact_file(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            missing_path = Path(tmp) / "missing.zip"
            artifact = await repo.add_artifact(job.id, "zip", str(missing_path), 123)

            with self.assertRaises(ArtifactDownloadError) as raised:
                await resolve_artifact_download_path(repo, artifact.id, "user-1")

            self.assertEqual(raised.exception.status_code, 404)
            self.assertEqual(raised.exception.detail, "artifact file missing")

    async def test_artifact_descriptor_omits_local_path_and_includes_download_url(self) -> None:
        with TemporaryDirectory() as tmp:
            repo = InMemoryJobRepository()
            job = await repo.create_job(CodingJob(id=new_id("job"), user_id="user-1", instruction="done"))
            path = Path(tmp) / "workspace.zip"
            path.write_bytes(b"zip")
            artifact = await repo.add_artifact(job.id, "zip", str(path), path.stat().st_size)

            descriptor = artifact_descriptor(artifact)

            self.assertNotIn("path", descriptor)
            self.assertEqual(descriptor["id"], artifact.id)
            self.assertEqual(descriptor["job_id"], job.id)
            self.assertEqual(descriptor["kind"], "zip")
            self.assertEqual(descriptor["filename"], "workspace.zip")
            self.assertEqual(descriptor["size_bytes"], 3)
            self.assertEqual(descriptor["download_url"], f"/v1/artifacts/{artifact.id}/download")
            self.assertEqual(descriptor["created_at"], artifact.created_at)


if __name__ == "__main__":
    import unittest

    unittest.main()
