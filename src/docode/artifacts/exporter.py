from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from docode.agent.verifier import VerificationResult
from docode.artifacts.github import GitHubExportRequest, GitHubExporter
from docode.artifacts.patch import changed_files_from_diff
from docode.artifacts.zip import zip_files
from docode.dobox.types import CommandResult
from docode.storage.models import CodingJob, DocodeArtifact, DocodeStep
from docode.storage.repository import JobRepository
from docode.storage.step_redaction import redacted_step_content


def terminal_artifact_id(artifacts: list[DocodeArtifact]) -> str | None:
    for artifact in artifacts:
        if artifact.kind == "result":
            return artifact.id
    return artifacts[0].id if artifacts else None


class ArtifactExporter:
    def __init__(
        self,
        artifact_dir: Path,
        repository: JobRepository,
        workspace_archive_provider: Callable[[], Awaitable[bytes]] | None = None,
        commit_provider: Callable[[str], Awaitable[CommandResult]] | None = None,
        github_exporter: GitHubExporter | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.repository = repository
        self.workspace_archive_provider = workspace_archive_provider
        self.commit_provider = commit_provider
        self.github_exporter = github_exporter

    async def export_success(self, job: CodingJob, verification: VerificationResult, summary: str) -> list[DocodeArtifact]:
        job_dir = self.artifact_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        report_path = job_dir / "final_report.md"
        report = self._render_report(job, verification, summary)
        report_path.write_text(report, encoding="utf-8")
        created_files.append(report_path)
        report_artifact = await self.repository.add_artifact(job.id, "report", str(report_path), report_path.stat().st_size)

        artifacts: list[DocodeArtifact] = []
        has_patch = exportable_patch(verification.git_diff, False)
        patch_path: Path | None = None
        if has_patch:
            patch_path = job_dir / "patch.diff"
            patch_path.write_text(verification.git_diff, encoding="utf-8")
            created_files.append(patch_path)
            patch = await self.repository.add_artifact(job.id, "patch", str(patch_path), patch_path.stat().st_size)
            artifacts.append(patch)

        artifacts.append(report_artifact)
        verification_log = self._render_verification_log(verification)
        has_log = False
        if verification_log:
            log_path = job_dir / "test_log.txt"
            log_path.write_text(verification_log, encoding="utf-8")
            created_files.append(log_path)
            log_artifact = await self.repository.add_artifact(job.id, "log", str(log_path), log_path.stat().st_size)
            artifacts.append(log_artifact)
            has_log = True

        commit_result = await self._commit_if_requested(job, summary)
        has_commit = False
        if commit_result is not None:
            commit_path = job_dir / "commit.txt"
            commit_path.write_text(commit_result.output, encoding="utf-8")
            created_files.append(commit_path)
            commit_artifact = await self.repository.add_artifact(job.id, "commit", str(commit_path), commit_path.stat().st_size)
            artifacts.append(commit_artifact)
            has_commit = True

        pr_result = await self._pr_if_requested(job, verification, summary, patch_path)
        has_pull_request = False
        if pr_result is not None:
            pr_path = job_dir / "pull_request.txt"
            pr_path.write_text(pr_result, encoding="utf-8")
            created_files.append(pr_path)
            pr_artifact = await self.repository.add_artifact(job.id, "pull_request", str(pr_path), pr_path.stat().st_size)
            artifacts.append(pr_artifact)
            has_pull_request = True

        has_archive = False
        archive = await self._workspace_archive()
        if archive:
            archive_path = job_dir / "workspace.tar"
            archive_path.write_bytes(archive)
            created_files.append(archive_path)
            archive_artifact = await self.repository.add_artifact(job.id, "archive", str(archive_path), archive_path.stat().st_size)
            artifacts.append(archive_artifact)
            has_archive = True

        result_path = job_dir / "result.json"
        result_payload = self._result_payload(
            job,
            verification,
            summary,
            has_patch=has_patch,
            has_log=has_log,
            has_commit=has_commit,
            has_pull_request=has_pull_request,
            has_archive=has_archive,
        )
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created_files.append(result_path)
        result_artifact = await self.repository.add_artifact(job.id, "result", str(result_path), result_path.stat().st_size)
        artifacts.append(result_artifact)

        zip_path = job_dir / "workspace.zip"
        zip_size = zip_files(zip_path, created_files, job_dir)
        zip_artifact = await self.repository.add_artifact(job.id, "zip", str(zip_path), zip_size)
        artifacts.append(zip_artifact)
        return artifacts

    async def export_failure(
        self,
        job: CodingJob,
        reason: str,
        *,
        steps: list[DocodeStep] | None = None,
        git_diff: str = "",
        git_diff_truncated: bool = False,
    ) -> list[DocodeArtifact]:
        job_dir = self.artifact_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        artifacts: list[DocodeArtifact] = []
        has_patch = exportable_patch(git_diff, git_diff_truncated)
        if has_patch:
            patch_path = job_dir / "patch.diff"
            patch_path.write_text(git_diff, encoding="utf-8")
            created_files.append(patch_path)
            artifacts.append(await self.repository.add_artifact(job.id, "patch", str(patch_path), patch_path.stat().st_size))

        report_path = job_dir / "failure_report.md"
        report_path.write_text(self._render_failure_report(job, reason, git_diff, git_diff_truncated), encoding="utf-8")
        created_files.append(report_path)
        artifacts.append(await self.repository.add_artifact(job.id, "report", str(report_path), report_path.stat().st_size))

        if steps:
            log_path = job_dir / "failure_log.txt"
            log_path.write_text(self._render_failure_log(steps), encoding="utf-8")
            created_files.append(log_path)
            artifacts.append(await self.repository.add_artifact(job.id, "log", str(log_path), log_path.stat().st_size))

        has_archive = False
        archive = await self._workspace_archive()
        if archive:
            archive_path = job_dir / "workspace.tar"
            archive_path.write_bytes(archive)
            created_files.append(archive_path)
            artifacts.append(await self.repository.add_artifact(job.id, "archive", str(archive_path), archive_path.stat().st_size))
            has_archive = True

        result_path = job_dir / "result.json"
        result_payload = self._failure_result_payload(
            job,
            reason,
            git_diff,
            git_diff_truncated=git_diff_truncated,
            has_log=bool(steps),
            has_patch=has_patch,
            has_archive=has_archive,
        )
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created_files.append(result_path)
        result_artifact = await self.repository.add_artifact(job.id, "result", str(result_path), result_path.stat().st_size)
        artifacts.append(result_artifact)

        zip_path = job_dir / "workspace.zip"
        zip_size = zip_files(zip_path, created_files, job_dir)
        artifacts.append(await self.repository.add_artifact(job.id, "zip", str(zip_path), zip_size))
        return artifacts

    async def export_stopped(
        self,
        job: CodingJob,
        reason: str,
        *,
        steps: list[DocodeStep] | None = None,
        git_diff: str = "",
        git_diff_truncated: bool = False,
    ) -> list[DocodeArtifact]:
        job_dir = self.artifact_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        artifacts: list[DocodeArtifact] = []
        has_patch = exportable_patch(git_diff, git_diff_truncated)
        if has_patch:
            patch_path = job_dir / "patch.diff"
            patch_path.write_text(git_diff, encoding="utf-8")
            created_files.append(patch_path)
            artifacts.append(await self.repository.add_artifact(job.id, "patch", str(patch_path), patch_path.stat().st_size))

        report_path = job_dir / "stopped_report.md"
        report_path.write_text(self._render_terminal_report(job, "stopped", reason, git_diff, git_diff_truncated), encoding="utf-8")
        created_files.append(report_path)
        artifacts.append(await self.repository.add_artifact(job.id, "report", str(report_path), report_path.stat().st_size))

        if steps:
            log_path = job_dir / "stopped_log.txt"
            log_path.write_text(self._render_step_log(steps), encoding="utf-8")
            created_files.append(log_path)
            artifacts.append(await self.repository.add_artifact(job.id, "log", str(log_path), log_path.stat().st_size))

        has_archive = False
        archive = await self._workspace_archive()
        if archive:
            archive_path = job_dir / "workspace.tar"
            archive_path.write_bytes(archive)
            created_files.append(archive_path)
            artifacts.append(await self.repository.add_artifact(job.id, "archive", str(archive_path), archive_path.stat().st_size))
            has_archive = True

        result_path = job_dir / "result.json"
        result_payload = self._terminal_result_payload(
            job,
            status="stopped",
            reason=reason,
            git_diff=git_diff,
            git_diff_truncated=git_diff_truncated,
            report_filename="stopped_report.md",
            log_filename="stopped_log.txt" if steps else None,
            has_patch=has_patch,
            has_archive=has_archive,
        )
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created_files.append(result_path)
        result_artifact = await self.repository.add_artifact(job.id, "result", str(result_path), result_path.stat().st_size)
        artifacts.append(result_artifact)

        zip_path = job_dir / "workspace.zip"
        zip_size = zip_files(zip_path, created_files, job_dir)
        artifacts.append(await self.repository.add_artifact(job.id, "zip", str(zip_path), zip_size))
        return artifacts

    async def _commit_if_requested(self, job: CodingJob, summary: str) -> CommandResult | None:
        if job.artifact_mode not in {"commit", "pr"} or self.commit_provider is None:
            return None
        message = f"DoCode: {summary or job.instruction}".strip()
        result = await self.commit_provider(message[:200])
        if result.exit_code != 0:
            return CommandResult(output=f"commit failed with exit {result.exit_code}\n{result.output}", exit_code=result.exit_code)
        return result

    async def _pr_if_requested(self, job: CodingJob, verification: VerificationResult, summary: str, patch_path: Path | None) -> str | None:
        if job.artifact_mode != "pr":
            return None
        if patch_path is None:
            return "status=skipped\nreason=patch_missing\n"
        if self.github_exporter is None:
            return "status=skipped\nreason=github_export_not_configured\n"
        repo = job.github_repo or github_repo_from_url(job.repo_url)
        if not repo:
            return "status=skipped\nreason=github_repo_missing\n"
        title = f"DoCode: {summary or job.instruction}".strip()[:200]
        result = await self.github_exporter.export_pull_request(
            GitHubExportRequest(
                repo=repo,
                branch=f"docode/{job.id}",
                base_branch=job.base_branch,
                title=title,
                body=self._render_report(job, verification, summary),
                patch_path=str(patch_path),
            )
        )
        return (
            f"status={result.status}\n"
            f"reason={result.reason or ''}\n"
            f"branch_url={result.branch_url or ''}\n"
            f"pull_request_url={result.pull_request_url or ''}\n"
            f"\n{result.output}"
        )

    async def _workspace_archive(self) -> bytes | None:
        if self.workspace_archive_provider is None:
            return None
        try:
            return await self.workspace_archive_provider()
        except Exception:
            return None

    def _render_report(self, job: CodingJob, verification: VerificationResult, summary: str) -> str:
        checks = "\n".join(f"- {line}" for line in self._verification_status_lines(verification)) or "- not run"
        changed_files = changed_files_from_diff(verification.git_diff)
        changed_file_lines = "\n".join(f"- `{path}`" for path in changed_files) or "- none detected"
        return (
            f"# DoCode Final Report\n\n"
            f"Job: `{job.id}`\n\n"
            f"Instruction: {job.instruction}\n\n"
            f"Summary: {summary or verification.reason}\n\n"
            f"Changed files:\n{changed_file_lines}\n\n"
            f"Verification: {verification.reason}\n\n"
            f"Checks:\n{checks}\n"
        )

    def _render_failure_report(self, job: CodingJob, reason: str, git_diff: str, git_diff_truncated: bool) -> str:
        return self._render_terminal_report(job, "failed", reason, git_diff, git_diff_truncated)

    def _render_terminal_report(self, job: CodingJob, status: str, reason: str, git_diff: str, git_diff_truncated: bool = False) -> str:
        changed_files = changed_files_from_diff(git_diff)
        changed_file_lines = "\n".join(f"- `{path}`" for path in changed_files) or "- none detected"
        title = "Failure" if status == "failed" else status.title()
        note = "\nDiff collection was truncated; patch artifact omitted.\n" if git_diff_truncated else ""
        return (
            f"# DoCode {title} Report\n\n"
            f"Job: `{job.id}`\n\n"
            f"Instruction: {job.instruction}\n\n"
            f"Status: {status}\n\n"
            f"Reason: {reason}\n\n"
            f"Changed files:\n{changed_file_lines}\n"
            f"{note}"
        )

    def _render_failure_log(self, steps: list[DocodeStep]) -> str:
        return self._render_step_log(steps)

    def _render_step_log(self, steps: list[DocodeStep]) -> str:
        lines: list[str] = []
        for step in steps:
            content = json.dumps(redacted_step_content(step.content), ensure_ascii=False, indent=2, default=str)
            lines.append(f"## step {step.step_index}: {step.kind}\n\n{content}")
        return "\n\n".join(lines)

    def _render_verification_log(self, verification: VerificationResult) -> str:
        parts: list[str] = []
        for label, result in (
            ("git status", verification.status_result),
            ("tests", verification.test_result),
            ("build", verification.build_result),
            ("lint", verification.lint_result),
        ):
            if result is None:
                continue
            parts.append(f"## {label}\n\n{result.tool}: exit {result.exit_code}\n\n{result.output}")
        return "\n\n".join(parts)

    def _verification_status_lines(self, verification: VerificationResult) -> list[str]:
        lines: list[str] = []
        for label, result in (
            ("git status", verification.status_result),
            ("tests", verification.test_result),
            ("build", verification.build_result),
            ("lint", verification.lint_result),
        ):
            if result is None:
                continue
            command = ""
            if result.metadata and result.metadata.get("command"):
                command = f" `{result.metadata['command']}`"
            detected = "detected" if not result.metadata or result.metadata.get("detected", True) else "not detected"
            lines.append(f"{label}{command}: exit {result.exit_code} ({detected})")
        return lines

    def _result_payload(
        self,
        job: CodingJob,
        verification: VerificationResult,
        summary: str,
        *,
        has_patch: bool,
        has_log: bool,
        has_commit: bool,
        has_pull_request: bool,
        has_archive: bool,
    ) -> dict[str, object]:
        changed_files = changed_files_from_diff(verification.git_diff)
        artifacts: dict[str, str] = {
            "report": "final_report.md",
            "result": "result.json",
            "zip": "workspace.zip",
        }
        if has_patch:
            artifacts["patch"] = "patch.diff"
        if has_log:
            artifacts["log"] = "test_log.txt"
        if has_commit:
            artifacts["commit"] = "commit.txt"
        if has_pull_request:
            artifacts["pull_request"] = "pull_request.txt"
        if has_archive:
            artifacts["archive"] = "workspace.tar"
        return {
            "status": "succeeded" if verification.passed else "failed",
            "job_id": job.id,
            "instruction": job.instruction,
            "summary": summary or verification.reason,
            "changed_files": changed_files,
            "checks": self._verification_checks(verification),
            "artifacts": artifacts,
            "verification": {
                "passed": verification.passed,
                "confidence": verification.confidence,
                "reason": verification.reason,
                "required_fixes": verification.required_fixes,
                "llm_judgement": {
                    "passed": verification.llm_judgement.passed,
                    "confidence": verification.llm_judgement.confidence,
                    "reason": verification.llm_judgement.reason,
                    "required_fixes": verification.llm_judgement.required_fixes,
                }
                if verification.llm_judgement
                else None,
            },
        }

    def _failure_result_payload(
        self,
        job: CodingJob,
        reason: str,
        git_diff: str,
        *,
        git_diff_truncated: bool,
        has_log: bool,
        has_patch: bool,
        has_archive: bool,
    ) -> dict[str, object]:
        return self._terminal_result_payload(
            job,
            status="failed",
            reason=reason,
            git_diff=git_diff,
            git_diff_truncated=git_diff_truncated,
            report_filename="failure_report.md",
            log_filename="failure_log.txt" if has_log else None,
            has_patch=has_patch,
            has_archive=has_archive,
        )

    def _terminal_result_payload(
        self,
        job: CodingJob,
        *,
        status: str,
        reason: str,
        git_diff: str,
        git_diff_truncated: bool,
        report_filename: str,
        log_filename: str | None,
        has_patch: bool,
        has_archive: bool,
    ) -> dict[str, object]:
        artifacts: dict[str, str] = {
            "report": report_filename,
            "result": "result.json",
            "zip": "workspace.zip",
        }
        if has_patch:
            artifacts["patch"] = "patch.diff"
        if has_archive:
            artifacts["archive"] = "workspace.tar"
        if log_filename is not None:
            artifacts["log"] = log_filename
        return {
            "status": status,
            "job_id": job.id,
            "instruction": job.instruction,
            "summary": reason,
            "failure_reason": reason if status == "failed" else None,
            "stopped_reason": reason if status == "stopped" else None,
            "changed_files": changed_files_from_diff(git_diff),
            "git_diff": {
                "truncated": git_diff_truncated,
                "bytes": len(git_diff.encode("utf-8")),
                "lines": len(git_diff.splitlines()),
            },
            "checks": [],
            "artifacts": artifacts,
            "verification": None,
        }

    def _verification_checks(self, verification: VerificationResult) -> list[dict[str, object]]:
        checks: list[dict[str, object]] = []
        for label, result in (
            ("git_status", verification.status_result),
            ("workspace", verification.workspace_result),
            ("test", verification.test_result),
            ("build", verification.build_result),
            ("lint", verification.lint_result),
        ):
            if result is None:
                continue
            checks.append(
                {
                    "name": label,
                    "tool": result.tool,
                    "command": result.metadata.get("command") if result.metadata else None,
                    "detected": bool(result.metadata.get("detected", True)) if result.metadata else True,
                    "exit_code": result.exit_code,
                    "status": "passed" if result.exit_code == 0 else "failed",
                    "truncated": result.truncated,
                }
            )
        return checks


def github_repo_from_url(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    value = repo_url.strip().removesuffix(".git")
    if value.startswith("git@github.com:"):
        return value.removeprefix("git@github.com:")
    marker = "github.com/"
    if marker in value:
        return value.split(marker, 1)[1]
    if "/" in value and not value.startswith(("http://", "https://", "git@")):
        return value
    return None


def exportable_patch(git_diff: str, git_diff_truncated: bool) -> bool:
    return bool(git_diff.strip()) and not git_diff_truncated
