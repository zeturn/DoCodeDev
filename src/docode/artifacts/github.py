from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitHubExportRequest:
    repo: str
    branch: str
    base_branch: str
    title: str
    body: str
    patch_path: str


@dataclass(frozen=True, slots=True)
class GitHubExportResult:
    status: str
    branch_url: str | None = None
    pull_request_url: str | None = None
    reason: str | None = None
    output: str = ""


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


CommandRunner = Callable[[list[str], Path | None], Awaitable[CommandResult]]


class GitHubExporter:
    """GitHub branch/PR export using the `gh` CLI when enabled."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        work_dir: Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.enabled = enabled
        self.work_dir = work_dir
        self.command_runner = command_runner or run_command

    async def export_pull_request(self, request: GitHubExportRequest) -> GitHubExportResult:
        if not self.enabled:
            return GitHubExportResult(status="skipped", reason="github_export_not_configured")
        patch_path = Path(request.patch_path)
        if not patch_path.exists():
            return GitHubExportResult(status="failed", reason="patch_file_missing")

        root = self.work_dir
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="docode-gh-")
            root = Path(temp_dir.name)
        root.mkdir(parents=True, exist_ok=True)
        clone_dir = root / sanitize_branch(request.branch)
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        try:
            commands = [
                ["gh", "repo", "clone", request.repo, str(clone_dir), "--", "--depth", "1", "--branch", request.base_branch],
                ["git", "checkout", "-B", request.branch],
                ["git", "apply", str(patch_path)],
                ["git", "status", "--short"],
                ["git", "add", "-A"],
                ["git", "commit", "-m", request.title],
                ["git", "push", "-u", "origin", request.branch],
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    request.repo,
                    "--base",
                    request.base_branch,
                    "--head",
                    request.branch,
                    "--title",
                    request.title,
                    "--body",
                    request.body,
                ],
            ]
            output: list[str] = []
            for index, command in enumerate(commands):
                cwd = None if index == 0 else clone_dir
                result = await self.command_runner(command, cwd)
                output.append(render_command(command, result))
                if result.exit_code != 0:
                    return GitHubExportResult(status="failed", reason=f"command_failed:{command[0]}", output="\n".join(output))
                if command[:3] == ["git", "status", "--short"] and not result.stdout.strip():
                    return GitHubExportResult(status="skipped", reason="no_changes_after_patch", output="\n".join(output))

            pr_url = last_nonempty_stdout(result.stdout)
            branch_url = f"https://github.com/{request.repo}/tree/{request.branch}"
            return GitHubExportResult(status="created", branch_url=branch_url, pull_request_url=pr_url, output="\n".join(output))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


async def run_command(command: list[str], cwd: Path | None) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return CommandResult(stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"), process.returncode)


def sanitize_branch(branch: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in branch).strip("-") or "docode-branch"


def render_command(command: list[str], result: CommandResult) -> str:
    return (
        f"$ {' '.join(command)}\n"
        f"exit_code={result.exit_code}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )


def last_nonempty_stdout(stdout: str) -> str | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None

