from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel


ROOT = Path(os.environ.get("MOCK_DOBOX_ROOT", ".docode/mock_dobox")).resolve()
PROJECTS: dict[str, Path] = {}

app = FastAPI(title="Mock DoBox")


class ProjectCreate(BaseModel):
    name: str
    repo_url: str | None = None
    branch: str | None = None
    image: str | None = None
    network_mode: str | None = None


class SessionCreate(BaseModel):
    name: str


class ExecRequest(BaseModel):
    command: str | list[str]
    working_dir: str = "/workspace"
    timeout_sec: int = 120
    output_limit: int = 1_000_000
    agent_session_id: int | None = None


class FileRead(BaseModel):
    path: str
    agent_session_id: int | None = None


class FileWrite(BaseModel):
    path: str
    content: str
    agent_session_id: int | None = None


class FileList(BaseModel):
    path: str = "."
    agent_session_id: int | None = None


class FileSearch(BaseModel):
    query: str
    path: str = "."
    agent_session_id: int | None = None


class CommitRequest(BaseModel):
    message: str
    agent_session_id: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    project_id = uuid.uuid4().hex[:12]
    workspace = ROOT / project_id / "workspace"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if payload.repo_url:
        source = Path(payload.repo_url)
        if source.exists():
            shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(".git"))
            run(["git", "init"], workspace)
            run(["git", "config", "user.email", "mock-dobox@example.local"], workspace)
            run(["git", "config", "user.name", "Mock DoBox"], workspace)
            run(["git", "add", "."], workspace)
            run(["git", "commit", "-m", "initial"], workspace)
        else:
            result = run(["git", "clone", payload.repo_url, str(workspace)], ROOT, timeout=300)
            if result.returncode != 0:
                raise HTTPException(status_code=400, detail=result.stdout[-4000:])
            if payload.branch:
                run(["git", "checkout", payload.branch], workspace)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        run(["git", "init"], workspace)
        run(["git", "config", "user.email", "mock-dobox@example.local"], workspace)
        run(["git", "config", "user.name", "Mock DoBox"], workspace)
    PROJECTS[project_id] = workspace
    return {"project": {"id": project_id}, "sandbox": {"id": f"sandbox-{project_id}"}}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    return {"id": project_id, "workspace": str(workspace)}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, str]:
    workspace = project_workspace(project_id)
    shutil.rmtree(workspace.parent, ignore_errors=True)
    PROJECTS.pop(project_id, None)
    return {"status": "deleted"}


@app.post("/api/projects/{project_id}/agent/sessions")
def create_agent_session(project_id: str, payload: SessionCreate) -> dict[str, Any]:
    project_workspace(project_id)
    return {"id": 1, "name": payload.name}


@app.post("/api/projects/{project_id}/exec")
def exec_command(project_id: str, payload: ExecRequest) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    cwd = resolve_workspace_path(workspace, payload.working_dir)
    command = payload.command if isinstance(payload.command, list) else payload.command
    result = run(command, cwd, timeout=payload.timeout_sec)
    output = result.stdout
    truncated = False
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) > payload.output_limit:
        output = encoded[: payload.output_limit].decode("utf-8", errors="replace")
        truncated = True
    return {"output": output, "exit_code": result.returncode, "truncated": truncated}


@app.post("/api/projects/{project_id}/files/read")
def read_file(project_id: str, payload: FileRead) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    path = resolve_workspace_path(workspace, payload.path)
    content = path.read_text(encoding="utf-8")
    return {"path": str(path), "file_name": path.name, "bytes": len(content.encode("utf-8")), "content": content}


@app.post("/api/projects/{project_id}/files/write")
def write_file(project_id: str, payload: FileWrite) -> dict[str, str]:
    workspace = project_workspace(project_id)
    path = resolve_workspace_path(workspace, payload.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.content, encoding="utf-8")
    return {"status": "ok"}


@app.post("/api/projects/{project_id}/files/list")
def list_files(project_id: str, payload: FileList) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    path = resolve_workspace_path(workspace, payload.path)
    rows = []
    for item in sorted(path.rglob("*") if path.is_dir() else [path]):
        if ".git" in item.parts:
            continue
        rel = item.relative_to(workspace).as_posix()
        rows.append(rel + ("/" if item.is_dir() else ""))
    return {"output": "\n".join(rows), "exit_code": 0, "truncated": False}


@app.post("/api/projects/{project_id}/files/search")
def search_files(project_id: str, payload: FileSearch) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    root = resolve_workspace_path(workspace, payload.path)
    matches: list[str] = []
    for item in root.rglob("*") if root.is_dir() else [root]:
        if not item.is_file() or ".git" in item.parts:
            continue
        try:
            for line_no, line in enumerate(item.read_text(encoding="utf-8").splitlines(), 1):
                if payload.query in line:
                    matches.append(f"{item.relative_to(workspace).as_posix()}:{line_no}:{line}")
        except UnicodeDecodeError:
            continue
    return {"output": "\n".join(matches), "exit_code": 0, "truncated": False}


@app.get("/api/projects/{project_id}/git/status")
def git_status(project_id: str) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    result = run(["git", "status", "--short"], workspace)
    return {"status": result.stdout, "exit_code": result.returncode, "truncated": False}


@app.get("/api/projects/{project_id}/git/diff")
def git_diff(project_id: str) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    result = run(["git", "diff"], workspace)
    return {"diff": result.stdout, "exit_code": result.returncode, "truncated": False}


@app.post("/api/projects/{project_id}/git/commit")
def git_commit(project_id: str, payload: CommitRequest) -> dict[str, Any]:
    workspace = project_workspace(project_id)
    run(["git", "add", "."], workspace)
    result = run(["git", "commit", "-m", payload.message], workspace)
    return {"output": result.stdout, "exit_code": result.returncode, "truncated": False}


@app.post("/api/projects/{project_id}/preview")
def preview(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    project_workspace(project_id)
    return {"status": "preview_descriptor", "port": payload.get("port"), "message": "mock preview"}


@app.get("/api/projects/{project_id}/logs")
def logs(project_id: str) -> dict[str, str]:
    project_workspace(project_id)
    return {"logs": "mock dobox has no container logs"}


@app.get("/api/projects/{project_id}/artifacts/archive")
def archive(project_id: str) -> Response:
    workspace = project_workspace(project_id)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for item in workspace.rglob("*"):
            if ".git" in item.parts:
                continue
            tar.add(item, arcname=item.relative_to(workspace))
    return Response(content=buffer.getvalue(), media_type="application/x-tar")


def project_workspace(project_id: str) -> Path:
    workspace = PROJECTS.get(project_id) or ROOT / project_id / "workspace"
    if not workspace.exists():
        raise HTTPException(status_code=404, detail="project not found")
    PROJECTS[project_id] = workspace
    return workspace


def resolve_workspace_path(workspace: Path, raw_path: str) -> Path:
    relative = raw_path.replace("\\", "/")
    if relative.startswith("/workspace"):
        relative = relative[len("/workspace") :].lstrip("/")
    relative = relative.lstrip("/")
    path = (workspace / relative).resolve()
    if path != workspace and workspace not in path.parents:
        raise HTTPException(status_code=400, detail="path escapes workspace")
    return path


def run(command: str | list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    if isinstance(command, list) and len(command) >= 3 and command[0] == "bash" and command[1] == "-lc":
        return run_bash_compatible(command[2], cwd, timeout)
    shell = isinstance(command, str)
    return subprocess.run(
        command,
        cwd=str(cwd),
        shell=shell,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def run_bash_compatible(command: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    stripped = command.strip()
    file_checks = {
        "test -f pyproject.toml || test -f pytest.ini": ["pyproject.toml", "pytest.ini"],
        "test -f go.mod": ["go.mod"],
        "test -f Cargo.toml": ["Cargo.toml"],
        "test -f pyproject.toml && command -v ruff >/dev/null 2>&1": ["pyproject.toml"],
    }
    if stripped in file_checks:
        exists = any((cwd / filename).is_file() for filename in file_checks[stripped])
        if stripped.endswith("command -v ruff >/dev/null 2>&1"):
            exists = exists and shutil.which("ruff") is not None
        return completed(0 if exists else 1, "")

    if stripped == "pytest":
        return subprocess.run(
            ["python", "-m", "pytest"],
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

    return subprocess.run(
        stripped,
        cwd=str(cwd),
        shell=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)
