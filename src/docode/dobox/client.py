from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from .types import AgentSession, CommandResult, FileResult, ProjectSandbox


class DoBoxClient:
    """Async client for DoBox project-level sandbox APIs.

    The client never accepts a container id. DoCode knows only the project id;
    DoBox maps that project to its single owned sandbox.
    """

    def __init__(self, base_url: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def create_project(
        self,
        *,
        name: str,
        repo_url: str | None = None,
        branch: str | None = None,
        image: str | None = None,
        network_mode: str | None = None,
    ) -> ProjectSandbox:
        data = await self._request(
            "POST",
            "/api/projects",
            json={
                "name": name,
                "repo_url": repo_url,
                "branch": branch,
                "image": image,
                "network_mode": network_mode,
            },
            timeout=300,
        )
        project = data.get("project", data)
        sandbox = data.get("sandbox") or project.get("sandbox") or {}
        return ProjectSandbox(
            project_id=str(project["id"]),
            sandbox_id=str(sandbox["id"]) if sandbox.get("id") is not None else None,
            raw=data,
        )

    async def get_project(self, project_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/api/projects/{project_id}")

    async def delete_project(self, project_id: str) -> None:
        await self._request("DELETE", f"/api/projects/{project_id}")

    async def create_agent_session(self, project_id: str, name: str) -> AgentSession:
        data = await self._request("POST", f"/api/projects/{project_id}/agent/sessions", json={"name": name})
        return AgentSession(session_id=str(data["id"]), raw=data)

    async def run_command(
        self,
        project_id: str,
        command: str | list[str],
        cwd: str = "/workspace",
        timeout_sec: int = 120,
        output_limit: int = 1_000_000,
        agent_session_id: str | None = None,
    ) -> CommandResult:
        payload = {
            "command": command,
            "working_dir": cwd,
            "timeout_sec": timeout_sec,
            "output_limit": output_limit,
            "agent_session_id": session_payload_id(agent_session_id),
        }
        data = await self._request("POST", f"/api/projects/{project_id}/exec", json=payload, timeout=timeout_sec + 10)
        return CommandResult(output=str(data.get("output", "")), exit_code=int(data.get("exit_code", 0)), truncated=bool(data.get("truncated", False)))

    async def read_file(self, project_id: str, path: str, agent_session_id: str | None = None) -> FileResult:
        data = await self._request("POST", f"/api/projects/{project_id}/files/read", json={"path": path, "agent_session_id": session_payload_id(agent_session_id)})
        return FileResult(
            content=str(data.get("content", "")),
            path=str(data["path"]) if data.get("path") is not None else None,
            file_name=str(data["file_name"]) if data.get("file_name") is not None else None,
            bytes_read=int(data["bytes"]) if data.get("bytes") is not None else None,
            truncated=bool(data.get("truncated", False)),
        )

    async def write_file(self, project_id: str, path: str, content: str, agent_session_id: str | None = None) -> None:
        await self._request("POST", f"/api/projects/{project_id}/files/write", json={"path": path, "content": content, "agent_session_id": session_payload_id(agent_session_id)})

    async def list_files(self, project_id: str, path: str = ".", agent_session_id: str | None = None) -> CommandResult:
        data = await self._request("POST", f"/api/projects/{project_id}/files/list", json={"path": path, "agent_session_id": session_payload_id(agent_session_id)})
        return CommandResult(output=str(data.get("output", "")), exit_code=int(data.get("exit_code", 0)), truncated=bool(data.get("truncated", False)))

    async def search(self, project_id: str, query: str, path: str = ".", agent_session_id: str | None = None) -> CommandResult:
        data = await self._request("POST", f"/api/projects/{project_id}/files/search", json={"query": query, "path": path, "agent_session_id": session_payload_id(agent_session_id)})
        return CommandResult(output=str(data.get("output", "")), exit_code=int(data.get("exit_code", 0)), truncated=bool(data.get("truncated", False)))

    async def git_status(self, project_id: str, agent_session_id: str | None = None) -> CommandResult:
        path = project_path(project_id, "git/status", agent_session_id=agent_session_id)
        data = await self._request("GET", path)
        return CommandResult(
            output=str(data.get("status", data.get("output", ""))),
            exit_code=int(data.get("exit_code", 0)),
            truncated=bool(data.get("truncated", False)),
        )

    async def git_diff(self, project_id: str, agent_session_id: str | None = None) -> str:
        return (await self.git_diff_result(project_id, agent_session_id=agent_session_id)).output

    async def git_diff_result(self, project_id: str, agent_session_id: str | None = None) -> CommandResult:
        path = project_path(project_id, "git/diff", agent_session_id=agent_session_id)
        data = await self._request("GET", path)
        return CommandResult(
            output=str(data.get("diff", data.get("output", ""))),
            exit_code=int(data.get("exit_code", 0)),
            truncated=bool(data.get("truncated", False)),
        )

    async def git_commit(self, project_id: str, message: str, agent_session_id: str | None = None) -> CommandResult:
        data = await self._request("POST", f"/api/projects/{project_id}/git/commit", json={"message": message, "agent_session_id": session_payload_id(agent_session_id)})
        return CommandResult(output=str(data.get("output", "")), exit_code=int(data.get("exit_code", 0)), truncated=bool(data.get("truncated", False)))

    async def preview(self, project_id: str, port: int, agent_session_id: str | None = None) -> dict[str, Any]:
        return await self._request("POST", f"/api/projects/{project_id}/preview", json={"port": port, "agent_session_id": session_payload_id(agent_session_id)})

    async def logs(self, project_id: str, tail: str = "200", agent_session_id: str | None = None) -> str:
        path = project_path(project_id, "logs", tail=tail, agent_session_id=agent_session_id)
        data = await self._request("GET", path)
        return str(data.get("logs", ""))

    async def archive_workspace(self, project_id: str, agent_session_id: str | None = None) -> bytes:
        import httpx

        path = project_path(project_id, "artifacts/archive", agent_session_id=agent_session_id)
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self.headers)
            raise_for_status(response, "GET", path)
            return response.content

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                json={k: v for k, v in (json or {}).items() if v is not None},
                headers=self.headers,
            )
            raise_for_status(response, method, path)
            if not response.content:
                return {}
            payload = response.json()
            return payload if isinstance(payload, dict) else {"data": payload}


def session_payload_id(agent_session_id: str | None) -> int | None:
    if not agent_session_id:
        return None
    try:
        return int(agent_session_id)
    except ValueError:
        return None


def project_path(project_id: str, suffix: str, **query: object) -> str:
    normalized_suffix = suffix.strip("/")
    path = f"/api/projects/{project_id}/{normalized_suffix}"
    params = {key: value for key, value in query.items() if value is not None and value != ""}
    if not params:
        return path
    return f"{path}?{urlencode(params)}"


def raise_for_status(response: Any, method: str, path: str) -> None:
    if not getattr(response, "is_error", False):
        return
    status_code = getattr(response, "status_code", "unknown")
    detail = str(getattr(response, "text", "") or "").strip()
    if len(detail) > 2000:
        detail = detail[:2000] + "...[truncated]"
    suffix = f": {detail}" if detail else ""
    raise RuntimeError(f"{method} {path} failed with HTTP {status_code}{suffix}")
