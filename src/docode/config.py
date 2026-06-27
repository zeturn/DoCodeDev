from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from docode.sandbox import DEFAULT_SANDBOX_NETWORK_MODE, normalize_sandbox_network_mode


@dataclass(slots=True)
class DocodeConfig:
    dobox_base_url: str = "http://localhost:3000"
    dobox_token: str = ""
    dobox_backend_dir: Path = Path("DoBoxDev/backend")
    dobox_start_timeout_seconds: float = 20.0
    apicred_base_url: str = "http://localhost:8103/v1"
    apicred_token: str = ""
    apicred_mode: str = "auto"
    auth_required: bool = False
    database_path: str = ".docode/docode.db"
    artifact_dir: Path = Path(".docode_artifacts")
    default_provider: str = "openai"
    default_model: str = "gpt-5.4"
    max_iterations: int = 50
    max_runtime_seconds: int = 1800
    max_tool_calls: int = 100
    max_llm_tokens: int = 100_000
    max_llm_cost: float | None = None
    command_timeout_seconds: int = 120
    output_limit_bytes: int = 1_000_000
    sandbox_retention: str = "keep"
    sandbox_network_mode: str = DEFAULT_SANDBOX_NETWORK_MODE
    github_export_enabled: bool = False
    github_base_branch: str = "main"
    github_work_dir: Path = Path(".docode/github")


def load_config() -> DocodeConfig:
    dobox_backend_dir = os.getenv("DOCODE_DOBOX_BACKEND_DIR")
    return DocodeConfig(
        dobox_base_url=os.getenv("DOCODE_DOBOX_BASE_URL", "http://localhost:3000"),
        dobox_token=os.getenv("DOCODE_DOBOX_TOKEN", ""),
        dobox_backend_dir=Path(dobox_backend_dir) if dobox_backend_dir else default_dobox_backend_dir(),
        dobox_start_timeout_seconds=float(os.getenv("DOCODE_DOBOX_START_TIMEOUT_SECONDS", "20")),
        apicred_base_url=os.getenv("DOCODE_APICRED_BASE_URL", "http://localhost:8103/v1"),
        apicred_token=os.getenv("DOCODE_APICRED_TOKEN", ""),
        apicred_mode=normalize_apicred_mode(os.getenv("DOCODE_APICRED_MODE", "auto")),
        auth_required=os.getenv("DOCODE_AUTH_REQUIRED", "").lower() in {"1", "true", "yes", "on"},
        database_path=os.getenv("DOCODE_DATABASE_PATH", ".docode/docode.db"),
        artifact_dir=Path(os.getenv("DOCODE_ARTIFACT_DIR", ".docode_artifacts")),
        default_provider=os.getenv("DOCODE_DEFAULT_PROVIDER", "openai"),
        default_model=os.getenv("DOCODE_DEFAULT_MODEL", "gpt-5.4"),
        max_iterations=int(os.getenv("DOCODE_MAX_ITERATIONS", "50")),
        max_runtime_seconds=int(os.getenv("DOCODE_MAX_RUNTIME_SECONDS", "1800")),
        max_tool_calls=int(os.getenv("DOCODE_MAX_TOOL_CALLS", "100")),
        max_llm_tokens=int(os.getenv("DOCODE_MAX_LLM_TOKENS", "100000")),
        max_llm_cost=float(os.environ["DOCODE_MAX_LLM_COST"]) if os.getenv("DOCODE_MAX_LLM_COST") else None,
        command_timeout_seconds=int(os.getenv("DOCODE_COMMAND_TIMEOUT_SECONDS", "120")),
        output_limit_bytes=int(os.getenv("DOCODE_OUTPUT_LIMIT_BYTES", "1000000")),
        sandbox_retention=os.getenv("DOCODE_SANDBOX_RETENTION", "keep"),
        sandbox_network_mode=normalize_sandbox_network_mode(os.getenv("DOCODE_SANDBOX_NETWORK_MODE")),
        github_export_enabled=os.getenv("DOCODE_GITHUB_EXPORT_ENABLED", "").lower() in {"1", "true", "yes", "on"},
        github_base_branch=os.getenv("DOCODE_GITHUB_BASE_BRANCH", "main"),
        github_work_dir=Path(os.getenv("DOCODE_GITHUB_WORK_DIR", ".docode/github")),
    )


def default_dobox_backend_dir() -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd / "DoBoxDev/backend",
        cwd.parent / "DoBoxDev/backend",
        Path("DoBoxDev/backend"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("DoBoxDev/backend")


def normalize_apicred_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    return mode if mode in {"auto", "runtime", "proxy"} else "auto"
