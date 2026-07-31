from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from docode.agent.prompts import DOCODE_SYSTEM_PROMPT
from docode.agent.failure_taxonomy import FailureCategory, TerminalResult
from tests.runtime_v2_release_eval.definitions import CASES
from tests.runtime_v2_release_eval.hashing import sha256_file, sha256_tree, write_json


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dobox-url", default=os.getenv("DOCODE_DOBOX_BASE_URL", "http://localhost:3000"))
    parser.add_argument("--apicred-url", default=os.getenv("DOCODE_APICRED_BASE_URL"))
    args = parser.parse_args()
    output = Path(args.output)
    root = Path(__file__).parent
    reasons: list[str] = []
    if not os.getenv("DEEPSEEK_API_KEY"):
        reasons.append("DEEPSEEK_API_KEY missing")
    if shutil.which("docker") is None:
        reasons.append("Docker CLI missing")
    else:
        docker = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=30, check=False
        )
        if docker.returncode:
            reasons.append("Docker daemon unavailable")
    free_bytes = shutil.disk_usage(output.parent.resolve()).free
    if free_bytes < 5 * 1024**3:
        reasons.append(f"insufficient disk space: {free_bytes} bytes free")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(args.dobox_url.rstrip("/") + "/health")
            response.raise_for_status()
    except Exception as exc:
        reasons.append(f"DoBox health failed: {type(exc).__name__}: {exc}")
    if not args.apicred_url:
        reasons.append("DOCODE_APICRED_BASE_URL missing")
    else:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(args.apicred_url.rstrip("/") + "/health")
                response.raise_for_status()
        except Exception as exc:
            reasons.append(f"APICred health failed: {type(exc).__name__}: {exc}")
    fixture_root = root / "fixtures"
    hidden_root = root / "hidden"
    if not fixture_root.is_dir() or not hidden_root.is_dir():
        reasons.append("frozen fixture or hidden directory missing")
    crawler_count = sum(case.category == "crawler" for case in CASES)
    repository_count = sum(case.category == "repository" for case in CASES)
    if (crawler_count, repository_count) != (8, 3):
        reasons.append(
            f"invalid release case inventory: {crawler_count} crawler, "
            f"{repository_count} repository"
        )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "commit_sha": commit,
        "prompt_sha256": __import__("hashlib").sha256(DOCODE_SYSTEM_PROMPT.encode()).hexdigest(),
        "definitions_sha256": sha256_file(root / "definitions.py"),
        "fixture_manifest_sha256": sha256_tree(fixture_root) if fixture_root.is_dir() else None,
        "hidden_checker_sha256": sha256_tree(hidden_root) if hidden_root.is_dir() else None,
        "provider": os.getenv("DOCODE_PROVIDER", "deepseek"),
        "model": os.getenv("DOCODE_MODEL", "deepseek-chat"),
        "budgets": {"max_iterations": 24, "max_tool_calls": 48, "max_runtime_seconds": 600},
        "cases": [case.name for case in CASES],
    }
    write_json(output / "manifest.json", manifest)
    if reasons:
        terminal = TerminalResult("failed", FailureCategory.ENVIRONMENT_FAILURE, "; ".join(reasons), harness_valid=False)
        write_json(output / "terminal_result.json", terminal.to_dict())
        return 2
    # The suite refuses to emit a success result until every frozen fixture and
    # independent checker is present and the full JobRunnerService harness runs.
    terminal = TerminalResult("failed", FailureCategory.HARNESS_FAILURE, "release harness execution not implemented", harness_valid=False)
    write_json(output / "terminal_result.json", terminal.to_dict())
    return 3


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
