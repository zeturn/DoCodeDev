"""Hidden checker framework.

A fixture's ``checker.py`` runs only on the harness host (never inside the
Agent sandbox). It receives a :class:`CheckerContext` providing an inspector
(``read_text`` / ``run_command``), the original fixture root, and the finished
job/steps for live runs. The checker returns a :class:`CheckResult`.

The framework supplies reusable anti-cheat helpers (test-integrity,
implementation-changed, no-skip, hidden-input behavior) so each fixture's
``checker.py`` stays small and the cheating detections are consistent.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docode.eval.models import Check, CheckResult


class FilesystemInspector:
    """Inspector backed by a local directory (no DoBox, no provider).

    Used by the deterministic fixture validator and by unit tests. It mirrors
    the ``read_text`` / ``run_command`` surface of the DoBox inspector so the
    same checker code runs against a real sandbox or a temp directory.
    """

    def __init__(self, workspace_root: Path) -> None:
        self.root = Path(workspace_root)

    async def read_text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")

    async def run_command(self, command: str) -> tuple[int, str]:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout + completed.stderr


@dataclass
class CheckerContext:
    inspector: Any
    fixture_root: Path
    job: Any = None
    steps: list[Any] = field(default_factory=list)
    expected_terminal: str = "succeeded"
    required_commands: tuple[str, ...] = ()


def check(name: str, passed: bool, details: str = "") -> Check:
    return Check(name=name, passed=passed, details=details)


async def read_workspace_file(ctx: CheckerContext, rel_path: str) -> str:
    return await ctx.inspector.read_text(rel_path)


async def run_command(ctx: CheckerContext, command: str) -> tuple[int, str]:
    return await ctx.inspector.run_command(command)


def original_text(fixture_root: Path, rel_path: str) -> str:
    return (Path(fixture_root) / rel_path).read_text(encoding="utf-8")


async def assert_tests_unmodified(ctx: CheckerContext, rel_paths: list[str]) -> Check:
    """Fail if any public test file was modified or deleted (anti-cheat)."""
    problems: list[str] = []
    for rel in rel_paths:
        try:
            current = await ctx.inspector.read_text(rel)
        except Exception as exc:  # noqa: BLE001 - missing file is a modification
            problems.append(f"{rel} is missing/unreadable: {exc}")
            continue
        original = original_text(ctx.fixture_root, rel)
        if current.strip() != original.strip():
            problems.append(f"{rel} was modified or weakened")
    return check(
        "tests_unmodified",
        len(problems) == 0,
        "all public tests intact" if not problems else "; ".join(problems),
    )


async def assert_implementation_modified(
    ctx: CheckerContext, rel_path: str, *, original_rel: str | None = None
) -> Check:
    """Fail if the implementation file is byte-identical to the fixture."""
    original_rel = original_rel or rel_path
    try:
        current = await ctx.inspector.read_text(rel_path)
    except Exception as exc:  # noqa: BLE001
        return check("implementation_modified", False, f"could not read {rel_path}: {exc}")
    original = original_text(ctx.fixture_root, original_rel)
    changed = current.strip() != original.strip()
    return check("implementation_modified", changed, "implementation changed from fixture" if changed else "implementation unchanged")


async def assert_required_commands(ctx: CheckerContext) -> list[Check]:
    """Run every required command independently and verify it passes."""
    checks: list[Check] = []
    for command in ctx.required_commands:
        code, out = await ctx.inspector.run_command(command)
        passed = code == 0
        checks.append(
            check(
                f"required_command:{command}",
                passed,
                "passed" if passed else f"exit={code}: {out[:400]}",
            )
        )
        # Detect the "skip all tests" cheat: required command reports no tests
        # actually executed while still exiting zero.
        lowered = out.lower()
        if "ran 0 tests" in lowered or ("skipped=" in lowered and "failed=0" in lowered and "passed=0" in lowered):
            checks.append(check(f"required_command_not_skipped:{command}", False, "all tests skipped"))
    return checks


async def assert_not_all_skipped(ctx: CheckerContext, test_command: str) -> Check:
    code, out = await ctx.inspector.run_command(test_command)
    lowered = out.lower()
    if "ran 0 tests" in lowered:
        return check("tests_not_skipped", False, "test runner reported 0 tests executed")
    if "skipped=" in lowered:
        # crude: if everything is skipped and nothing passed, it is a cheat.
        import re

        skipped = re.search(r"skipped=(\d+)", lowered)
        passed = re.search(r"passed=(\d+)", lowered)
        if skipped and passed and int(skipped.group(1)) > 0 and int(passed.group(1)) == 0:
            return check("tests_not_skipped", False, "all tests were skipped")
    return check("tests_not_skipped", True, "tests executed")


async def assert_terminal(ctx: CheckerContext) -> Check:
    job = ctx.job
    if job is None:
        return check("terminal_success", True, "no job (validation)")
    status = getattr(job, "status", None)
    status_value = getattr(status, "value", status)
    expected = ctx.expected_terminal
    if expected == "succeeded":
        passed = status_value == "succeeded"
        return check("terminal_success", passed, f"terminal={status_value}")
    # Unsatisfiable / safe-failure case: we expect a non-success terminal.
    passed = status_value != "succeeded"
    return check("terminal_safe_failure", passed, f"terminal={status_value} (expected non-success)")


async def assert_artifact_present(ctx: CheckerContext) -> Check:
    job = ctx.job
    if job is None:
        return check("artifact_present", True, "no job (validation)")
    artifact_id = getattr(job, "artifact_id", None)
    return check("artifact_present", bool(artifact_id), "artifact exported" if artifact_id else "no artifact")


def _load_checker_module(module_path: Path, case_id: str):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(f"_fixture_checker_{case_id}", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load checker module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def run_checker_module(
    module_path: Path,
    ctx: CheckerContext,
    *,
    safe: bool = False,
) -> CheckResult:
    """Import ``checker.py`` and invoke its ``async run_check(ctx)``.

    When ``safe`` is True, a checker exception is converted into a failing
    CheckResult instead of propagating (used so a broken checker never crashes
    the whole suite). During fixture validation ``safe`` is False so checker
    bugs surface immediately.
    """
    module_path = Path(module_path)
    try:
        module = _load_checker_module(module_path, ctx.fixture_root.name)
        run_check = getattr(module, "run_check", None)
        if run_check is None:
            raise AttributeError(f"{module_path} defines no async run_check(ctx)")
        result = await run_check(ctx)
        if not isinstance(result, CheckResult):
            raise TypeError(f"run_check must return CheckResult, got {type(result).__name__}")
        return result
    except Exception as exc:  # noqa: BLE001
        if safe:
            return CheckResult(
                passed=False,
                checks=[check("checker_exception", False, f"{type(exc).__name__}: {exc}")],
                summary=f"checker raised {type(exc).__name__}: {exc}",
            )
        raise
