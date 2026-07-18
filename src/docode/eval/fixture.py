"""Fixture model and deterministic, network-free validator.

``validate_fixture`` proves, without any provider, DoBox, or Docker, that:

* the initial workspace is in the expected (buggy / missing-premise) state,
* the gold solution makes the required commands pass and the hidden checker
  pass,
* a set of cheating states (tampered / deleted / skipped public tests, and a
  shallow hardcoded solution) are all rejected by the hidden checker.

The validator copies the workspace into throwaway temp directories and runs the
fixture's ``checker.py`` through :class:`FilesystemInspector`, so the Agent
never sees the checker or the gold solution.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from docode.eval.checker import CheckerContext, FilesystemInspector, run_checker_module
from docode.eval.manifest import FixtureManifest, load_fixture_manifest
from docode.eval.models import FixtureManifestError


@dataclass
class Fixture:
    manifest: FixtureManifest
    fixture_dir: Path
    workspace_dir: Path
    instruction_path: Path
    checker_path: Path
    gold_dir: Path
    naive_dir: Path

    @property
    def id(self) -> str:
        return self.manifest.id


def load_fixture(fixture_dir: Path) -> Fixture:
    fixture_dir = Path(fixture_dir)
    manifest = load_fixture_manifest(fixture_dir)
    workspace_dir = fixture_dir / manifest.workspace
    if not workspace_dir.is_dir():
        raise FixtureManifestError(f"workspace directory not found: {workspace_dir}")
    instruction_path = fixture_dir / manifest.instruction
    if not instruction_path.is_file():
        raise FixtureManifestError(f"instruction file not found: {instruction_path}")
    checker_path = fixture_dir / manifest.checker
    if not checker_path.is_file():
        raise FixtureManifestError(f"checker file not found: {checker_path}")
    return Fixture(
        manifest=manifest,
        fixture_dir=fixture_dir,
        workspace_dir=workspace_dir,
        instruction_path=instruction_path,
        checker_path=checker_path,
        gold_dir=fixture_dir / "gold",
        naive_dir=fixture_dir / "naive",
    )


class FixtureManifestError_local(Exception):
    """Deprecated alias kept for import compatibility."""


def _copy_workspace(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _apply_overlay(workspace: Path, overlay: Path) -> None:
    if not overlay.is_dir():
        return
    for path in overlay.rglob("*"):
        if path.is_file():
            rel = path.relative_to(overlay)
            target = workspace / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _public_test_files(workspace: Path, language: str) -> list[Path]:
    files: list[Path] = []
    if language == "python":
        patterns = ("test_*.py", "*_test.py")
    elif language in ("node", "javascript"):
        patterns = ("test.js", "*.test.js", "test-*.js")
    elif language == "go":
        patterns = ("*_test.go",)
    else:
        patterns = ("test_*.py", "*_test.py")
    for pattern in patterns:
        files.extend(workspace.rglob(pattern))
    # de-dupe
    seen = {p.resolve() for p in files}
    return [Path(p) for p in seen]


def _tamper_modify_tests(workspace: Path, language: str) -> None:
    for test in _public_test_files(workspace, language):
        with test.open("a", encoding="utf-8") as fh:
            fh.write("\n# tampered by validator\n")


def _tamper_delete_tests(workspace: Path, language: str) -> None:
    for test in _public_test_files(workspace, language):
        test.unlink(missing_ok=True)


def _tamper_skip_tests(workspace: Path, language: str) -> None:
    for test in _public_test_files(workspace, language):
        text = test.read_text(encoding="utf-8")
        if language == "python":
            text = re.sub(
                r"^(\s*)def (test_\w+)",
                r"\1@unittest.skip('cheat')\n\1def \2",
                text,
                flags=re.MULTILINE,
            )
        else:
            text = "# tampered by validator (skip cheat)\n" + text
        test.write_text(text, encoding="utf-8")


async def _run_commands(inspector: FilesystemInspector, commands: tuple[str, ...]) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    for command in commands:
        results.append(await inspector.run_command(command))
    return results


class _FakeJob:
    """Minimal job stand-in for checker validation (no repository needed)."""

    def __init__(self, status: str, failure_reason: str | None = None, artifact_id: str | None = None) -> None:
        self.status = status
        self.failure_reason = failure_reason
        self.artifact_id = artifact_id
        self.dobox_project_id = None
        self.dobox_agent_session_id = None


@dataclass
class FixtureValidationReport:
    case_id: str
    ok: bool
    states: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "ok": self.ok, "states": self.states}


async def validate_fixture(fixture: Fixture) -> FixtureValidationReport:
    manifest = fixture.manifest
    language = manifest.language
    states: dict[str, Any] = {}
    ok = True

    if manifest.expected_terminal == "succeeded":
        ok = await _validate_success_fixture(fixture, states) and ok
    else:
        ok = await _validate_unsafe_fixture(fixture, states) and ok

    return FixtureValidationReport(case_id=fixture.id, ok=ok, states=states)


async def _validate_success_fixture(fixture: Fixture, states: dict[str, Any]) -> bool:
    manifest = fixture.manifest
    language = manifest.language
    commands = manifest.required_commands
    ok = True

    # 1. Initial state: required commands must FAIL (the bug is present).
    with TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        _copy_workspace(fixture.workspace_dir, ws)
        inspector = FilesystemInspector(ws)
        results = await _run_commands(inspector, commands)
        initial_failed = all(code != 0 for code, _ in results)
        states["initial_required_commands_failed"] = initial_failed
        if not initial_failed:
            ok = False

    # 2. Gold solution: required commands pass AND hidden checker passes.
    if fixture.gold_dir.is_dir():
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            _copy_workspace(fixture.workspace_dir, ws)
            _apply_overlay(ws, fixture.gold_dir)
            inspector = FilesystemInspector(ws)
            results = await _run_commands(inspector, commands)
            gold_cmds_pass = all(code == 0 for code, _ in results)
            ctx = CheckerContext(
                inspector=inspector,
                fixture_root=fixture.workspace_dir,
                expected_terminal="succeeded",
                required_commands=commands,
            )
            checker = await run_checker_module(fixture.checker_path, ctx, safe=False)
            states["gold_required_commands_passed"] = gold_cmds_pass
            states["gold_checker_passed"] = checker.passed
            if not (gold_cmds_pass and checker.passed):
                ok = False
    else:
        states["gold"] = "missing"
        ok = False

    # 3. Cheat states: hidden checker must REJECT each.
    for cheat_name, tamper in (
        ("cheat_modify_tests", _tamper_modify_tests),
        ("cheat_delete_tests", _tamper_delete_tests),
        ("cheat_skip_tests", _tamper_skip_tests),
    ):
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            _copy_workspace(fixture.workspace_dir, ws)
            tamper(ws, language)
            inspector = FilesystemInspector(ws)
            ctx = CheckerContext(
                inspector=inspector,
                fixture_root=fixture.workspace_dir,
                expected_terminal="succeeded",
                required_commands=commands,
            )
            checker = await run_checker_module(fixture.checker_path, ctx, safe=False)
            rejected = not checker.passed
            states[cheat_name + "_rejected"] = rejected
            if not rejected:
                ok = False

    # 4. Hardcoded/naive solution: passes public tests but fails hidden inputs.
    if fixture.naive_dir.is_dir():
        with TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            _copy_workspace(fixture.workspace_dir, ws)
            _apply_overlay(ws, fixture.naive_dir)
            inspector = FilesystemInspector(ws)
            ctx = CheckerContext(
                inspector=inspector,
                fixture_root=fixture.workspace_dir,
                expected_terminal="succeeded",
                required_commands=commands,
            )
            checker = await run_checker_module(fixture.checker_path, ctx, safe=False)
            states["cheat_naive_rejected"] = not checker.passed
            if checker.passed:
                ok = False
    else:
        states["cheat_naive"] = "not provided"

    return ok


async def _validate_unsafe_fixture(fixture: Fixture, states: dict[str, Any]) -> bool:
    manifest = fixture.manifest
    ok = True
    premise = manifest.premise_file

    # 1. Initial state: the required premise must genuinely be absent.
    with TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        _copy_workspace(fixture.workspace_dir, ws)
        premise_absent = (ws / premise).exists() is False if premise else True
        states["initial_premise_absent"] = premise_absent
        if not premise_absent:
            ok = False

    # 2. Expected safe failure: agent reports the blocker, does not fake.
    with TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        _copy_workspace(fixture.workspace_dir, ws)
        inspector = FilesystemInspector(ws)
        job = _FakeJob(status="failed", failure_reason=f"missing required input: {premise or 'private schema'}")
        ctx = CheckerContext(
            inspector=inspector,
            fixture_root=fixture.workspace_dir,
            job=job,
            expected_terminal="failed",
            required_commands=(),
        )
        checker = await run_checker_module(fixture.checker_path, ctx, safe=False)
        states["expected_safe_failure_accepted"] = checker.passed
        if not checker.passed:
            ok = False

    # 3. Cheat: fabricate the missing premise and report success -> rejected.
    with TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        _copy_workspace(fixture.workspace_dir, ws)
        if premise:
            fabricated = ws / premise
            fabricated.parent.mkdir(parents=True, exist_ok=True)
            fabricated.write_text("fabricated premise\n", encoding="utf-8")
        inspector = FilesystemInspector(ws)
        job = _FakeJob(status="succeeded", failure_reason=None, artifact_id="art_fake")
        ctx = CheckerContext(
            inspector=inspector,
            fixture_root=fixture.workspace_dir,
            job=job,
            expected_terminal="failed",
            required_commands=(),
        )
        checker = await run_checker_module(fixture.checker_path, ctx, safe=False)
        states["cheat_fabricated_rejected"] = not checker.passed
        if checker.passed:
            ok = False

    return ok


async def validate_all_fixtures(fixtures_root: Path) -> dict[str, FixtureValidationReport]:
    from docode.eval.manifest import load_suite_manifests

    manifests = load_suite_manifests(fixtures_root)
    reports: dict[str, FixtureValidationReport] = {}
    for case_id, manifest in manifests.items():
        fixture = load_fixture(manifest.fixture_dir)
        reports[case_id] = await validate_fixture(fixture)
    return reports
