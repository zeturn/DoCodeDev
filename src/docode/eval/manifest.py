"""Fixture manifest schema (V1) and strict validation.

Every evaluation case is described by a machine-readable manifest
(``fixture.json``). The loader rejects:

* duplicate case ids (across the whole suite),
* workspace paths that escape the fixture directory (``..`` / absolute),
* a checker located inside the workspace (it must run only on the host),
* absolute paths anywhere in the manifest,
* empty ``required_commands``,
* an invalid ``expected_terminal``.

All errors are raised as :class:`FixtureManifestError` with a stable, readable
message so CI and the fixture validator can fail loudly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docode.eval.models import VALID_EXPECTED_TERMINALS, FixtureManifestError

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FixtureManifest:
    schema_version: int
    id: str
    title: str
    category: str
    difficulty: str
    language: str
    workspace: str
    instruction: str
    checker: str
    required_commands: tuple[str, ...]
    expected_terminal: str
    network_mode: str
    premise_file: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    _fixture_dir: Path = field(default=Path(), init=False, repr=False)

    @property
    def fixture_dir(self) -> Path:
        # Set by the loader; not part of the constructor contract.
        return self._fixture_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "difficulty": self.difficulty,
            "language": self.language,
            "workspace": self.workspace,
            "instruction": self.instruction,
            "checker": self.checker,
            "required_commands": list(self.required_commands),
            "expected_terminal": self.expected_terminal,
            "network_mode": self.network_mode,
            "tags": list(self.tags),
        }


def _assert_safe_rel(name: str, *, where: str) -> None:
    """Reject absolute paths and ``..`` traversal in any manifest path field."""
    if not name:
        raise FixtureManifestError(f"{where}: path must not be empty")
    if name.startswith("/") or (len(name) > 1 and name[1:2] == ":" and name[0].isalpha()):
        raise FixtureManifestError(f"{where}: absolute path is not allowed: {name!r}")
    if ".." in Path(name).parts:
        raise FixtureManifestError(f"{where}: path traversal ('..') is not allowed: {name!r}")
    if name.startswith("\\"):
        raise FixtureManifestError(f"{where}: absolute path is not allowed: {name!r}")


def _enforce_workspace_outside_checker(workspace: str, checker: str) -> None:
    """The hidden checker must never live inside the agent workspace."""
    ws = Path(workspace)
    ck = Path(checker)
    try:
        ck.relative_to(ws)
    except ValueError:
        # checker is not inside the workspace -> ok
        return
    # checker resolves inside the workspace -> must be rejected. The raise is
    # placed outside the try/except because FixtureManifestError subclasses
    # ValueError, so an inner raise would otherwise be swallowed.
    raise FixtureManifestError(
        f"checker {checker!r} must not be located inside the workspace {workspace!r}; "
        "the hidden checker runs only on the harness host."
    )


def load_fixture_manifest(fixture_dir: Path) -> FixtureManifest:
    """Load and strictly validate a single fixture manifest from ``fixture_dir``.

    ``fixture_dir`` must contain a ``fixture.json``. The returned manifest has
    its ``fixture_dir`` property bound to the directory so callers can resolve
    sibling files (``instruction.md``, ``checker.py``, the workspace, ``gold/``).
    """
    fixture_dir = Path(fixture_dir)
    manifest_path = fixture_dir / "fixture.json"
    if not manifest_path.is_file():
        raise FixtureManifestError(f"fixture.json not found in {fixture_dir}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureManifestError(f"{manifest_path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise FixtureManifestError(f"{manifest_path}: manifest must be a JSON object")

    errors: list[str] = []

    schema_version = raw.get("schema_version")
    if schema_version is None:
        errors.append("schema_version is required (add \"schema_version\": 1)")
    elif schema_version != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {schema_version!r}; "
            f"only {MANIFEST_SCHEMA_VERSION} is supported by this harness"
        )

    case_id = raw.get("id")
    if not case_id or not isinstance(case_id, str):
        errors.append("id is required and must be a non-empty string")

    title = raw.get("title")
    if not title or not isinstance(title, str):
        errors.append("title is required")

    for field_name in ("category", "difficulty", "language", "workspace", "instruction", "checker"):
        value = raw.get(field_name)
        if not value or not isinstance(value, str):
            errors.append(f"{field_name} is required")

    required_commands = raw.get("required_commands")
    if not isinstance(required_commands, list) or not required_commands:
        errors.append("required_commands must be a non-empty list")
    elif not all(isinstance(cmd, str) and cmd.strip() for cmd in required_commands):
        errors.append("required_commands must contain only non-empty strings")

    expected_terminal = raw.get("expected_terminal", "succeeded")
    if expected_terminal not in VALID_EXPECTED_TERMINALS:
        errors.append(
            f"expected_terminal must be one of {sorted(VALID_EXPECTED_TERMINALS)}; got {expected_terminal!r}"
        )

    network_mode = raw.get("network_mode", "no_internet")
    if not isinstance(network_mode, str):
        errors.append("network_mode must be a string")

    premise_file = raw.get("premise_file")
    if premise_file is not None:
        if not isinstance(premise_file, str):
            errors.append("premise_file must be a string")
        else:
            try:
                _assert_safe_rel(premise_file, where="premise_file")
            except FixtureManifestError as exc:
                errors.append(str(exc))

    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        errors.append("tags must be a list of strings")

    # Path-safety checks (only when the fields are strings).
    try:
        if isinstance(raw.get("workspace"), str):
            _assert_safe_rel(raw["workspace"], where="workspace")
        if isinstance(raw.get("instruction"), str):
            _assert_safe_rel(raw["instruction"], where="instruction")
        if isinstance(raw.get("checker"), str):
            _assert_safe_rel(raw["checker"], where="checker")
        if isinstance(raw.get("workspace"), str) and isinstance(raw.get("checker"), str):
            _enforce_workspace_outside_checker(raw["workspace"], raw["checker"])
    except FixtureManifestError as exc:
        errors.append(str(exc))

    if errors:
        joined = "; ".join(errors)
        raise FixtureManifestError(f"{manifest_path}: {joined}")

    manifest = FixtureManifest(
        schema_version=int(schema_version),  # type: ignore[arg-type]
        id=case_id,  # type: ignore[arg-type]
        title=title,  # type: ignore[arg-type]
        category=raw["category"],
        difficulty=raw["difficulty"],
        language=raw["language"],
        workspace=raw["workspace"],
        instruction=raw["instruction"],
        checker=raw["checker"],
        required_commands=tuple(required_commands),  # type: ignore[arg-type]
        expected_terminal=expected_terminal,
        network_mode=network_mode,
        premise_file=premise_file if isinstance(premise_file, str) else None,  # type: ignore[arg-type]
        tags=tuple(tags),
    )
    object.__setattr__(manifest, "_fixture_dir", fixture_dir)
    return manifest


def load_suite_manifests(fixtures_root: Path) -> dict[str, FixtureManifest]:
    """Load every fixture under ``fixtures_root`` and reject duplicate ids."""
    fixtures_root = Path(fixtures_root)
    if not fixtures_root.is_dir():
        raise FixtureManifestError(f"fixtures root not found: {fixtures_root}")

    manifests: dict[str, FixtureManifest] = {}
    seen_ids: dict[str, str] = {}
    errors: list[str] = []
    for candidate in sorted(fixtures_root.iterdir()):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "fixture.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = load_fixture_manifest(candidate)
        except FixtureManifestError as exc:
            errors.append(str(exc))
            continue
        if manifest.id in seen_ids:
            errors.append(
                f"duplicate fixture id {manifest.id!r}: {seen_ids[manifest.id]!r} and {str(candidate)!r}"
            )
            continue
        seen_ids[manifest.id] = str(candidate)
        manifests[manifest.id] = manifest

    if errors:
        joined = "; ".join(errors)
        raise FixtureManifestError(f"suite manifest load failed: {joined}")
    if not manifests:
        raise FixtureManifestError(f"no valid fixtures found under {fixtures_root}")
    return manifests
