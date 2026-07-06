from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from docode.agent.task_contract import TaskContract
from docode.agent.verifier import changed_paths_from_status, meaningful_change_path
from docode.dobox.tools import DoBoxTools
from docode.dobox.types import ToolResult


Severity = Literal["blocker", "warning"]


@dataclass(frozen=True, slots=True)
class QualityIssue:
    severity: Severity
    code: str
    message: str
    path: str | None = None
    repair_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "repair_hint": self.repair_hint,
        }


@dataclass(frozen=True, slots=True)
class ArtifactSample:
    path: str
    kind: str
    summary: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "summary": self.summary,
            "data": self.data,
        }


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)
    samples: list[ArtifactSample] = field(default_factory=list)
    git_status: str = ""
    git_diff: str = ""

    def blockers(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "blocker"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "quality_gate",
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "samples": [sample.to_dict() for sample in self.samples],
            "git_status": self.git_status,
            "git_diff": self.git_diff[:20000],
        }


class QualityGate:
    async def run(
        self,
        *,
        tools: DoBoxTools,
        task_contract: TaskContract | None,
        instruction: str,
    ) -> QualityGateResult:
        await safe_optional_tool_call("run_command", tools, "git add -N . >/dev/null 2>&1 || true", "/workspace")
        status_result = await safe_tool_call("git_status", tools.git_status)
        diff_result = await safe_tool_call("git_diff", tools.git_diff)

        status = status_result.output or ""
        diff = diff_result.output or ""
        issues: list[QualityIssue] = []
        samples: list[ArtifactSample] = []

        issues.extend(detect_empty_or_generated_only_diff(status, diff))
        issues.extend(detect_duplicate_python_implementations(diff))
        issues.extend(detect_placeholder_code(diff))
        issues.extend(detect_undeclared_python_dependencies(diff))

        artifact_issues, artifact_samples = await inspect_common_artifacts(
            tools=tools,
            instruction=instruction,
            task_contract=task_contract,
            status=status,
        )
        issues.extend(artifact_issues)
        samples.extend(artifact_samples)
        issues.extend(await inspect_markdown_artifacts(tools, task_contract))

        blockers = [issue for issue in issues if issue.severity == "blocker"]
        return QualityGateResult(
            passed=not blockers,
            issues=issues,
            samples=samples,
            git_status=status,
            git_diff=diff,
        )


async def safe_tool_call(tool_name: str, call) -> ToolResult:
    try:
        return await call()
    except Exception as exc:
        return ToolResult(
            tool=tool_name,
            output=f"{tool_name} failed: {exc}",
            exit_code=1,
            metadata={"exception_type": type(exc).__name__, "error": str(exc)},
        )


async def safe_optional_tool_call(tool_name: str, tools: DoBoxTools, *args) -> ToolResult:
    call = getattr(tools, tool_name, None)
    if call is None:
        return ToolResult(tool=tool_name, output=f"{tool_name} unavailable", exit_code=1)
    return await safe_tool_call(tool_name, lambda: call(*args))


def detect_empty_or_generated_only_diff(status: str, diff: str) -> list[QualityIssue]:
    changed = changed_paths_from_status(status)
    meaningful = [path for path in changed if meaningful_quality_path(path)]
    if not changed and not diff.strip():
        return [
            QualityIssue(
                severity="blocker",
                code="no_changes",
                message="No changed files detected.",
                repair_hint="Modify the target source, test, document, or artifact files before final verification.",
            )
        ]
    if changed and not meaningful:
        return [
            QualityIssue(
                severity="blocker",
                code="generated_only_changes",
                message="Only generated/cache/probe files changed.",
                repair_hint="Make a meaningful change to a target source, test, document, or artifact file.",
            )
        ]
    return []


def meaningful_quality_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    parts = normalized.split("/")
    if not meaningful_change_path(normalized):
        return False
    return not (
        ".pytest_cache" in parts
        or ".mypy_cache" in parts
        or "node_modules" in parts
        or normalized.endswith((".log", ".tmp"))
    )


def detect_duplicate_python_implementations(diff: str) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for path, text in split_diff_by_file(diff).items():
        if not path.endswith(".py"):
            continue
        main_count = text.count('if __name__ == "__main__"') + text.count("if __name__ == '__main__'")
        if main_count > 1:
            issues.append(
                QualityIssue(
                    severity="blocker",
                    code="duplicate_python_entrypoint",
                    path=path,
                    message=f"{path} appears to contain multiple Python entrypoints.",
                    repair_hint="Rewrite the file once cleanly instead of appending another implementation.",
                )
            )
        function_names = re.findall(r"^\+\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.MULTILINE)
        repeated = sorted({name for name in function_names if function_names.count(name) > 1})
        if repeated:
            issues.append(
                QualityIssue(
                    severity="blocker",
                    code="duplicate_python_functions",
                    path=path,
                    message=f"{path} defines duplicate functions: {', '.join(repeated)}.",
                    repair_hint="Remove duplicate implementations and keep one coherent version.",
                )
            )
    return issues


def detect_placeholder_code(diff: str) -> list[QualityIssue]:
    code_suffixes = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cs")
    for path, file_diff in split_diff_by_file(diff).items():
        if not path.endswith(code_suffixes):
            continue
        lowered = file_diff.lower()
        for marker in ("todo", "placeholder", "not implemented", "stub", "pass  #", "raise notimplementederror"):
            if marker in lowered:
                return [
                    QualityIssue(
                        severity="blocker",
                        code="placeholder_left_in_diff",
                        path=path,
                        message=f"Diff still contains placeholder marker: {marker}",
                        repair_hint="Replace placeholder/stub code with a real implementation before verification.",
                    )
                ]
    return []


THIRD_PARTY_IMPORTS = {
    "bs4": "beautifulsoup4",
    "requests": "requests",
    "httpx": "httpx",
    "lxml": "lxml",
    "pandas": "pandas",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "scrapy": "scrapy",
    "selenium": "selenium",
    "playwright": "playwright",
}


GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def detect_undeclared_python_dependencies(diff: str) -> list[QualityIssue]:
    imports: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pattern in (r"\+\s*import\s+([A-Za-z_][A-Za-z0-9_]*)", r"\+\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+"):
            match = re.match(pattern, line)
            if match:
                imports.add(match.group(1))
    used = sorted(THIRD_PARTY_IMPORTS[name] for name in imports if name in THIRD_PARTY_IMPORTS)
    if not used:
        return []
    declared = has_dependency_manifest(diff)
    return [
        QualityIssue(
            severity="warning" if declared else "blocker",
            code="third_party_dependency_declared" if declared else "undeclared_third_party_dependency",
            message=(
                f"Third-party imports detected: {', '.join(used)}."
                if declared
                else f"Third-party imports detected without dependency declaration: {', '.join(used)}."
            ),
            repair_hint=(
                "Run import checks for the declared dependencies."
                if declared
                else "Either remove the dependency and use the standard library, or add dependency metadata and verify imports."
            ),
        )
    ]


def has_dependency_manifest(diff: str) -> bool:
    lowered = diff.lower()
    return any(name in lowered for name in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "pipfile"))


async def inspect_common_artifacts(
    *,
    tools: DoBoxTools,
    instruction: str,
    task_contract: TaskContract | None,
    status: str,
) -> tuple[list[QualityIssue], list[ArtifactSample]]:
    issues: list[QualityIssue] = []
    samples: list[ArtifactSample] = []
    json_paths = inferred_json_artifact_paths(instruction, task_contract, status)
    for path in json_paths[:5]:
        artifact_issues, sample = await inspect_json_artifact(tools, path, instruction)
        issues.extend(artifact_issues)
        if sample is not None:
            samples.append(sample)
    return issues, samples


def inferred_json_artifact_paths(instruction: str, task_contract: TaskContract | None, status: str) -> list[str]:
    candidates: list[str] = []
    config_json = {"manifest.json", "sources.json", "schemas/output.schema.json"}
    for path in task_contract.must_modify_files if task_contract is not None else []:
        normalized = path.strip("./").replace("\\", "/")
        if normalized.endswith(".json") and normalized not in config_json and not normalized.startswith(("schemas/", "fixtures/")):
            candidates.append(path)
    for path in changed_paths_from_status(status):
        normalized = path.strip().replace("\\", "/")
        if normalized.endswith(".json") and normalized.strip("./") not in config_json and not normalized.startswith(("schemas/", "fixtures/")):
            candidates.append(normalized)
    lowered = instruction.lower()
    for path in re.findall(r"\b[\w./-]+\.json\b", instruction):
        normalized = path.strip("./").replace("\\", "/")
        if normalized not in config_json and not normalized.startswith(("schemas/", "fixtures/")):
            candidates.append(normalized)
    if ("json" in lowered or "crawler" in lowered or "scraper" in lowered or "爬虫" in lowered) and not candidates:
        candidates.extend(["data/output.json", "output.json"])
    return unique_paths(candidates)


async def inspect_json_artifact(tools: DoBoxTools, path: str, instruction: str) -> tuple[list[QualityIssue], ArtifactSample | None]:
    result = await safe_optional_tool_call("read_file", tools, path)
    if result.exit_code != 0:
        return [], None
    try:
        data = json.loads(result.output)
    except Exception as exc:
        return [
            QualityIssue(
                severity="blocker",
                code="json_artifact_invalid",
                path=path,
                message=f"JSON artifact is invalid: {exc}",
                repair_hint="Write valid JSON before final verification.",
            )
        ], None

    issues = inspect_json_data(data, path, instruction)
    sample_data = data[:3] if isinstance(data, list) else data
    summary = json_sample_summary(data)
    return issues, ArtifactSample(path=path, kind="json", summary=summary, data=sample_data)


def inspect_json_data(data: Any, path: str, instruction: str) -> list[QualityIssue]:
    if isinstance(data, list):
        return inspect_json_records(data, path, instruction)
    if isinstance(data, dict):
        return []
    return [
        QualityIssue(
            severity="blocker",
            code="json_artifact_unexpected_shape",
            path=path,
            message="JSON artifact must be an object or a list of objects.",
            repair_hint="Write stable structured JSON rather than a scalar value.",
        )
    ]


def inspect_json_records(records: list[Any], path: str, instruction: str) -> list[QualityIssue]:
    if not records:
        return [
            QualityIssue(
                severity="blocker",
                code="json_records_empty",
                path=path,
                message="JSON artifact is an empty list.",
                repair_hint="Write at least one meaningful record.",
            )
        ]
    issues: list[QualityIssue] = []
    required_fields = inferred_required_json_fields(instruction)
    for index, row in enumerate(records[:10]):
        if not isinstance(row, dict):
            issues.append(
                QualityIssue(
                    severity="blocker",
                    code="json_record_not_object",
                    path=path,
                    message=f"JSON row {index} is not an object.",
                    repair_hint="Output a list of JSON objects with stable fields.",
                )
            )
            continue
        for field in required_fields:
            value = github_repository_value(row) if field == "__github_repository__" else row.get(field)
            if value is None or value == "":
                label = "repository_name/repository/name" if field == "__github_repository__" else field
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        code="json_required_field_empty",
                        path=path,
                        message=f"JSON row {index} has empty required field: {label}",
                        repair_hint=f"Populate {label} with a meaningful non-empty value for every record.",
                    )
                )
            elif isinstance(value, str) and dirty_required_value(value):
                label = "repository_name/repository/name" if field == "__github_repository__" else field
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        code="json_required_field_dirty",
                        path=path,
                        message=f"JSON row {index} has dirty required field {label}: {preview(value)}",
                        repair_hint=f"Normalize {label}; derive stable identifiers from URLs or structured attributes instead of raw text.",
                    )
                )
        url = row.get("url")
        repository = github_repository_value(row)
        if "github" in instruction.lower():
            if isinstance(url, str) and not url.startswith("https://github.com/"):
                issues.append(
                    QualityIssue(
                        severity="blocker",
                        code="json_github_url_invalid",
                        path=path,
                        message=f"JSON row {index} has invalid GitHub URL: {url}",
                        repair_hint="Build absolute repository URLs as https://github.com/owner/repo.",
                    )
                )
            if isinstance(repository, str) and repository:
                if not GITHUB_REPO_RE.match(repository):
                    issues.append(
                        QualityIssue(
                            severity="blocker",
                            code="json_repository_invalid_format",
                            path=path,
                            message=f"JSON row {index} repository must look like owner/repo, got {preview(repository)}",
                            repair_hint="Derive repository from the GitHub href path /owner/repo.",
                        )
                    )
                elif isinstance(url, str) and url.startswith("https://github.com/"):
                    expected_url = f"https://github.com/{repository}"
                    if url.rstrip("/") != expected_url:
                        issues.append(
                            QualityIssue(
                                severity="blocker",
                                code="json_repository_url_mismatch",
                                path=path,
                                message=f"JSON row {index} repository {repository!r} does not match url {url!r}",
                                repair_hint="Ensure repository and url are derived from the same owner/repo href.",
                            )
                        )
    return dedupe_issues(issues)


def inferred_required_json_fields(instruction: str) -> list[str]:
    lowered = instruction.lower()
    if "github" in lowered and ("trending" in lowered or "repository" in lowered or "repo" in lowered):
        return ["__github_repository__", "url"]
    if "crawler" in lowered or "scraper" in lowered or "爬虫" in lowered:
        return ["url"]
    return []


def github_repository_value(row: dict[str, Any]) -> Any:
    for key in ("repository_name", "repository", "name"):
        value = row.get(key)
        if value not in {None, ""}:
            return value
    owner = row.get("owner")
    name = row.get("repo") or row.get("project")
    if owner and name:
        return f"{owner}/{name}"
    return None


def dirty_required_value(value: str) -> bool:
    stripped = value.strip()
    return value != stripped or "\n" in value or "\r" in value or re.search(r"\s{3,}", value) is not None


async def inspect_markdown_artifacts(tools: DoBoxTools, task_contract: TaskContract | None) -> list[QualityIssue]:
    if task_contract is None:
        return []
    issues: list[QualityIssue] = []
    for path in task_contract.must_modify_files:
        if not path.lower().endswith(".md"):
            continue
        result = await safe_optional_tool_call("read_file", tools, path)
        if result.exit_code == 0:
            issues.extend(detect_empty_markdown_sections(path, result.output))
    return issues


def detect_empty_markdown_sections(path: str, text: str) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    sections = re.split(r"(?m)^#{1,6}\s+", text)
    for section in sections:
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0].lower()
        body = lines[1:]
        if title in {"installation", "usage", "configuration", "examples"} and len(" ".join(body)) < 20:
            issues.append(
                QualityIssue(
                    severity="blocker",
                    code="markdown_section_empty",
                    path=path,
                    message=f"Markdown section '{lines[0]}' is empty or too thin.",
                    repair_hint=f"Add concrete content, commands, or examples to the '{lines[0]}' section.",
                )
            )
    return issues


def split_diff_by_file(diff: str) -> dict[str, str]:
    files: dict[str, list[str]] = {}
    current = ""
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            match = re.search(r" b/(.+)$", line)
            current = match.group(1).strip() if match else ""
            if current:
                files.setdefault(current, [])
            continue
        if current:
            files[current].append(line)
    return {path: "\n".join(lines) for path, lines in files.items()}


def json_sample_summary(data: Any) -> str:
    if isinstance(data, list):
        return f"list records={len(data)} sample={preview(json.dumps(data[:1], ensure_ascii=False))}"
    if isinstance(data, dict):
        return f"object keys={sorted(data)[:10]}"
    return type(data).__name__


def unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        normalized = path.strip().replace("\\", "/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def dedupe_issues(issues: list[QualityIssue]) -> list[QualityIssue]:
    seen: set[tuple[str, str | None, str]] = set()
    result: list[QualityIssue] = []
    for issue in issues:
        key = (issue.code, issue.path, issue.message)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def preview(value: str, limit: int = 120) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
