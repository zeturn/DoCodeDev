from __future__ import annotations

import re
from dataclasses import dataclass, field

from docode.dobox.tools import DoBoxTools


IMPORTANT_FILES = (
    "README.md",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "Cargo.toml",
)


@dataclass(frozen=True, slots=True)
class ProjectInspection:
    listing: str
    important_files: dict[str, str] = field(default_factory=dict)
    detected_commands: dict[str, str | None] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    def summary(self) -> str:
        files = ", ".join(self.important_files) if self.important_files else "none"
        commands = ", ".join(f"{name}={command or 'not detected'}" for name, command in self.detected_commands.items())
        plan = "\n".join(f"- {item}" for item in self.plan)
        criteria = "\n".join(f"- {item}" for item in self.acceptance_criteria)
        return (
            f"Important files: {files}\n"
            f"Detected commands: {commands}\n"
            f"Plan:\n{plan}\n"
            f"Acceptance criteria:\n{criteria}"
        )


class ProjectInspector:
    async def inspect(self, instruction: str, tools: DoBoxTools) -> ProjectInspection:
        listing_result = await tools.list_files(".")
        listing = listing_result.output
        important_files: dict[str, str] = {}
        for path in IMPORTANT_FILES:
            if _listing_contains(listing, path):
                try:
                    result = await tools.read_file(path)
                except Exception as exc:
                    important_files[path] = f"<read failed: {exc}>"
                    continue
                important_files[path] = _clip(result.output, 8000)

        detected_commands = {
            "test": await tools.detect_test_command(),
            "build": await tools.detect_build_command(),
            "lint": await tools.detect_lint_command(),
        }
        explicit_test_command = extract_explicit_test_command(instruction)
        if explicit_test_command:
            detected_commands["test"] = explicit_test_command
            tools.set_detected_command("test", explicit_test_command)
        return ProjectInspection(
            listing=listing,
            important_files=important_files,
            detected_commands=detected_commands,
            plan=build_initial_plan(instruction, important_files, detected_commands),
            acceptance_criteria=build_acceptance_criteria(instruction, detected_commands),
        )


def build_initial_plan(instruction: str, important_files: dict[str, str], detected_commands: dict[str, str | None]) -> list[str]:
    plan = [
        f"Inspect the repository context relevant to: {instruction}",
        "Make the smallest code changes that satisfy the requested behavior.",
    ]
    if important_files:
        plan.insert(1, "Use the detected project manifests and README to follow the existing stack and conventions.")
    checks = [command for command in detected_commands.values() if command]
    if checks:
        plan.append("Run detected verification commands: " + "; ".join(checks) + ".")
    else:
        plan.append("No standard verification command was detected; create or explain a task-appropriate verification path.")
    plan.append("Finish only after verification passes and a final summary can be exported.")
    return plan


def build_acceptance_criteria(instruction: str, detected_commands: dict[str, str | None]) -> list[str]:
    criteria = [
        f"The implementation directly satisfies the user instruction: {instruction}",
        "The git diff is non-empty or an explicit requested artifact was produced.",
        "The final summary describes changed files, verification results, and remaining caveats.",
    ]
    for name in ("test", "build", "lint"):
        command = detected_commands.get(name)
        if command:
            criteria.append(f"`{command}` exits successfully.")
    if not any(detected_commands.values()):
        criteria.append("A reasonable manual or command-based verification explanation is recorded.")
    return criteria


def extract_explicit_test_command(instruction: str) -> str | None:
    commands = (
        (r"\bpython(?:3)?\s+-m\s+unittest\b", "python -m unittest"),
        (r"\bpytest\b", "pytest"),
        (r"\bnpm\s+test\b", "npm test"),
        (r"\bpnpm\s+test\b", "pnpm test"),
        (r"\byarn\s+test\b", "yarn test"),
        (r"\bgo\s+test\s+\./\.\.\.", "go test ./..."),
        (r"\bcargo\s+test\b", "cargo test"),
    )
    for pattern, command in commands:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match:
            return command
    return None


def _listing_contains(listing: str, path: str) -> bool:
    for line in listing.splitlines():
        parts = line.strip().rstrip("/").split()
        if not parts:
            continue
        if parts[-1] == path:
            return True
    return False


def _clip(text: str, limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace") + "\n<truncated>"
