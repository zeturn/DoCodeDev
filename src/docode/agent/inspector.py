from __future__ import annotations

import re
from dataclasses import dataclass, field

from docode.dobox.tools import DoBoxTools
from docode.agent.task_contract import TaskContract, is_crawler_instruction, task_contract_from_instruction, text_outside_verification_blocks


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
    explicit_commands: list[str] = field(default_factory=list)
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
    async def inspect(
        self,
        instruction: str,
        tools: DoBoxTools,
        task_contract: TaskContract | None = None,
    ) -> ProjectInspection:
        task_contract = task_contract or task_contract_from_instruction(instruction)
        listing_result = await tools.list_files(".")
        listing = listing_result.output
        important_files: dict[str, str] = {}
        if not should_skip_important_file_reads(instruction):
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
        explicit_commands = list(task_contract.must_run_commands)
        return ProjectInspection(
            listing=listing,
            important_files=important_files,
            detected_commands=detected_commands,
            explicit_commands=explicit_commands,
            plan=build_initial_plan(instruction, important_files, detected_commands, explicit_commands),
            acceptance_criteria=build_acceptance_criteria(instruction, detected_commands, explicit_commands),
        )


def build_initial_plan(
    instruction: str,
    important_files: dict[str, str],
    detected_commands: dict[str, str | None],
    explicit_commands: list[str] | None = None,
) -> list[str]:
    task_summary = summarize_instruction(instruction)
    plan = [
        f"Inspect the repository context relevant to: {task_summary}",
        "Make the smallest code changes that satisfy the requested behavior.",
    ]
    if important_files:
        plan.insert(1, "Use the detected project manifests and README to follow the existing stack and conventions.")
    explicit = list(explicit_commands or [])
    checks = [command for command in detected_commands.values() if command and command not in explicit]
    if explicit:
        plan.append("Run the exact required verification commands: " + "; ".join(_display_command(command) for command in explicit) + ".")
    if checks:
        plan.append("Run detected verification commands: " + "; ".join(checks) + ".")
    elif not explicit:
        plan.append("No standard verification command was detected; create or explain a task-appropriate verification path.")
    plan.append("Finish only after verification passes and a final summary can be exported.")
    return plan


def build_acceptance_criteria(
    instruction: str,
    detected_commands: dict[str, str | None],
    explicit_commands: list[str] | None = None,
) -> list[str]:
    task_summary = summarize_instruction(instruction)
    criteria = [
        f"The implementation directly satisfies the requested task: {task_summary}",
        "The git diff is non-empty or an explicit requested artifact was produced.",
        "The final summary describes changed files, verification results, and remaining caveats.",
    ]
    explicit = list(explicit_commands or [])
    for command in explicit:
        criteria.append(f"The exact required command `{_display_command(command)}` exits successfully after the latest edit.")
    for name in ("test", "build", "lint"):
        command = detected_commands.get(name)
        if command and command not in explicit:
            criteria.append(f"`{command}` exits successfully.")
    if not explicit and not any(detected_commands.values()):
        criteria.append("A reasonable manual or command-based verification explanation is recorded.")
    return criteria


def should_skip_important_file_reads(instruction: str) -> bool:
    return is_crawler_instruction(instruction) and has_public_source_url(instruction)


def has_public_source_url(instruction: str) -> bool:
    return bool(re.search(r"https?://[^\s'\"`)>]+", instruction or "", flags=re.IGNORECASE))


def summarize_instruction(instruction: str, limit: int = 240) -> str:
    compact = " ".join(part.strip() for part in text_outside_verification_blocks(instruction).splitlines() if part.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 13].rstrip() + " <truncated>"


def _display_command(command: str) -> str:
    lines = command.replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")
    if len(lines) <= 1:
        return lines[0].strip() if lines else "<empty command>"
    return f"{lines[0].strip() or '<empty first line>'} [multiline verification command, {len(lines)} lines]"


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
