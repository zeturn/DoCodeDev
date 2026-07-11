from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

CommandKind = Literal["producer", "validator", "test", "build", "lint", "smoke"]


@dataclass(slots=True)
class VerificationCommand:
    command: str
    kind: CommandKind
    explicit: bool = True
    depends_on: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    validates: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CommandEvidence:
    command: str
    edit_epoch: int
    passed: bool


class VerificationScheduler:
    def __init__(self, commands: list[VerificationCommand]) -> None:
        self.commands = commands
        self.edit_epoch = 0
        self.evidence: dict[str, CommandEvidence] = {}

    @classmethod
    def from_explicit_commands(cls, commands: list[str]) -> "VerificationScheduler":
        nodes: list[VerificationCommand] = []
        producers: dict[str, str] = {}
        for command in commands:
            reads, writes = command_artifact_paths(command)
            kind: CommandKind = "producer" if writes else "validator" if reads else infer_command_kind(command)
            dependencies = [producers[path] for path in reads if path in producers]
            node = VerificationCommand(command=command, kind=kind, depends_on=list(dict.fromkeys(dependencies)), produces=writes, validates=reads)
            nodes.append(node)
            for path in writes:
                producers[path] = command
        return cls(nodes)

    def mark_edit(self) -> None:
        self.edit_epoch += 1

    def record(self, command: str, passed: bool) -> None:
        self.evidence[command] = CommandEvidence(command, self.edit_epoch, passed)

    def is_fresh_success(self, command: str) -> bool:
        evidence = self.evidence.get(command)
        return bool(evidence and evidence.edit_epoch == self.edit_epoch and evidence.passed)

    def next_command(self) -> str | None:
        for node in self.commands:
            if self.is_fresh_success(node.command):
                continue
            if any(not self.is_fresh_success(dependency) for dependency in node.depends_on):
                dependency = next(item for item in node.depends_on if not self.is_fresh_success(item))
                return dependency
            return node.command
        return None


def command_artifact_paths(command: str) -> tuple[list[str], list[str]]:
    paths = list(dict.fromkeys(re.findall(r"[\w./-]+\.(?:json|csv|tsv|xml|txt)", command, re.IGNORECASE)))
    lowered = command.lower()
    writes = [path for path in paths if re.search(rf"(?:--output|-o|write_text|open\s*\(|>)\s*[=\"']?{re.escape(path)}", command, re.IGNORECASE)]
    if not writes and paths and any(token in lowered for token in ("build", "generate", "produce", "collect", "crawl", "export")):
        writes = paths[-1:]
    return [path for path in paths if path not in writes], writes


def infer_command_kind(command: str) -> CommandKind:
    lowered = command.lower()
    if "lint" in lowered or "ruff" in lowered:
        return "lint"
    if "build" in lowered:
        return "build"
    if "test" in lowered or "pytest" in lowered or "unittest" in lowered:
        return "test"
    return "smoke"
