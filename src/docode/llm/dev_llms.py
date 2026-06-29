from __future__ import annotations

import re

from docode.dobox.tools import ToolDefinition

from .decision import AgentDecision


class ScriptedDecisionLLM:
    """Deterministic development LLM for smoke tests and local end-to-end runs."""

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.calls = 0

    async def decide(self, *, system: str, messages: list[dict[str, object]], tools: list[ToolDefinition], context: str) -> AgentDecision:
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(
                type="tool_call",
                tool_name="write_file",
                args={
                    "path": "DOCODE_RESULT.md",
                    "content": f"# DoCode Result\n\nInstruction: {self.instruction}\n\nStatus: implemented by scripted development agent.\n",
                },
            )
        return AgentDecision(type="final_candidate", summary="Created DOCODE_RESULT.md and verified the workspace.")


class GitHubTrendingCrawlerDecisionLLM:
    def __init__(self, instruction: str) -> None:
        from docode.llm.crawler_templates import github_trending_files

        self.calls = 0
        self.files = github_trending_files(objective_id=objective_id_from_instruction(instruction))
        self.paths = list(self.files)

    async def decide(self, *, system: str, messages: list[dict[str, object]], tools: list[ToolDefinition], context: str) -> AgentDecision:
        _ = system, messages, tools, context
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(type="tool_call", tool_name="fetch_url", args={"url": "https://github.com/trending"})
        file_index = self.calls - 2
        if 0 <= file_index < len(self.paths):
            path = self.paths[file_index]
            return AgentDecision(type="tool_call", tool_name="write_file", args={"path": path, "content": self.files[path]})
        if self.calls == len(self.paths) + 2:
            return AgentDecision(
                type="tool_call",
                tool_name="run_command",
                args={
                    "command": (
                        "python3 -m unittest discover -s tests && "
                        "python3 crawler.py --preflight && "
                        "python3 crawler.py --dry-run && "
                        "python3 crawler.py"
                    ),
                    "cwd": "/workspace",
                },
            )
        return AgentDecision(
            type="final_candidate",
            summary="Built a standard-library GitHub Trending crawler artifact with parser tests, preflight, dry-run, CSV output, and Araneae structured sink events.",
        )


def is_github_trending_araneae_instruction(instruction: str) -> bool:
    lowered = instruction.lower()
    return "github" in lowered and "trending" in lowered and "araneae-ready" in lowered and "crawler" in lowered


def objective_id_from_instruction(instruction: str) -> str:
    match = re.search(r"(?im)^Objective id:\s*([A-Za-z0-9_.:-]+)\s*$", instruction)
    return match.group(1) if match else "obj_github_trending"
