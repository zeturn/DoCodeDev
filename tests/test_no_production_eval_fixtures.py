from __future__ import annotations

from pathlib import Path


PRODUCTION_ROOTS = [
    Path("src/docode/agent"),
]

FORBIDDEN_PRODUCTION_SNIPPETS = [
    "default_crawler_artifact_file_content",
    "github-trending-crawler",
    "https_github_com_trending",
    "GitHub Trending Page",
    "owner/repo",
    "stars today",
    "Box-row",
    "owner1",
    "repo1",
]


def test_production_agent_runtime_does_not_embed_github_trends_eval_fixture() -> None:
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for snippet in FORBIDDEN_PRODUCTION_SNIPPETS:
                if snippet in text:
                    offenders.append(f"{path}: contains {snippet!r}")

    assert not offenders, "GitHub Trends eval fixture leaked into production runtime:\n" + "\n".join(offenders)
