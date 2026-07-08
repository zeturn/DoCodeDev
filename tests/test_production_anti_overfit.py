from __future__ import annotations

from pathlib import Path
from unittest import TestCase


FORBIDDEN_PRODUCTION_STRINGS = (
    "default_crawler_artifact_file_content",
    "github-trending-crawler",
    "https_github_com_trending",
    "GitHub Trending Page",
    "owner1",
    "repo1",
    "owner/repo",
    "Box-row",
    "stars today",
    "lamp-001",
    "mug-002",
    "Desk Lamp",
    "Travel Mug",
    "product-card",
    "data-sku",
    "generic_crawler_cli",
    "EXPECTED_PRODUCTS",
    "fixtures/products.html",
    "out.json",
    "item-101",
    "item-202",
    "Trail Bottle",
    "Canvas Tote",
    "external_crawler_cli",
)


class ProductionAntiOverfitTests(TestCase):
    def test_production_runtime_has_no_eval_specific_solution_strings(self) -> None:
        root = Path(__file__).resolve().parents[1]
        production = root / "src" / "docode"
        offenders: list[str] = []
        for path in production.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".zip"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in FORBIDDEN_PRODUCTION_STRINGS:
                if needle in text:
                    offenders.append(f"{path.relative_to(root)}: {needle}")

        self.assertEqual(offenders, [])
