import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from crawler import parse_products


EXPECTED_PRODUCTS = [
    {
        "sku": "lamp-001",
        "name": "Desk Lamp",
        "url": "/catalog/desk-lamp",
        "price": 24.99,
        "category": "Lighting",
        "in_stock": True,
    },
    {
        "sku": "mug-002",
        "name": "Travel Mug",
        "url": "/catalog/travel-mug",
        "price": 12.5,
        "category": "Kitchen",
        "in_stock": False,
    },
]


class CrawlerTests(TestCase):
    def test_parse_products_from_fixture(self):
        html = Path("fixtures/products.html").read_text(encoding="utf-8")

        self.assertEqual(parse_products(html), EXPECTED_PRODUCTS)

    def test_cli_writes_json_output(self):
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.json"
            completed = subprocess.run(
                [sys.executable, "crawler.py", "fixtures/products.html", "--output", str(output_path)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), EXPECTED_PRODUCTS)
