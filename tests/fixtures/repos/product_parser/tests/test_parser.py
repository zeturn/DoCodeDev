from pathlib import Path
from unittest import TestCase

from parser import parse_products


class ProductParserTests(TestCase):
    def test_parse_products_from_fixture(self) -> None:
        html = Path("fixtures/products.html").read_text(encoding="utf-8")

        self.assertEqual(
            parse_products(html),
            [
                {
                    "id": "sku-001",
                    "name": "Desk Lamp",
                    "url": "/products/desk-lamp",
                    "price": 24.99,
                    "rating": 4.7,
                    "in_stock": True,
                },
                {
                    "id": "sku-002",
                    "name": "Notebook",
                    "url": "/products/notebook",
                    "price": 5.50,
                    "rating": 4.2,
                    "in_stock": False,
                },
            ],
        )
