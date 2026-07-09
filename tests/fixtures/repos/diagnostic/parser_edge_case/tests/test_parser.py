from __future__ import annotations

from unittest import TestCase

from parser import load_items


class ParserTests(TestCase):
    def test_normalizes_all_fixture_items(self) -> None:
        self.assertEqual(
            load_items("fixtures/items.json"),
            [
                {"name": "Notebook", "price": 12.5, "available": True, "category": "stationery"},
                {"name": "Pencil", "price": 1.25, "available": True, "category": "uncategorized"},
                {"name": "Desk", "price": 120.0, "available": False, "category": "furniture"},
            ],
        )
