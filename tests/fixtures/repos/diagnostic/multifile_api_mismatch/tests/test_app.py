from __future__ import annotations

from unittest import TestCase

from app import build_profile


class AppTests(TestCase):
    def test_build_profile(self) -> None:
        self.assertEqual(
            build_profile({"first": "Ada", "last": "Lovelace"}),
            {"display_name": "Ada Lovelace", "slug": "ada-lovelace"},
        )
