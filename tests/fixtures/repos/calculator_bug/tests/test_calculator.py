from calculator import add
from unittest import TestCase


class CalculatorTests(TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
