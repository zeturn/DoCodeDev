import unittest

from calculator import add, subtract


class CalculatorTests(unittest.TestCase):
    def test_add_basic(self):
        self.assertEqual(add(2, 3), 5)

    def test_add_zero(self):
        self.assertEqual(add(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
