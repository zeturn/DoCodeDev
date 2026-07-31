import unittest

from calculator import add


class CalculatorTests(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-2, 2), 0)


if __name__ == "__main__":
    unittest.main()
