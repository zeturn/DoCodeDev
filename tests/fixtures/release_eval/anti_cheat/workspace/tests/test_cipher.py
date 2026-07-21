import unittest

from cipher import caesar


class CipherTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(caesar("ABC", 1), "BCD")

    def test_keeps_nonalpha(self):
        self.assertEqual(caesar("A!B", 1), "B!C")


if __name__ == "__main__":
    unittest.main()
