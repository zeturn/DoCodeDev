import unittest

from zephyr_lattice import resolve_band


class ResolveBandTests(unittest.TestCase):
    def test_below_band_uses_low(self):
        self.assertEqual(resolve_band(-4, 2, 8), 2)

    def test_inside_band_is_unchanged(self):
        self.assertEqual(resolve_band(5, 2, 8), 5)

    def test_above_band_uses_high(self):
        self.assertEqual(resolve_band(19, 2, 8), 8)


if __name__ == "__main__":
    unittest.main()
