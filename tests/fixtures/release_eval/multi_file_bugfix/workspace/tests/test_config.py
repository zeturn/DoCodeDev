import unittest

from config import load


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = load()
        self.assertEqual(cfg["timeout"], 30)
        self.assertEqual(cfg["mode"], "safe")

    def test_override_timeout(self):
        cfg = load({"timeout": 99})
        self.assertEqual(cfg["timeout"], 99)

    def test_override_mode(self):
        cfg = load({"mode": "fast"})
        self.assertEqual(cfg["mode"], "fast")


if __name__ == "__main__":
    unittest.main()
