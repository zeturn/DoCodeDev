import unittest

from slugify import slugify


class SlugifyTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_lowercase(self):
        self.assertEqual(slugify("ABC"), "abc")


if __name__ == "__main__":
    unittest.main()
