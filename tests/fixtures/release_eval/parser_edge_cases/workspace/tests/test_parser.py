import unittest

from parser import parse_pairs


class ParserTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_pairs("a=1\nb=2"), {"a": "1", "b": "2"})

    def test_comments(self):
        self.assertEqual(parse_pairs("#ignore\nx=9"), {"x": "9"})

    def test_whitespace(self):
        # BUG: parser does not strip spaces around key/value.
        self.assertEqual(parse_pairs("a = 1"), {"a": "1"})


if __name__ == "__main__":
    unittest.main()
