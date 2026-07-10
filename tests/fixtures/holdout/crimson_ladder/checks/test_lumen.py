import unittest

from lumen_quota import normalize_units, reserve_slots


class LumenQuotaTests(unittest.TestCase):
    def test_normalize_rejects_negative_units(self):
        with self.assertRaises(ValueError):
            normalize_units(-1)

    def test_reservation_is_capped_at_ceiling(self):
        self.assertEqual(reserve_slots(4, 9), 4)
        self.assertEqual(reserve_slots(14, 9), 9)


if __name__ == "__main__":
    unittest.main()
