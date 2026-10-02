import unittest

from clefforge.utils.stable import stable_float, stable_ints, stable_vector


class TestStable(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(stable_float("k"), stable_float("k"))

    def test_high_entropy(self):
        vals = {stable_float(f"k{i}") for i in range(50)}
        self.assertGreater(len(vals), 45)

    def test_ints_range(self):
        for v in stable_ints("x", 100, 0, 9):
            self.assertTrue(0 <= v <= 9)

    def test_vector_len(self):
        self.assertEqual(len(stable_vector("v", 64)), 64)


if __name__ == "__main__":
    unittest.main()
