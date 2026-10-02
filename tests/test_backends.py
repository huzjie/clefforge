import unittest

from clefforge.backends import list_backends, get_backend
from clefforge.config import Config
from clefforge.world import world_answer


class TestBackends(unittest.TestCase):
    def test_list_backends(self):
        names = list_backends()
        for n in ("mock", "cpu", "openai", "vllm", "transformers"):
            self.assertIn(n, names)

    def test_mock_decide(self):
        b = get_backend("mock")(Config())
        r = b.decide("选择工具", ["a", "b", "c"])
        self.assertIn("choice", r)
        self.assertTrue(0 <= r["choice"] < 3)
        self.assertAlmostEqual(sum(r["probs"]), 1.0, places=5)

    def test_mock_trainable(self):
        b = get_backend("mock")(Config())
        s0 = b.skill
        for _ in range(20):
            b.train_step(delta=0.05)
        self.assertGreater(b.skill, s0)

    def test_world_deterministic(self):
        self.assertEqual(world_answer("q", ["a", "b"]), world_answer("q", ["a", "b"]))

    def test_world_image_shifts(self):
        a = world_answer("q", ["x", "y"], image="img-1")
        self.assertIn(a, (0, 1))


if __name__ == "__main__":
    unittest.main()
