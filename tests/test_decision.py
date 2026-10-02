import unittest

from clefforge.decision.head import PointerHead
from clefforge.decision.scorer import score_options
from clefforge.decision.calibration import expected_calibration_error, temperature_scale
from clefforge.decision.permutation import permutation_score
from clefforge.decision.gate import Gate
from clefforge.decision.router import Router


class TestDecision(unittest.TestCase):
    def test_head_returns_best(self):
        h = PointerHead(dim=64)
        opt_vecs = [[1.0] * 64, [2.0] * 64, [3.0] * 64]
        best, logits = h.decide([0.0] * 64, opt_vecs)
        self.assertIn(best, (0, 1, 2))
        self.assertEqual(len(logits), 3)

    def test_score_options(self):
        r = score_options([0.1, 0.9, 0.3])
        self.assertEqual(r["choice"], 1)
        self.assertAlmostEqual(sum(r["probs"]), 1.0, places=5)

    def test_ece_range(self):
        e = expected_calibration_error([0.8, 0.2, 0.9], [1, 0, 1])
        self.assertGreaterEqual(e, 0.0)

    def test_temperature_scale(self):
        p = temperature_scale([0.5, 0.5], 2.0)
        self.assertAlmostEqual(sum(p), 1.0, places=5)

    def test_gate(self):
        g = Gate(threshold=0.5, degrade_threshold=0.7)
        self.assertEqual(g.apply(0.9).action, "accept")
        self.assertEqual(g.apply(0.6).action, "degrade")
        self.assertEqual(g.apply(0.2).action, "reject")

    def test_router(self):
        r = Router(fast_slow=True)
        self.assertEqual(r.route("短问题", ["a", "b"]).model, "clef-flash")


if __name__ == "__main__":
    unittest.main()
