import unittest

from clefforge.bench import list_benchmarks
from clefforge.config import Config
from clefforge.engine import ClefEngine
from clefforge.bench.base import run_benchmark


class TestBench(unittest.TestCase):
    def test_benchmarks_registered(self):
        names = list_benchmarks()
        for n in ("decision_index", "vision_discrimination", "routing", "calibration"):
            self.assertIn(n, names)

    def test_benchmarks_run(self):
        engine = ClefEngine(Config())
        for n in list_benchmarks():
            r = run_benchmark(n, engine)
            self.assertIn("score", r)
            self.assertTrue(0.0 <= r["score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
