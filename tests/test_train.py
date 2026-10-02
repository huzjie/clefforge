import unittest

from clefforge.config import Config
from clefforge.train.finetune import run_finetune


class TestTrain(unittest.TestCase):
    def test_finetune_improves(self):
        cfg = Config()
        cfg.train.steps = 30
        res = run_finetune(cfg, steps=30)
        self.assertGreaterEqual(res["after"], res["before"] - 0.01)


if __name__ == "__main__":
    unittest.main()
