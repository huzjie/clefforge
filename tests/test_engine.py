import unittest

from clefforge.config import Config
from clefforge.engine import ClefEngine


class TestEngine(unittest.TestCase):
    def test_decide_text(self):
        e = ClefEngine(Config())
        r = e.decide("选择工具", ["a", "b", "c"])
        self.assertIn(r["answer"], ["a", "b", "c"])
        self.assertIn(r["gate"], ("accept", "degrade", "reject"))
        self.assertIn(r["route"], ("clef", "clef-flash"))

    def test_decide_image(self):
        e = ClefEngine(Config())
        r = e.decide("识别物体", ["cat", "dog"], image="img-1")
        self.assertIn(r["answer"], ["cat", "dog"])


if __name__ == "__main__":
    unittest.main()
