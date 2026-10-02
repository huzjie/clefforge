import unittest

from clefforge.config import Config
from clefforge.vision.encoder import VisionEncoder
from clefforge.vision.projector import VisualProjector
from clefforge.vision.resolution import resolve_resolution


class TestVision(unittest.TestCase):
    def test_encoder(self):
        enc = VisionEncoder(Config().vision)
        feats = enc.encode("img-1", n_patches=4)
        self.assertEqual(feats.shape[0], 5)  # 4 patches + CLS
        self.assertEqual(feats.shape[1], 128)

    def test_resolution_single(self):
        r = resolve_resolution(224, 224, max_patches=512, patch_size=16)
        self.assertEqual(r["mode"], "single")

    def test_resolution_tiled(self):
        r = resolve_resolution(4096, 4096, max_patches=512, patch_size=16)
        self.assertEqual(r["mode"], "tiled")

    def test_projector(self):
        p = VisualProjector(in_dim=128, out_dim=128)
        enc = VisionEncoder(Config().vision)
        toks = enc.encode("img-2", n_patches=4)
        out = p.project(toks)
        self.assertEqual(out.shape[1], 128)


if __name__ == "__main__":
    unittest.main()
