"""ViT-style vision encoder over synthesized patch features.

Layers of linear projection + GELU + layer norm over patch tokens; the final
[CLS]-style pooled token summarizes the whole image. Returns a tensor of shape
(n_patches+1, hidden).
"""
from ..core.nn import Linear
from ..core.ops import gelu, layernorm
from ..core.tensor import Tensor
from .patcher import Patcher
from .resolution import resolve_resolution


class VisionEncoder:
    def __init__(self, vision_cfg):
        self.cfg = vision_cfg
        self.hidden = getattr(vision_cfg, "hidden", 128)
        self.layers = getattr(vision_cfg, "layers", 2)
        self.patcher = Patcher(getattr(vision_cfg, "patch_size", 16), self.hidden)
        self.proj = Linear(self.hidden, self.hidden, seed=1)
        self.head = Linear(self.hidden, self.hidden, seed=2)

    def encode(self, image_id, width=224, height=224, n_patches=None):
        if n_patches is None:
            r = resolve_resolution(width, height, getattr(self.cfg, "max_patches", 512),
                                   getattr(self.cfg, "patch_size", 16))
            n_patches = r["n_patches"]
        feats = self.patcher.patch_features(image_id, n_patches)
        x = Tensor(feats)
        for _ in range(self.layers):
            x = gelu(layernorm(self.proj(x)))
        # CLS token = mean of patch tokens
        pooled = [[sum(x.data[i][j] for i in range(x.shape[0])) / x.shape[0]
                   for j in range(x.shape[1])]]
        return Tensor(feats + pooled)
