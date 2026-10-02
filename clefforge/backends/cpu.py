"""CPU reference backend: real pointer-head math on the tensor core.

Unlike the mock (which uses a scalar skill), this backend actually projects the
query/options through the pointer head and vision encoder on the tensor core, so
it exercises the full model graph without a GPU.
"""
from ..decision.head import PointerHead
from ..decision.scorer import score_options
from ..vision.encoder import VisionEncoder
from ..vision.projector import VisualProjector
from .base import Backend
from .registry import register_backend


@register_backend("cpu")
class CpuBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        self.head = PointerHead(dim=getattr(config.vision, "hidden", 128))
        self.encoder = VisionEncoder(config.vision)
        self.projector = VisualProjector(in_dim=getattr(config.vision, "hidden", 128),
                                         out_dim=getattr(config.vision, "hidden", 128))

    def _option_vec(self, option, dim):
        from ..utils.stable import stable_vector
        return stable_vector(f"opt:{option}", dim)

    def score(self, query, options, image=None):
        dim = getattr(self.config.vision, "hidden", 128)
        opt_vecs = [self._option_vec(o, dim) for o in options]
        state = [0.0] * dim
        if image:
            toks = self.encoder.encode(image, n_patches=4)
            state = self.projector.project(toks).data[0]
        return self.head.logits(state, opt_vecs)

    def decide(self, query, options, image=None):
        logits = self.score(query, options, image=image)
        r = score_options(logits, temperature=1.0)
        r["logits"] = logits
        return r
