"""Pointer head: score a fixed option set instead of generating tokens.

This is the core Clef idea. A language model normally emits logits over its whole
vocabulary; a decision model only needs logits over the N candidate options. The
pointer head projects the fused hidden state and each option embedding into a
shared space and returns dot-product scores -> logits over options.
"""
import math

from ..core.nn import Linear
from ..core.tensor import Tensor


class PointerHead:
    def __init__(self, dim=256, seed=5):
        self.q = Linear(dim, dim, seed=seed)
        self.k = Linear(dim, dim, seed=seed + 1)

    def logits(self, state, option_vecs):
        qs = self.q(Tensor([state]))
        out = []
        for ov in option_vecs:
            ks = self.k(Tensor([ov]))
            score = sum(qs.data[0][j] * ks.data[0][j] for j in range(len(ov)))
            out.append(score)
        return out

    def decide(self, state, option_vecs):
        logits = self.logits(state, option_vecs)
        best = max(range(len(logits)), key=lambda i: logits[i])
        return best, logits
