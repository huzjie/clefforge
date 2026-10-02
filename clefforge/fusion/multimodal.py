"""Gated fusion: mix visual and textual representations by a learned gate.

Reusable recipe: instead of concatenating modalities (which blows up dims), compute
a scalar gate g in [0,1] and blend: h = g*text + (1-g)*visual. This keeps the
hidden size constant and lets the model learn how much to trust vision per sample.
"""
import math


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class GatedFusion:
    def __init__(self, dim=256, seed=4):
        self.dim = dim

    def fuse(self, text_vec, visual_vec, gate_logit=0.0):
        g = _sigmoid(gate_logit)
        return [g * t + (1 - g) * v for t, v in zip(text_vec, visual_vec)], g
