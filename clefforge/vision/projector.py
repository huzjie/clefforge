"""Project visual tokens into the language-token space.

Clef fuses vision by projecting visual tokens with a learned linear map (or MLP),
then gating them into the LLM's hidden stream. The projector here maps hidden ->
llm_hidden and returns the gating weight via a sigmoid on a confidence logit.
"""
import math

from ..core.nn import Linear
from ..core.tensor import Tensor


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class VisualProjector:
    def __init__(self, in_dim=128, out_dim=256, seed=3):
        self.fc = Linear(in_dim, out_dim, seed=seed)

    def project(self, visual_tokens):
        return self.fc(visual_tokens)

    def gate(self, visual_tokens):
        # deterministic gating weight per token, aggregated
        x = self.fc(visual_tokens)
        flat = [v for row in x.data for v in row]
        mean = sum(flat) / max(len(flat), 1)
        return _sigmoid(mean)
