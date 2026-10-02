"""Deterministic, trainable mock backend.

Models the Clef decision head with a single scalar `skill` plus per-option noise:

    logit_i = skill * truth_i + noise_i
    truth_i = 1.0 if option i is the world answer else 0.0

`skill` genuinely drives accuracy: when skill is low the noise dominates (near
random), when skill -> 1.0 the correct option always wins. `train_step` applies a
strictly positive delta so skill climbs monotonically toward 1.0, mirroring RL
finetuning against a business reward.
"""
import math

from ..core.ops import softmax
from ..utils.stable import stable_float
from ..world import world_answer
from .base import Backend
from .registry import register_backend


@register_backend("mock")
class MockBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        self.skill = 0.35
        self.noise = 0.55
        self.temperature = getattr(config.backend, "temperature", 0.0) or 1.0

    def score(self, query, options, image=None):
        correct = world_answer(query, options, image=image)
        logits = []
        for i in range(len(options)):
            truth = 1.0 if i == correct else 0.0
            noise = stable_float(f"noise:{query}:{image}:{i}", -self.noise, self.noise)
            logits.append(self.skill * 1.0 * truth + noise)
        return logits

    def decide(self, query, options, image=None):
        logits = self.score(query, options, image=image)
        probs = softmax(logits, temperature=self.temperature).data[0]
        best = max(range(len(probs)), key=lambda i: probs[i])
        return {"choice": best, "probs": probs, "logits": logits, "confidence": probs[best]}

    def train_step(self, delta=0.0):
        # strictly positive delta -> skill monotonically approaches 1.0
        self.skill = min(1.0, self.skill + max(0.0, delta))
        return self.skill
