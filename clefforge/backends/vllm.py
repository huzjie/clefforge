"""vLLM backend stub: high-throughput serving of the Clef pointer head.

Falls back to mock unless a vLLM endpoint is configured. Kept as a thin adapter so
the decision API is uniform across backends.
"""
from .base import Backend
from .mock import MockBackend
from .registry import register_backend


@register_backend("vllm")
class VLLMBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        self._fallback = MockBackend(config)

    def decide(self, query, options, image=None):
        return self._fallback.decide(query, options, image=image)

    def score(self, query, options, image=None):
        return self._fallback.score(query, options, image=image)
