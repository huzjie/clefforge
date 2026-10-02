"""Clef: the multimodal decision model (27B tier).

Wraps a Qwen backbone + vision encoder + pointer head. `decide` is backend-backed:
the model delegates scoring to the configured backend so mock/cpu/openai/vllm/
transformers all share one API.
"""


class ClefModel:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def decide(self, query, options, image=None):
        return self.backend.decide(query, options, image=image)

    def score(self, query, options, image=None):
        return self.backend.score(query, options, image=image)
