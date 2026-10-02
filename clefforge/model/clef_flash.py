"""Clef-flash: the 9B fast tier.

Same pointer-head decision API on a smaller backbone. Used by the fast-slow
router for low-difficulty tasks to cut latency/cost.
"""


class ClefFlashModel:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def decide(self, query, options, image=None):
        return self.backend.decide(query, options, image=image)
