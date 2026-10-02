"""Backend interface."""


class Backend:
    name = "base"

    def __init__(self, config):
        self.config = config

    def score(self, query, options, image=None):
        raise NotImplementedError

    def decide(self, query, options, image=None):
        raise NotImplementedError

    def train_step(self, delta=0.0):
        """Apply one training signal step. No-op by default."""
        return 0.0
