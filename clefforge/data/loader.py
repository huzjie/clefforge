"""Task loading helpers."""


def load_tasks(n_tasks=200, seed=0):
    from .synth import SyntheticDecisionData
    return SyntheticDecisionData(seed=seed, n_tasks=n_tasks).generate()
