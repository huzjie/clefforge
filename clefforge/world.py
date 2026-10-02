"""Deterministic synthetic world model: the ground-truth answer for any decision.

The mock backend and every benchmark share this single source of truth, so
"accuracy" is well-defined and reproducible. A decision is just: given a query
(and optionally an image), which option is correct. This mirrors how a real
decision model is trained against business labels.
"""
from .utils.stable import stable_ints, stable_float


def world_answer(query, options, image=None):
    """Return the index of the correct option, deterministically."""
    n = len(options)
    if n == 0:
        raise ValueError("empty options")
    key = f"world:{query}" + (f":{image}" if image else "")
    # seed with the query + image so the "correct" answer is stable but sensitive
    # to both modalities: providing an image genuinely shifts the ground truth.
    idx = stable_ints(key, 1, 0, n - 1)[0]
    return idx


def world_truth_score(query, option, image=None, options=None):
    """Continuous truth value (0/1) for a single option vs the world answer."""
    if options is None:
        return 0.5
    idx = world_answer(query, options, image=image)
    return 1.0 if options.index(option) == idx else 0.0
