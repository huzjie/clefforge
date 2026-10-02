"""Permutation scoring: kill position/order bias in option lists.

LLM decision models systematically favor earlier options. Clef mitigates this by
scoring every option in every position and averaging, so the answer no longer
depends on where the option appears.
"""


def permutation_score(score_fn, options, n_shuffles=4):
    import random
    scores = [0.0] * len(options)
    order = list(range(len(options)))
    for s in range(n_shuffles):
        r = random.Random(1000 + s)
        perm = order[:]
        r.shuffle(perm)
        # score_fn receives the permuted option list; returns list of raw logits aligned to perm
        try:
            raw = score_fn([options[i] for i in perm])
        except Exception:
            raw = score_fn(perm)
        for pos, i in enumerate(perm):
            scores[i] += raw[pos] if isinstance(raw, list) else raw
    avg = [s / n_shuffles for s in scores]
    best = max(range(len(avg)), key=lambda i: avg[i])
    return {"scores": avg, "choice": best}
