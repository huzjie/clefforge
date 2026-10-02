"""Option scoring helpers shared by backends."""

from ..core.ops import softmax


def score_options(logits, temperature=1.0):
    """Turn raw option logits into probabilities + a chosen index."""
    probs = softmax(logits, temperature=temperature)
    p = probs.data[0]
    best = max(range(len(p)), key=lambda i: p[i])
    return {"probs": p, "choice": best, "confidence": p[best]}
