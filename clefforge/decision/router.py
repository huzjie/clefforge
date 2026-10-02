"""Fast-slow routing: pick Clef-flash vs Clef by task difficulty.

Cheap tasks (short query, few options, deterministic) go to the 9B flash model;
hard tasks (long context, ambiguous, image-heavy) go to the 27B full model.
"""
from dataclasses import dataclass


@dataclass
class RouteDecision:
    model: str           # "clef" | "clef-flash"
    reason: str


class Router:
    def __init__(self, fast_slow=True):
        self.fast_slow = fast_slow

    def route(self, query, options, image=None):
        if not self.fast_slow:
            return RouteDecision("clef", "routing disabled")
        difficulty = _estimate_difficulty(query, options, image)
        if difficulty < 3:
            return RouteDecision("clef-flash", "low difficulty")
        if difficulty < 6:
            return RouteDecision("clef", "medium difficulty")
        return RouteDecision("clef", "high difficulty (image / long context)")


def _estimate_difficulty(query, options, image=None):
    score = 0
    score += min(len(query) // 120, 3)
    score += min(len(options), 4)
    if image:
        score += 2
    # ambiguous wording raises difficulty
    for w in ("或者", "可能", "maybe", "either", "不确定"):
        if w in query:
            score += 1
    return score
