"""A minimal decision-driven agent loop.

Demonstrates how a decision model slots into an agentic workflow: perceive state,
enumerate candidate actions, let Clef pick one with confidence, gate the result.
"""
from ..config import Config
from ..engine import ClefEngine


class DecisionAgent:
    def __init__(self, config=None, max_steps=10):
        self.engine = ClefEngine(config or Config())
        self.max_steps = max_steps

    def run(self, goal, actions, step_fn):
        """Run up to max_steps; step_fn(goal, state) -> (new_state, options, done)."""
        state = {}
        trace = []
        for _ in range(self.max_steps):
            options = actions
            r = self.engine.decide(goal, options)
            if r["gate"] == "reject":
                trace.append({"action": None, "reason": "rejected"})
                break
            trace.append({"action": r["answer"], "confidence": r["confidence"]})
            state, _, done = step_fn(goal, state, r["answer"])
            if done:
                break
        return trace
