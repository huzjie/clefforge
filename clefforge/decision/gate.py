"""Gating: accept / reject / degrade based on confidence.

A decision model should refuse (or fall back to a stronger model) when its
confidence is low, rather than emit a confident wrong answer. The gate maps
confidence against a threshold to an action.
"""
from dataclasses import dataclass


@dataclass
class GateDecision:
    action: str          # "accept" | "degrade" | "reject"
    confidence: float


class Gate:
    def __init__(self, threshold=0.5, degrade_threshold=0.7):
        self.threshold = threshold
        self.degrade_threshold = degrade_threshold

    def apply(self, confidence):
        if confidence >= self.degrade_threshold:
            return GateDecision("accept", confidence)
        if confidence >= self.threshold:
            return GateDecision("degrade", confidence)
        return GateDecision("reject", confidence)
