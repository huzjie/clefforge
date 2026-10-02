"""Jev Decision Index style benchmark: decision accuracy across domains.

Mirrors the Jev Decision Index (0.2.1) Clef was evaluated on: a battery of option-
selection tasks across tool selection / model routing / guardrails / vision.
"""
from ..data.loader import load_tasks
from ..world import world_answer
from .base import register_benchmark


@register_benchmark("decision_index")
def decision_index(engine, n_tasks=120):
    tasks = load_tasks(n_tasks=n_tasks)
    correct = 0
    for t in tasks:
        r = engine.decide(t["query"], t["options"], image=t["image"])
        if r["choice"] == world_answer(t["query"], t["options"], image=t["image"]):
            correct += 1
    acc = correct / len(tasks)
    return {"name": "decision_index", "score": acc,
            "detail": f"{correct}/{len(tasks)} correct"}
