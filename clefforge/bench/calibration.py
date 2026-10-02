"""Calibration benchmark: ECE / Brier of the confidence outputs."""
from ..data.loader import load_tasks
from ..decision.calibration import expected_calibration_error, brier
from ..world import world_answer
from .base import register_benchmark


@register_benchmark("calibration")
def calibration(engine):
    tasks = load_tasks(n_tasks=120)
    confs, corrects = [], []
    brier_sum = 0.0
    for t in tasks:
        r = engine.decide(t["query"], t["options"], image=t["image"])
        confs.append(r["confidence"])
        corrects.append(1.0 if r["choice"] == world_answer(t["query"], t["options"], image=t["image"]) else 0.0)
        brier_sum += brier(r["probs"], world_answer(t["query"], t["options"], image=t["image"]))
    ece = expected_calibration_error(confs, corrects)
    # score = 1 - ECE (lower ECE is better)
    score = max(0.0, 1.0 - ece)
    return {"name": "calibration", "score": score,
            "detail": f"ECE={ece:.4f} Brier={brier_sum/len(tasks):.4f}"}
