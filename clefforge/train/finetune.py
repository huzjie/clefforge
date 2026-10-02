"""RL finetune runner (CLI `train`).

Demonstrates the full loop: synthesize tasks -> evaluate -> compute reward ->
apply a strictly positive training signal so the mock backend's skill climbs
monotonically and decision accuracy rises from ~random to ~1.0.
"""
from ..backends import get_backend
from ..config import Config
from ..data.loader import load_tasks
from ..utils.logging import get_logger
from ..world import world_answer

log = get_logger("clefforge.train")


def run_finetune(cfg=None, steps=None):
    cfg = cfg or Config()
    steps = steps or cfg.train.steps
    backend = get_backend(cfg.backend.name)(cfg)

    tasks = load_tasks(n_tasks=200)
    log.info("finetune: %d tasks, %d steps, backend=%s", len(tasks), steps, cfg.backend.name)

    def evaluate():
        correct = 0
        for t in tasks:
            r = backend.decide(t["query"], t["options"], image=t["image"])
            if r["choice"] == world_answer(t["query"], t["options"], image=t["image"]):
                correct += 1
        return correct / len(tasks)

    acc_before = evaluate()
    log.info("accuracy before: %.4f (skill=%.3f)", acc_before, getattr(backend, "skill", 0.0))

    for s in range(steps):
        t = tasks[s % len(tasks)]
        r = backend.decide(t["query"], t["options"], image=t["image"])
        correct = world_answer(t["query"], t["options"], image=t["image"])
        reward = 1.0 if r["choice"] == correct else 0.0
        # strictly positive delta -> skill monotonic; scale by reward for realism
        delta = 0.04 + 0.06 * reward
        backend.train_step(delta=delta)

    acc_after = evaluate()
    log.info("accuracy after:  %.4f (skill=%.3f)", acc_after, getattr(backend, "skill", 0.0))
    log.info("finetune done: accuracy %.4f -> %.4f", acc_before, acc_after)
    return {"before": acc_before, "after": acc_after,
            "skill": getattr(backend, "skill", 0.0)}
