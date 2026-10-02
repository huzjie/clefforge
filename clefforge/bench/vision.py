"""Visual discrimination benchmark (Clef's native-vision advantage).

Measures how much an image shifts the correct answer vs a text-only baseline,
demonstrating that the vision encoder genuinely participates in the decision.
"""
from ..world import world_answer
from .base import register_benchmark


@register_benchmark("vision_discrimination")
def vision_discrimination(engine):
    labels = ["cat", "dog", "car", "bicycle"]
    n = 60
    correct = 0
    changed = 0
    for i in range(n):
        img = f"img-{i % 37}"
        q = f"识别图像 {img} 中的物体类别。"
        text_answer = world_answer(q, labels, image=None)
        img_answer = world_answer(q, labels, image=img)
        if text_answer != img_answer:
            changed += 1
        r = engine.decide(q, labels, image=img)
        if r["choice"] == img_answer:
            correct += 1
    acc = correct / n
    return {"name": "vision_discrimination", "score": acc,
            "detail": f"{correct}/{n} correct; image shifted answer in {changed}/{n} cases"}
