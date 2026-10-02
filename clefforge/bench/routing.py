"""Fast-slow routing benchmark: does the router pick the right tier."""
from .base import register_benchmark


@register_benchmark("routing")
def routing(engine):
    cases = [
        ("短问题，两个选项", ["a", "b"], None, "clef-flash"),
        ("一个需要图像识别、上下文很长的问题 " + "详细背景" * 60, ["x", "y"], "img-1", "clef"),
    ]
    correct = 0
    for q, opts, img, want in cases:
        d = engine.router.route(q, opts, image=img)
        if d.model == want:
            correct += 1
    acc = correct / len(cases)
    return {"name": "routing", "score": acc, "detail": f"{correct}/{len(cases)} correct"}
