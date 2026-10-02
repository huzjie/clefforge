"""Fast-slow routing demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.decision.router import Router


def main():
    r = Router(fast_slow=True)
    for q, opts, img in [
        ("短问题", ["a", "b"], None),
        ("需要图像识别、上下文很长的问题 " + "细节" * 60, ["x", "y"], "img-3"),
    ]:
        d = r.route(q, opts, image=img)
        print(f"model={d.model:12s} reason={d.reason}")


if __name__ == "__main__":
    main()
