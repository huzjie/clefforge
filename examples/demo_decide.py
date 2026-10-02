"""Single decision demo (text + multimodal)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.config import load_config
from clefforge.engine import ClefEngine


def main():
    cfg = load_config()
    engine = ClefEngine(cfg)

    print("== text decision ==")
    r = engine.decide("给定任务，选择最合适的工具", ["web_search", "python_exec", "file_read"])
    for k in ("route", "answer", "confidence", "gate"):
        print(f"  {k}: {r[k]}")

    print("== multimodal decision ==")
    r = engine.decide("识别图像 img-7 中的物体类别", ["cat", "dog", "car", "bicycle"], image="img-7")
    for k in ("route", "answer", "confidence", "gate"):
        print(f"  {k}: {r[k]}")


if __name__ == "__main__":
    main()
