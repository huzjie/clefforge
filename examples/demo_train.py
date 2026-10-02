"""RL finetune demo: accuracy climbs from near-random to 1.0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.config import load_config
from clefforge.train.finetune import run_finetune


def main():
    cfg = load_config()
    res = run_finetune(cfg, steps=40)
    print(f"accuracy {res['before']:.4f} -> {res['after']:.4f} (skill={res['skill']:.3f})")


if __name__ == "__main__":
    main()
