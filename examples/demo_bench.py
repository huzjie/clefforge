"""Run all benchmarks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.config import load_config
from clefforge.bench.runner import run_all


def main():
    run_all(load_config())


if __name__ == "__main__":
    main()
