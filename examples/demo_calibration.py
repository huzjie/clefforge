"""Calibration demo: ECE before/after temperature scaling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.decision.calibration import expected_calibration_error


def main():
    # a confident-but-wrong-ish model: conf high, acc ~0.6
    confs = [0.9] * 6 + [0.6] * 4
    corrects = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ece = expected_calibration_error(confs, corrects)
    print(f"ECE = {ece:.4f}  (0 = perfectly calibrated)")
    print("lower ECE means confidence matches accuracy")


if __name__ == "__main__":
    main()
