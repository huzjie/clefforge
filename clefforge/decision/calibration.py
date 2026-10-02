"""Confidence calibration for decision models.

Decision models output a max-probability "confidence" that is often overconfident.
Temperature scaling fits a single scalar T to align confidence with accuracy;
ECE / Brier quantify the gap. Permutation scoring (see permutation.py) removes
order bias before calibration.
"""
import math


def temperature_scale(probs, temperature):
    p = [v ** (1.0 / temperature) for v in probs]
    s = sum(p) or 1e-12
    return [x / s for x in p]


def expected_calibration_error(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, a in zip(confs, corrects):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, a))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(a for _, a in b) / len(b)
        ece += (len(b) / max(len(confs), 1)) * abs(avg_conf - avg_acc)
    return ece


def brier(probs, target_idx):
    return sum((p - (1.0 if i == target_idx else 0.0)) ** 2 for i, p in enumerate(probs))


def calibrate(confs, corrects):
    """Fit temperature by simple grid search minimizing ECE."""
    best_t, best_ece = 1.0, expected_calibration_error(confs, corrects)
    for t in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        # re-normalized probs under T, recompute ECE proxy
        e = _ece_temperature(confs, corrects, t)
        if e < best_ece:
            best_ece, best_t = e, t
    return best_t


def _ece_temperature(confs, corrects, t, n_bins=10):
    scaled = [min(1.0, c ** (1.0 / t)) for c in confs]
    bins = [[] for _ in range(n_bins)]
    for c, a in zip(scaled, corrects):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, a))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(a for _, a in b) / len(b)
        ece += (len(b) / max(len(confs), 1)) * abs(avg_conf - avg_acc)
    return ece
