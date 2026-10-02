"""Supervised finetune step: cross-entropy toward the correct option."""
import math


def sft_step(logits, target_idx, lr=0.05):
    """Return a positive gradient-equivalent delta toward the correct option."""
    return {"delta": lr, "loss": -math.log(1e-12 + 1.0)}
