"""DPO step: prefer the chosen option over a rejected one."""
import math


def dpo_step(chosen_logit, rejected_logit, beta=0.1):
    """Return a positive delta scaled by the margin between chosen/rejected."""
    margin = max(0.0, chosen_logit - rejected_logit)
    return {"delta": beta * (1.0 + margin), "margin": margin}
