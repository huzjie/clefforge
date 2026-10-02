"""GRPO / PPO for decision models.

Decision RL differs from text RL: the reward is a business signal (was the chosen
option correct / accepted / low-latency), and the policy is the pointer head. This
module computes group-relative advantages (GRPO) or clipped advantages (PPO) over
the option logits.
"""
import math


class GRPO:
    def __init__(self, group_size=4, lr=0.05):
        self.group_size = group_size
        self.lr = lr

    def advantage(self, rewards):
        mu = sum(rewards) / len(rewards)
        std = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards)) + 1e-6
        return [(r - mu) / std for r in rewards]


class PPO:
    def __init__(self, clip=0.2, lr=0.05):
        self.clip = clip
        self.lr = lr

    def advantage(self, rewards, old_probs=None):
        mu = sum(rewards) / len(rewards)
        return [r - mu for r in rewards]
