"""MoE routing of visual tokens: send each patch token to top-k experts.

Clef routes dense visual patches through a small MoE so only a subset of experts
activate per token (compute stays flat while capacity grows).
"""
from ..utils.stable import stable_ints


class MoETokenRouter:
    def __init__(self, n_experts=8, top_k=2):
        self.n_experts = n_experts
        self.top_k = top_k

    def route(self, token_idx, n_tokens):
        # deterministic top-k expert selection per token
        experts = stable_ints(f"moe:{token_idx}", n_tokens * self.top_k, 0, self.n_experts - 1)
        return experts[:self.top_k]

    def load_balance(self, n_tokens):
        counts = [0] * self.n_experts
        for i in range(n_tokens):
            for e in self.route(i, n_tokens):
                counts[e] += 1
        return counts
