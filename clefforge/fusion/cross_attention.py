"""Cross-attention fusion: let text tokens attend to visual tokens.

Q/K/V over visual tokens, queried by textual tokens, to pull the relevant visual
signal into the language stream before the pointer head scores options.
"""
from ..core.ops import softmax
from ..core.tensor import Tensor


class CrossAttentionFusion:
    def __init__(self, dim=256):
        self.dim = dim

    def attend(self, query, keys, values):
        # query: [q_dim]; keys/values: [n_tokens, dim]; returns attended vector
        scores = [sum(q * k for q, k in zip(query, key)) for key in keys]
        w = softmax(Tensor([scores])).data[0]
        out = [sum(w[i] * values[i][j] for i in range(len(values))) for j in range(len(values[0]))]
        return out
