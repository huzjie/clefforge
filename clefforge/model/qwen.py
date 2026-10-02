"""Qwen backbone (structural reference; real weights come from a backend).

This module describes the Qwen decoder-only architecture the Clef heads attach to:
causal self-attention, SwiGLU FFN, RMSNorm, RoPE. The mock backend does not
instantiate the full weights; it models the backbone's hidden stream through the
Backend interface instead.
"""


class QwenConfig:
    def __init__(self, hidden=256, layers=12, heads=8, vocab=151936):
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.vocab = vocab
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1000000.0


class QwenBackbone:
    """Abstract backbone descriptor used by configs and docs."""
    def __init__(self, cfg=None):
        self.cfg = cfg or QwenConfig()
