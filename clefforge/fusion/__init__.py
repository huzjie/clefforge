"""Multimodal fusion of visual tokens into the language stream."""
from .multimodal import GatedFusion
from .cross_attention import CrossAttentionFusion
from .router import MoETokenRouter

__all__ = ["GatedFusion", "CrossAttentionFusion", "MoETokenRouter"]
