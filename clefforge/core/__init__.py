"""Zero-dependency tensor core."""
from .tensor import Tensor, tensor, zeros, ones, randn
from .ops import softmax, layernorm, relu, gelu, silu, cross_entropy, accuracy

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn",
    "softmax", "layernorm", "relu", "gelu", "silu", "cross_entropy", "accuracy",
]
