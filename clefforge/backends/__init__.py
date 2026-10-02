"""Backend registry. Every backend exposes `decide` / `score` / `train_step`."""
from .base import Backend
from .registry import register_backend, list_backends, get_backend

# import all backends so their @register_backend decorators run
from . import mock, cpu, openai, vllm, transformers  # noqa: F401

__all__ = ["Backend", "register_backend", "list_backends", "get_backend"]
