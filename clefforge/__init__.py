"""clefforge: multimodal decision-model training & inference framework.

A zero-dependency, fully runnable reference implementation inspired by Cloudflare's
open-sourced Clef multimodal decision model (2026-10-02). Clef replaces the language
head of a Qwen backbone with a pointer head that scores candidate options -- a
"system one" decision model that picks an option and returns a confidence instead of
generating free text. Unlike Jev (text-only), Clef carries a native vision encoder
and a 64k context window.

Highlights
----------
- Pointer head   -- scores a fixed set of options instead of generating tokens
- Native vision  -- ViT patch encoder + arbitrary-resolution tiling + projector
- Multimodal fusion -- gated fusion + cross-attention + MoE routing of visual tokens
- Calibration    -- temperature scaling + permutation scoring to kill order bias
- Gating         -- reject / degrade to the slow model when confidence is low
- Fast-slow routing -- Clef (27B) vs Clef-flash (9B) by task difficulty
- RL finetune    -- GRPO / PPO / DPO against business reward signals
- Backends       -- deterministic trainable mock, CPU, OpenAI, vLLM, Transformers
- Serving        -- Jev-API-compatible stdlib HTTP server + CLI + Docker/K8s/Helm/CI
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
