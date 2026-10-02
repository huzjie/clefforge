"""Model registry (mirrors Clef family sizes and capabilities)."""

MODEL_REGISTRY = {
    "clef": {
        "base": "Qwen3.8-27B",
        "params": "27B",
        "context": 65536,
        "vision": True,
        "tier": "slow",
    },
    "clef-flash": {
        "base": "Qwen3.5-9B",
        "params": "9B",
        "context": 65536,
        "vision": True,
        "tier": "fast",
    },
}


def get_model_config(name):
    return MODEL_REGISTRY.get(name, MODEL_REGISTRY["clef-flash"])
