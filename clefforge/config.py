"""Configuration model and loader (YAML with zero-dep fallback)."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .utils.yamlish import parse as _yaml_parse


@dataclass
class BackendConfig:
    name: str = "mock"
    model: str = "clef-flash"
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.0


@dataclass
class VisionConfig:
    enabled: bool = True
    patch_size: int = 16
    hidden: int = 128
    layers: int = 2
    max_patches: int = 512
    resolution: str = "dynamic"


@dataclass
class DecisionConfig:
    temperature: float = 1.0
    calibrate: bool = True
    gate_threshold: float = 0.5
    permutation: bool = True
    fast_slow: bool = True


@dataclass
class TrainConfig:
    steps: int = 40
    lr: float = 0.05
    reward_noise: float = 0.5
    algorithm: str = "grpo"


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: str = "clef-flash"
    context: int = 65536
    seed: int = 0

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(
            backend=BackendConfig(**d.get("backend", {})),
            vision=VisionConfig(**d.get("vision", {})),
            decision=DecisionConfig(**d.get("decision", {})),
            train=TrainConfig(**d.get("train", {})),
            model=d.get("model", "clef-flash"),
            context=int(d.get("context", 65536)),
            seed=int(d.get("seed", 0)),
        )


def load_config(path=None):
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # noqa: F401
        data = yaml.safe_load(text)
    except Exception:
        data = _yaml_parse(text)
    return Config.from_dict(data)
