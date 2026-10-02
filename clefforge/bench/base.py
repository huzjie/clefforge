"""Benchmark base + registry."""
from dataclasses import dataclass, field

_BENCHMARKS = {}


def register_benchmark(name):
    def _wrap(fn):
        _BENCHMARKS[name] = fn
        return fn
    return _wrap


def list_benchmarks():
    return sorted(_BENCHMARKS.keys())


class Benchmark:
    """Marker base class for benchmark implementations."""


@dataclass
class Result:
    name: str
    score: float
    detail: str = ""


def run_benchmark(name, engine):
    if name not in _BENCHMARKS:
        raise KeyError(name)
    return _BENCHMARKS[name](engine)
