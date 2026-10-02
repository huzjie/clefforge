"""Benchmark suite."""
from .base import Benchmark, Result, list_benchmarks, run_benchmark
from .runner import run_all

# import all benchmarks so their @register_benchmark decorators run
from . import decision_index, vision, routing, calibration  # noqa: F401

__all__ = ["run_all", "Benchmark", "list_benchmarks"]
