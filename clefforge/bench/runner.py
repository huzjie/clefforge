"""Run all benchmarks and print a summary."""
from ..config import Config
from ..engine import ClefEngine
from ..utils.logging import get_logger
from .base import list_benchmarks, run_benchmark

log = get_logger("clefforge.bench")


def run_all(cfg=None):
    cfg = cfg or Config()
    engine = ClefEngine(cfg)
    results = []
    for name in list_benchmarks():
        r = run_benchmark(name, engine)
        results.append(r)
        log.info("bench %-22s score=%.4f  (%s)", r["name"], r["score"], r["detail"])
    avg = sum(r["score"] for r in results) / max(len(results), 1)
    log.info("bench AVG = %.4f", avg)
    return results
