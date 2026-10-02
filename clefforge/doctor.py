"""Component / backend / benchmark smoke check."""
from .config import Config
from .utils.logging import get_logger

log = get_logger("clefforge.doctor")


def run_doctor(cfg: Config):
    ok = True
    try:
        from .vision.encoder import VisionEncoder
        enc = VisionEncoder(cfg.vision)
        feats = enc.encode("doctor-check-image", n_patches=4)
        log.info("vision encoder OK (%d patches -> %d-d tokens)", feats.shape[0], feats.shape[1])
    except Exception as e:
        ok = False
        log.error("vision encoder FAILED: %s", e)

    try:
        from .backends import list_backends, get_backend
        names = list_backends()
        b = get_backend(cfg.backend.name)(cfg)
        log.info("backends OK (%s) -> %s", ", ".join(names), type(b).__name__)
    except Exception as e:
        ok = False
        log.error("backends FAILED: %s", e)

    try:
        from .decision.head import PointerHead
        from .decision.calibration import calibrate
        from .decision.gate import Gate
        from .decision.router import Router
        PointerHead(); calibrate([0.1, 0.3, 0.6], [1, 1, 1]); Gate(); Router()
        log.info("decision components OK (head/calibrate/gate/router)")
    except Exception as e:
        ok = False
        log.error("decision components FAILED: %s", e)

    try:
        from .bench import list_benchmarks
        log.info("benchmarks OK (%s)", ", ".join(list_benchmarks()))
    except Exception as e:
        ok = False
        log.error("benchmarks FAILED: %s", e)

    log.info("doctor %s", "PASS" if ok else "FAIL")
    return ok
