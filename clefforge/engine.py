"""ClefEngine: the orchestration layer tying backend + vision + decision together.

One call to `decide` runs the full Clef pipeline:
route (fast/slow) -> score (backend) -> permutation (kill order bias)
-> calibrate (temperature) -> gate (accept/degrade/reject).
"""
from .backends import get_backend
from .config import Config
from .decision.calibration import temperature_scale
from .decision.gate import Gate
from .decision.permutation import permutation_score
from .decision.router import Router
from .model.config import get_model_config


class ClefEngine:
    def __init__(self, config=None):
        self.config = config or Config()
        backend_cls = get_backend(self.config.backend.name)
        self.backend = backend_cls(self.config)
        self.router = Router(fast_slow=self.config.decision.fast_slow)
        self.gate = Gate(threshold=self.config.decision.gate_threshold)

    def decide(self, query, options, image=None):
        route = self.router.route(query, options, image=image)
        raw = self.backend.decide(query, options, image=image)
        probs = raw["probs"]

        if self.config.decision.permutation and len(options) > 2:
            def score_fn(permuted):
                return self.backend.score(query, permuted, image=image)
            p = permutation_score(score_fn, options)
            # remap: rebuild probs from permutation scores via softmax
            from .core.ops import softmax
            probs = softmax(p["scores"]).data[0]

        if self.config.decision.calibrate:
            probs = temperature_scale(probs, self.config.decision.temperature)

        best = max(range(len(probs)), key=lambda i: probs[i])
        conf = probs[best]
        g = self.gate.apply(conf)

        return {
            "query": query,
            "options": options,
            "image": image,
            "route": route.model,
            "route_reason": route.reason,
            "choice": best,
            "answer": options[best],
            "confidence": conf,
            "probs": probs,
            "gate": g.action,
            "model_tier": get_model_config(route.model)["params"],
        }
