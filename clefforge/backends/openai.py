"""OpenAI-compatible backend (Jev-API / any OpenAI-compatible decision endpoint).

When an api_base + api_key are configured, `decide` POSTs a decision request to
the remote endpoint; otherwise it transparently falls back to the mock backend so
the CLI stays runnable out of the box.
"""
import json
import urllib.request

from .base import Backend
from .mock import MockBackend
from .registry import register_backend


@register_backend("openai")
class OpenAIBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        self.api_base = config.backend.api_base
        self.api_key = config.backend.api_key
        self.model = config.backend.model
        self._fallback = MockBackend(config)

    def _remote(self, query, options, image=None):
        payload = {"model": self.model, "query": query, "options": options, "image": image}
        req = urllib.request.Request(
            self.api_base.rstrip("/") + "/decide",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def decide(self, query, options, image=None):
        if self.api_base and self.api_key:
            return self._remote(query, options, image=image)
        return self._fallback.decide(query, options, image=image)

    def score(self, query, options, image=None):
        return self.decide(query, options, image=image).get("logits", [])
