"""HTTP client for the Jev-API-compatible server."""
import json
import urllib.request


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def decide(self, query, options, image=None):
        payload = json.dumps({"query": query, "options": options, "image": image}).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/decide", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self):
        with urllib.request.urlopen(self.base_url + "/health", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
