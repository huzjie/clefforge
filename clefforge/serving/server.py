"""Jev-API-compatible stdlib HTTP server.

Exposes the decision model over HTTP with zero external dependencies
(`http.server`). The `/decide` endpoint accepts `{model, query, options, image}`
and returns `{choice, confidence, probs, gate, route}` -- the same shape as the
Jev decision API, so existing Jev clients can be pointed at it unchanged.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..config import Config
from ..engine import ClefEngine
from ..utils.logging import get_logger

log = get_logger("clefforge.serve")


def serve(cfg=None, host="127.0.0.1", port=8000):
    cfg = cfg or Config()
    engine = ClefEngine(cfg)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "model": cfg.model})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/decide":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as e:
                return self._send(400, {"error": f"bad json: {e}"})
            query = data.get("query", "")
            options = data.get("options", [])
            image = data.get("image")
            if not query or not options:
                return self._send(422, {"error": "query and options required"})
            result = engine.decide(query, options, image=image)
            return self._send(200, result)

        def log_message(self, *args):
            pass

    server = HTTPServer((host, port), Handler)
    log.info("Jev-API-compatible server on http://%s:%s (POST /decide)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
