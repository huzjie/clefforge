"""Start a server in a thread, call it via the client."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clefforge.config import load_config
from clefforge.serving.server import serve
from clefforge.serving.client import Client


def main():
    t = threading.Thread(target=serve, args=(load_config(), "127.0.0.1", 8765), daemon=True)
    t.start()
    time.sleep(1)
    c = Client("http://127.0.0.1:8765")
    print("health:", c.health())
    r = c.decide("选择工具", ["web_search", "python_exec"])
    print("decide:", {k: r[k] for k in ("answer", "confidence", "gate")})


if __name__ == "__main__":
    main()
