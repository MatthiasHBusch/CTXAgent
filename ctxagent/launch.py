"""Launcher used by run.cmd / run.ps1: refuse a stale backend, open the browser, serve.

Lives in Python rather than only in run.ps1 so it also works where PowerShell
refuses unsigned scripts (execution policy AllSigned).
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser

from .config import load_env


def main() -> int:
    load_env()
    port = int(os.environ.get("CTXAGENT_PORT", "8765"))
    with socket.socket() as probe:
        probe.settimeout(1.0)
        occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    if occupied:
        print(f"Port {port} is already occupied. Opening the browser would reuse that OLD server.")
        print("Finish the active turn, press Ctrl+C in the existing server terminal, then run this launcher again.")
        print("Inspect backend identity with: python -B -m ctxagent.check_backend")
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"ctxagent on {url} (Ctrl+C to stop)")
    if not os.environ.get("CTXAGENT_NO_BROWSER"):
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    from .server import main as serve
    serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
