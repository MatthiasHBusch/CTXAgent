"""Launcher used by run.cmd / run.ps1: refuse a stale backend, open the browser, serve.

Lives in Python rather than only in run.ps1 so it also works where PowerShell
refuses unsigned scripts (execution policy AllSigned).

    run.cmd              server on :8765 + system browser (development)
    run.cmd --desktop    the native CTXAgent window (python -m ctxagent.desktop)
"""

from __future__ import annotations

import os
import socket
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # config is imported only after --data has been applied (it fixes DATA_DIR on import).
    from .desktop import main as desktop_main
    if argv[:1] == ["--desktop"]:
        return desktop_main(argv[1:])
    from .config import load_env
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
    server_args = ["--server", "--port", str(port)]
    if not os.environ.get("CTXAGENT_NO_BROWSER"):
        server_args.append("--open")
    return desktop_main(server_args + argv)


if __name__ == "__main__":
    sys.exit(main())
