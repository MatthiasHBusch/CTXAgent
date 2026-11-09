"""`python -m ctxagent`: the server alone on 127.0.0.1:$CTXAGENT_PORT (default 8765).

The native window is `python -m ctxagent.desktop`; see DESKTOP_APP.md.
"""
import sys

from .desktop import serve_main

if __name__ == "__main__":
    sys.exit(serve_main())
