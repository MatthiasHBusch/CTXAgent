#!/usr/bin/env bash
# Server on :8765 (development). Native window: python -m ctxagent.desktop
set -euo pipefail
cd "$(dirname "$0")"
python -m ctxagent "$@"
